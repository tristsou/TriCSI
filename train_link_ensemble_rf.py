from __future__ import annotations

"""
TriCSI two-link late-fusion RF, adapted from the spatially distributed TX-RX
ensemble idea in Fahad, Touhiduzzaman & Bulut (ICNC 2025).

This is deliberately ONE focused next step using EXISTING S4/S5/S6 data.

Architecture
------------
All engineered features
    -> RF EMPTY vs OCCUPIED

If occupied:
    LEFT-link features  -> RF_L -> p_L(Z1..Z9)
    RIGHT-link features -> RF_R -> p_R(Z1..Z9)

Class-specific link reliability is estimated ONLY from source-session
out-of-fold predictions.  For each zone, LEFT/RIGHT F1 scores are converted
to softmax weights.  The weighted probabilities are stacked and a small
logistic-regression combiner is trained on those same source-only OOF
predictions.

Exact zone:
    argmax(fused probability)

Continuous location:
    sum_z p(z) * zone_center(z)

Evaluation
----------
Leave-one-session-out on the already-development sessions S4/S5/S6:
    S5+S6 -> S4
    S4+S6 -> S5
    S4+S5 -> S6

A plain combined-feature RF is evaluated in the same splits as the control.

NO new data collection is required.
NO S7 is used.
NO threshold sweep / hyperparameter search is performed.

At the end, a final ensemble is trained on S4+S5+S6 and saved.

Run:
    python train_link_ensemble_rf.py
"""

import argparse
import csv
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from project_config import (
    CLASS_ORDER,
    RAW_DATA_DIR,
    RESULTS_DIR,
    MODEL_DIR,
    WINDOW_SECONDS,
    HOP_SECONDS,
)
from train_crossval import gather_windows, make_rf, trial_number_from_row
from train_csi_cnn import trial_paths_for_session


SESSIONS = ("S4", "S5", "S6")
OCCUPIED_CLASSES = tuple(z for z in CLASS_ORDER if z != "EMPTY")

ZONE_CENTERS = {
    "Z1": (0.4064, 1.5240),
    "Z2": (1.1176, 1.5240),
    "Z3": (1.8288, 1.5240),
    "Z4": (0.4064, 0.9144),
    "Z5": (1.1176, 0.9144),
    "Z6": (1.8288, 0.9144),
    "Z7": (0.4064, 0.3048),
    "Z8": (1.1176, 0.3048),
    "Z9": (1.8288, 0.3048),
}

GRID_POS = {
    "Z1": (0, 0), "Z2": (1, 0), "Z3": (2, 0),
    "Z4": (0, 1), "Z5": (1, 1), "Z6": (2, 1),
    "Z7": (0, 2), "Z8": (1, 2), "Z9": (2, 2),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_rows(raw_dir: Path, session: str, window: float, hop: float):
    paths = trial_paths_for_session(raw_dir, session)
    if not paths:
        raise SystemExit(f"No trial files found for {session}")

    rows = gather_windows(
        paths,
        window_seconds=window,
        hop_seconds=hop,
    )
    for r in rows:
        r["_session"] = session

    print(f"{session}: {len(paths)} trials, {len(rows)} windows")
    return rows


def matrix(rows, feature_names):
    X = np.asarray(
        [[float(r["features"][n]) for n in feature_names] for r in rows],
        dtype=np.float64,
    )
    y = np.asarray([str(r["zone"]) for r in rows], dtype=object)
    return X, y


def trial_group(r):
    return (
        f"{r['_session']}|{r['zone']}|"
        f"{trial_number_from_row(r)}|{r['orientation']}"
    )


# ---------------------------------------------------------------------------
# Robust link-feature discovery
# ---------------------------------------------------------------------------

def token_present(name: str, tokens: tuple[str, ...]) -> bool:
    n = name.lower()
    return any(re.search(p, n) is not None for p in tokens)


LEFT_PATTERNS = (
    r"^left(?:_|-|\.|$)",
    r"^l(?:_|-|\.|$)",
    r"(?:_|-|\.)(?:left)(?:_|-|\.|$)",
    r"(?:_|-|\.)(?:l)$",
)

RIGHT_PATTERNS = (
    r"^right(?:_|-|\.|$)",
    r"^r(?:_|-|\.|$)",
    r"(?:_|-|\.)(?:right)(?:_|-|\.|$)",
    r"(?:_|-|\.)(?:r)$",
)

# Only explicit cross-link markers.  Do NOT classify every feature containing
# "diff" as cross-link because a single link can have temporal/subcarrier diffs.
CROSS_PATTERNS = (
    r"^cross(?:_|-|\.|$)",
    r"^x(?:_|-|\.|$)",
    r"^lr(?:_|-|\.|$)",
    r"^rl(?:_|-|\.|$)",
    r"left[_\-.]?right",
    r"right[_\-.]?left",
    r"(?:_|-|\.)(?:lr|rl)(?:_|-|\.|$)",
)


def split_features(feature_names):
    left, right, cross, other = [], [], [], []

    for name in feature_names:
        has_l = token_present(name, LEFT_PATTERNS)
        has_r = token_present(name, RIGHT_PATTERNS)
        has_x = token_present(name, CROSS_PATTERNS)

        if has_x or (has_l and has_r):
            cross.append(name)
        elif has_l:
            left.append(name)
        elif has_r:
            right.append(name)
        else:
            other.append(name)

    return left, right, cross, other


def validate_link_features(left, right, feature_names):
    print("\nFEATURE DISCOVERY")
    print(f"  total : {len(feature_names)}")
    print(f"  LEFT  : {len(left)}")
    print(f"  RIGHT : {len(right)}")

    print("\n  LEFT examples:")
    for x in left[:12]:
        print("   ", x)

    print("\n  RIGHT examples:")
    for x in right[:12]:
        print("   ", x)

    if len(left) < 20 or len(right) < 20:
        raise SystemExit(
            "\nCould not safely identify enough LEFT/RIGHT engineered features.\n"
            "The script intentionally stops instead of silently mixing links.\n"
            "Send the printed feature examples/inventory and the selector can be "
            "adjusted in one line."
        )


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def align_proba(model, X, ordered_classes):
    raw = model.predict_proba(X)
    lookup = {str(c): i for i, c in enumerate(model.classes_)}
    out = np.zeros((len(X), len(ordered_classes)), dtype=np.float64)

    for j, c in enumerate(ordered_classes):
        if c in lookup:
            out[:, j] = raw[:, lookup[c]]

    s = out.sum(axis=1, keepdims=True)
    return np.divide(
        out,
        s,
        out=np.full_like(out, 1.0 / len(ordered_classes)),
        where=s > 0,
    )


def softmax_across_links(f1_left, f1_right):
    # shape = [2 links, C classes]
    a = np.vstack([f1_left, f1_right]).astype(np.float64)
    a = a - np.max(a, axis=0, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=0, keepdims=True)


def weighted_xy(zone_probs):
    centers = np.asarray(
        [ZONE_CENTERS[z] for z in OCCUPIED_CLASSES],
        dtype=np.float64,
    )
    return zone_probs @ centers


def physical_error(true_zone, xy):
    if true_zone == "EMPTY":
        return None
    return float(
        np.linalg.norm(
            np.asarray(ZONE_CENTERS[true_zone], dtype=np.float64)
            - np.asarray(xy, dtype=np.float64)
        )
    )


def adjacent(true_zone, pred_zone):
    if true_zone == "EMPTY":
        return int(pred_zone == "EMPTY")
    if pred_zone == "EMPTY":
        return 0

    tx, ty = GRID_POS[true_zone]
    px, py = GRID_POS[pred_zone]
    return int(max(abs(tx - px), abs(ty - py)) <= 1)


# ---------------------------------------------------------------------------
# Source-only OOF calibration for the late fusion
# ---------------------------------------------------------------------------

def source_oof_predictions(
    rows_by_session,
    train_sessions,
    left_features,
    right_features,
    seed,
):
    """
    For every source session, train on the OTHER source session(s) and predict
    this held-out source session.  These predictions are used only to learn
    link reliability weights and the logistic stacker.

    This avoids using the outer test session to choose fusion weights.
    """
    all_y = []
    all_left = []
    all_right = []

    for k, val_session in enumerate(train_sessions):
        inner_train_sessions = [s for s in train_sessions if s != val_session]

        inner_train = []
        for s in inner_train_sessions:
            inner_train.extend(rows_by_session[s])

        inner_val = rows_by_session[val_session]

        train_occ = [r for r in inner_train if str(r["zone"]) != "EMPTY"]
        val_occ = [r for r in inner_val if str(r["zone"]) != "EMPTY"]

        XLtr, ytr = matrix(train_occ, left_features)
        XRtr, _ = matrix(train_occ, right_features)
        XLv, yv = matrix(val_occ, left_features)
        XRv, _ = matrix(val_occ, right_features)

        ml = make_rf(seed + 10 * k + 1)
        mr = make_rf(seed + 10 * k + 2)

        ml.fit(XLtr, ytr)
        mr.fit(XRtr, ytr)

        all_y.append(yv)
        all_left.append(align_proba(ml, XLv, OCCUPIED_CLASSES))
        all_right.append(align_proba(mr, XRv, OCCUPIED_CLASSES))

    y = np.concatenate(all_y)
    p_left = np.vstack(all_left)
    p_right = np.vstack(all_right)

    pred_left = np.asarray(
        [OCCUPIED_CLASSES[i] for i in np.argmax(p_left, axis=1)],
        dtype=object,
    )
    pred_right = np.asarray(
        [OCCUPIED_CLASSES[i] for i in np.argmax(p_right, axis=1)],
        dtype=object,
    )

    f1_left = f1_score(
        y,
        pred_left,
        labels=OCCUPIED_CLASSES,
        average=None,
        zero_division=0,
    )
    f1_right = f1_score(
        y,
        pred_right,
        labels=OCCUPIED_CLASSES,
        average=None,
        zero_division=0,
    )

    weights = softmax_across_links(f1_left, f1_right)

    pw_left = p_left * weights[0][None, :]
    pw_right = p_right * weights[1][None, :]

    stack_X = np.hstack([pw_left, pw_right])

    stacker = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed + 500,
    )
    stacker.fit(stack_X, y)

    return {
        "f1_left": f1_left,
        "f1_right": f1_right,
        "weights": weights,
        "stacker": stacker,
    }


# ---------------------------------------------------------------------------
# Fit / predict ensemble
# ---------------------------------------------------------------------------

def fit_ensemble(
    rows_by_session,
    train_sessions,
    all_features,
    left_features,
    right_features,
    seed,
):
    train_rows = []
    for s in train_sessions:
        train_rows.extend(rows_by_session[s])

    # Stage 1: occupancy gate, using the full existing engineered representation.
    Xall, yall = matrix(train_rows, all_features)
    ybin = np.where(yall == "EMPTY", "EMPTY", "OCCUPIED")

    occupancy_model = make_rf(seed)
    occupancy_model.fit(Xall, ybin)

    # Source-only OOF reliability/stacking.
    calibration = source_oof_predictions(
        rows_by_session,
        train_sessions,
        left_features,
        right_features,
        seed + 1000,
    )

    # Final per-link occupied-zone models on all source data.
    occ = [r for r in train_rows if str(r["zone"]) != "EMPTY"]

    XL, yz = matrix(occ, left_features)
    XR, _ = matrix(occ, right_features)

    left_model = make_rf(seed + 2001)
    right_model = make_rf(seed + 2002)

    left_model.fit(XL, yz)
    right_model.fit(XR, yz)

    return {
        "occupancy_model": occupancy_model,
        "left_model": left_model,
        "right_model": right_model,
        "stacker": calibration["stacker"],
        "f1_left": calibration["f1_left"],
        "f1_right": calibration["f1_right"],
        "weights": calibration["weights"],
    }


def predict_ensemble(
    bundle,
    rows,
    all_features,
    left_features,
    right_features,
    empty_threshold,
):
    Xall, y = matrix(rows, all_features)
    XL, _ = matrix(rows, left_features)
    XR, _ = matrix(rows, right_features)

    # EMPTY probability.
    occ_model = bundle["occupancy_model"]
    raw_occ = occ_model.predict_proba(Xall)
    occ_lookup = {str(c): i for i, c in enumerate(occ_model.classes_)}
    p_empty = raw_occ[:, occ_lookup["EMPTY"]]

    # Per-link zone probabilities.
    p_left = align_proba(bundle["left_model"], XL, OCCUPIED_CLASSES)
    p_right = align_proba(bundle["right_model"], XR, OCCUPIED_CLASSES)

    w = bundle["weights"]
    pw_left = p_left * w[0][None, :]
    pw_right = p_right * w[1][None, :]

    stack_X = np.hstack([pw_left, pw_right])
    p_stack = align_proba(bundle["stacker"], stack_X, OCCUPIED_CLASSES)

    # Paper-adapted final aggregation:
    # weighted link predictions + learned stacking prediction.
    p_zone = pw_left + pw_right + p_stack
    p_zone /= p_zone.sum(axis=1, keepdims=True)

    hard_zone = np.asarray(
        [OCCUPIED_CLASSES[i] for i in np.argmax(p_zone, axis=1)],
        dtype=object,
    )

    pred = np.where(p_empty >= empty_threshold, "EMPTY", hard_zone)
    xy = weighted_xy(p_zone)

    return {
        "y": y,
        "pred": pred,
        "p_empty": p_empty,
        "p_zone": p_zone,
        "xy": xy,
        "p_left": p_left,
        "p_right": p_right,
    }


# ---------------------------------------------------------------------------
# Trial metrics
# ---------------------------------------------------------------------------

def evaluate_predictions(rows, pred_pack, empty_threshold):
    y = pred_pack["y"]
    pred = pred_pack["pred"]

    window_acc = float(accuracy_score(y, pred))
    window_f1 = float(
        f1_score(
            y,
            pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )
    )

    groups = np.asarray([trial_group(r) for r in rows], dtype=object)

    trial_rows = []
    true_trial = []
    pred_trial = []
    occupied_errors = []

    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        true_zone = str(y[idx[0]])

        mean_empty = float(np.mean(pred_pack["p_empty"][idx]))
        mean_zone = np.mean(pred_pack["p_zone"][idx], axis=0)
        mean_zone /= mean_zone.sum()

        hard_zone = OCCUPIED_CLASSES[int(np.argmax(mean_zone))]
        final_zone = "EMPTY" if mean_empty >= empty_threshold else hard_zone

        xy = weighted_xy(mean_zone[None, :])[0]
        err = physical_error(true_zone, xy)

        if err is not None:
            occupied_errors.append(err)

        sample = rows[int(idx[0])]

        trial_rows.append(
            {
                "trial_group": g,
                "true_zone": true_zone,
                "pred_zone": final_zone,
                "trial": int(trial_number_from_row(sample)),
                "orientation": str(sample["orientation"]),
                "p_empty": mean_empty,
                "x_m": float(xy[0]),
                "y_m": float(xy[1]),
                "physical_error_m": err,
                "physical_error_ft": None if err is None else err * 3.28084,
                "correct": int(final_zone == true_zone),
                "exact_or_adjacent": adjacent(true_zone, final_zone),
            }
        )

        true_trial.append(true_zone)
        pred_trial.append(final_zone)

    true_trial = np.asarray(true_trial, dtype=object)
    pred_trial = np.asarray(pred_trial, dtype=object)

    occupied_trial_rows = [r for r in trial_rows if r["true_zone"] != "EMPTY"]
    errors = np.asarray(occupied_errors, dtype=float)

    return {
        "window_accuracy": window_acc,
        "window_macro_f1": window_f1,
        "trial_accuracy": float(accuracy_score(true_trial, pred_trial)),
        "trial_macro_f1": float(
            f1_score(
                true_trial,
                pred_trial,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "adjacent_accuracy_occupied": float(
            np.mean([r["exact_or_adjacent"] for r in occupied_trial_rows])
        ),
        "median_xy_error_m": float(np.median(errors)),
        "p90_xy_error_m": float(np.percentile(errors, 90)),
        "trial_rows": trial_rows,
    }


# ---------------------------------------------------------------------------
# Plain RF control
# ---------------------------------------------------------------------------

def evaluate_plain_control(train_rows, test_rows, all_features, seed):
    Xtr, ytr = matrix(train_rows, all_features)
    Xte, yte = matrix(test_rows, all_features)

    model = make_rf(seed)
    model.fit(Xtr, ytr)

    prob = align_proba(model, Xte, CLASS_ORDER)

    groups = np.asarray([trial_group(r) for r in test_rows], dtype=object)
    yt, yp = [], []

    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        mean_prob = prob[idx].mean(axis=0)
        yt.append(str(yte[idx[0]]))
        yp.append(CLASS_ORDER[int(np.argmax(mean_prob))])

    return float(accuracy_score(yt, yp))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return

    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--window", type=float, default=WINDOW_SECONDS)
    p.add_argument("--hop", type=float, default=HOP_SECONDS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--empty-threshold", type=float, default=0.5)
    args = p.parse_args()

    print("\n" + "=" * 78)
    print("TriCSI TWO-LINK LATE-FUSION RF")
    print("paper-guided, one focused development step")
    print("=" * 78)
    print("Data: S4/S5/S6 only")
    print("New trials required: NO")
    print("Hyperparameter sweep: NO")
    print("Outer evaluation: leave-one-session-out")
    print("=" * 78)

    rows_by_session = {
        s: load_rows(args.raw_dir, s, args.window, args.hop)
        for s in SESSIONS
    }

    all_features = sorted(rows_by_session["S4"][0]["features"])

    for s in ("S5", "S6"):
        if sorted(rows_by_session[s][0]["features"]) != all_features:
            raise RuntimeError(f"{s} feature set does not match S4.")

    left_features, right_features, cross_features, other_features = split_features(
        all_features
    )
    validate_link_features(left_features, right_features, all_features)

    out_dir = RESULTS_DIR / "link_ensemble_rf_s4_s6"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the audit trail of what each base model actually receives.
    inventory = []
    left_set = set(left_features)
    right_set = set(right_features)
    cross_set = set(cross_features)

    for f in all_features:
        if f in left_set:
            side = "LEFT"
        elif f in right_set:
            side = "RIGHT"
        elif f in cross_set:
            side = "CROSS"
        else:
            side = "OTHER"
        inventory.append({"feature": f, "group": side})

    write_csv(out_dir / "feature_inventory.csv", inventory)

    print(f"\nCross-link features excluded from base models: {len(cross_features)}")
    print(f"Other/global features excluded from base models: {len(other_features)}")
    print("All 542 features remain available to the EMPTY/OCCUPIED gate.")

    results = []
    trial_rows_all = []
    reliability_rows = []

    for outer_i, test_session in enumerate(SESSIONS):
        train_sessions = [s for s in SESSIONS if s != test_session]

        train_rows = []
        for s in train_sessions:
            train_rows.extend(rows_by_session[s])

        test_rows = rows_by_session[test_session]

        print("\n" + "=" * 78)
        print(f"{'+'.join(train_sessions)} -> {test_session}")
        print("=" * 78)

        ensemble = fit_ensemble(
            rows_by_session,
            train_sessions,
            all_features,
            left_features,
            right_features,
            args.seed + outer_i * 10000,
        )

        pack = predict_ensemble(
            ensemble,
            test_rows,
            all_features,
            left_features,
            right_features,
            args.empty_threshold,
        )

        m = evaluate_predictions(
            test_rows,
            pack,
            args.empty_threshold,
        )

        plain_trial_acc = evaluate_plain_control(
            train_rows,
            test_rows,
            all_features,
            args.seed + outer_i * 10000 + 9000,
        )

        result = {
            "train_sessions": "+".join(train_sessions),
            "test_session": test_session,
            "plain_rf_trial_accuracy": plain_trial_acc,
            "ensemble_window_accuracy": m["window_accuracy"],
            "ensemble_window_macro_f1": m["window_macro_f1"],
            "ensemble_trial_accuracy": m["trial_accuracy"],
            "ensemble_trial_macro_f1": m["trial_macro_f1"],
            "ensemble_adjacent_accuracy_occupied": m["adjacent_accuracy_occupied"],
            "ensemble_median_xy_error_m": m["median_xy_error_m"],
            "ensemble_p90_xy_error_m": m["p90_xy_error_m"],
        }
        results.append(result)

        for row in m["trial_rows"]:
            trial_rows_all.append(
                {"test_session": test_session, **row}
            )

        for j, zone in enumerate(OCCUPIED_CLASSES):
            reliability_rows.append(
                {
                    "test_session": test_session,
                    "zone": zone,
                    "left_oof_f1": float(ensemble["f1_left"][j]),
                    "right_oof_f1": float(ensemble["f1_right"][j]),
                    "left_weight": float(ensemble["weights"][0, j]),
                    "right_weight": float(ensemble["weights"][1, j]),
                }
            )

        print(f"Plain combined RF trial accuracy : {plain_trial_acc:.3f}")
        print(f"Link-ensemble exact trial        : {m['trial_accuracy']:.3f}")
        print(f"Link-ensemble adjacent occupied  : {m['adjacent_accuracy_occupied']:.3f}")
        print(f"Link-ensemble median XY error    : {m['median_xy_error_m']:.3f} m")
        print(f"Link-ensemble P90 XY error       : {m['p90_xy_error_m']:.3f} m")

    # Final model on all S4/S5/S6.  Its fusion calibration is generated with
    # leave-one-session-out OOF predictions from these three development sessions.
    print("\nTraining final S4+S5+S6 link ensemble...")
    final_bundle = fit_ensemble(
        rows_by_session,
        list(SESSIONS),
        all_features,
        left_features,
        right_features,
        args.seed + 77777,
    )

    model_bundle = {
        **final_bundle,
        "all_features": all_features,
        "left_features": left_features,
        "right_features": right_features,
        "cross_features_excluded_from_base_models": cross_features,
        "other_features_excluded_from_base_models": other_features,
        "occupied_classes": list(OCCUPIED_CLASSES),
        "zone_centers": ZONE_CENTERS,
        "empty_threshold": args.empty_threshold,
        "train_sessions": list(SESSIONS),
        "method": "class_weighted_two_link_late_fusion_rf_with_logistic_stacker",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "link_ensemble_rf_s4_s5_s6.joblib"
    joblib.dump(model_bundle, model_path)

    write_csv(out_dir / "loso_metrics.csv", results)
    write_csv(out_dir / "trial_predictions.csv", trial_rows_all)
    write_csv(out_dir / "class_link_weights.csv", reliability_rows)

    plain = np.asarray([r["plain_rf_trial_accuracy"] for r in results], dtype=float)
    ens = np.asarray([r["ensemble_trial_accuracy"] for r in results], dtype=float)
    adj = np.asarray(
        [r["ensemble_adjacent_accuracy_occupied"] for r in results],
        dtype=float,
    )
    med = np.asarray([r["ensemble_median_xy_error_m"] for r in results], dtype=float)
    p90 = np.asarray([r["ensemble_p90_xy_error_m"] for r in results], dtype=float)

    summary = {
        "results": results,
        "mean_plain_rf_trial_accuracy": float(plain.mean()),
        "mean_ensemble_trial_accuracy": float(ens.mean()),
        "mean_ensemble_adjacent_accuracy_occupied": float(adj.mean()),
        "mean_ensemble_median_xy_error_m": float(med.mean()),
        "mean_ensemble_p90_xy_error_m": float(p90.mean()),
        "left_feature_count": len(left_features),
        "right_feature_count": len(right_features),
        "final_model": str(model_path),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("TWO-LINK LATE-FUSION SUMMARY")
    print("=" * 78)

    for r in results:
        print(
            f"{r['train_sessions']:5s} -> {r['test_session']} | "
            f"plain={r['plain_rf_trial_accuracy']:.3f} | "
            f"ensemble={r['ensemble_trial_accuracy']:.3f} | "
            f"adj={r['ensemble_adjacent_accuracy_occupied']:.3f} | "
            f"med={r['ensemble_median_xy_error_m']:.3f} m | "
            f"p90={r['ensemble_p90_xy_error_m']:.3f} m"
        )

    print(f"\nMean plain RF exact trial : {plain.mean():.3f}")
    print(f"Mean ensemble exact trial : {ens.mean():.3f}")
    print(f"Mean ensemble adjacent    : {adj.mean():.3f}")
    print(f"Mean median XY error      : {med.mean():.3f} m")
    print(f"Mean P90 XY error         : {p90.mean():.3f} m")
    print(f"\nSaved: {out_dir}")
    print(f"Final model: {model_path}")
    print("\nSTOP HERE after this run. No new session is needed for this development step.")


if __name__ == "__main__":
    main()
