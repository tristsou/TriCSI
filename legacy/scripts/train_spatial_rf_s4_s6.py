from __future__ import annotations

"""
TriCSI hierarchical + spatial Random Forest.

This implements the next model:

    CSI features
        ↓
    Stage 1: EMPTY vs OCCUPIED RF
        ↓ occupied
    Stage 2: Z1..Z9 RF
        ↓
    probability-weighted continuous (x, y)

Development data:
    S4, S5, S6

Evaluation:
    Leave-one-session-out:
        train S5+S6 -> test S4
        train S4+S6 -> test S5
        train S4+S5 -> test S6

After evaluation, the script trains one final model on S4+S5+S6 and saves it
for a future untouched S7 test.

IMPORTANT:
S6 has already been inspected, so S4-S6 are now development data.
Do not use S7 to tune this model before the first S7 evaluation.

Run:
    python train_spatial_rf_s4_s6.py
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
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

# Same 3x3 grid centers used previously, in meters.
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
# Data helpers
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


def feature_matrix(rows, names):
    X = np.asarray(
        [[float(r["features"][n]) for n in names] for r in rows],
        dtype=float,
    )
    y = np.asarray([str(r["zone"]) for r in rows], dtype=object)
    return X, y


def trial_group(r):
    return (
        f"{r['_session']}|{r['zone']}|"
        f"{trial_number_from_row(r)}|{r['orientation']}"
    )


def nearest_zone(xy):
    xy = np.asarray(xy, dtype=float)
    best_zone = None
    best_dist = float("inf")

    for zone, center in ZONE_CENTERS.items():
        d = float(np.linalg.norm(xy - np.asarray(center, dtype=float)))
        if d < best_dist:
            best_dist = d
            best_zone = zone

    return best_zone


def physical_error_m(true_zone, xy):
    if true_zone == "EMPTY":
        return None
    return float(
        np.linalg.norm(
            np.asarray(ZONE_CENTERS[true_zone], dtype=float)
            - np.asarray(xy, dtype=float)
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
# Hierarchical model
# ---------------------------------------------------------------------------

def fit_hierarchical(train_rows, feature_names, seed):
    X, y = feature_matrix(train_rows, feature_names)

    # Stage 1: EMPTY vs OCCUPIED
    y_binary = np.where(y == "EMPTY", "EMPTY", "OCCUPIED")

    empty_model = make_rf(seed)
    empty_model.fit(X, y_binary)

    # Stage 2: occupied zone only
    occupied_mask = y != "EMPTY"

    zone_model = make_rf(seed + 1000)
    zone_model.fit(
        X[occupied_mask],
        y[occupied_mask],
    )

    return empty_model, zone_model


def occupied_probabilities(zone_model, X):
    """
    Return probabilities in fixed Z1..Z9 order.
    """
    raw = zone_model.predict_proba(X)
    class_to_col = {
        str(c): i
        for i, c in enumerate(zone_model.classes_)
    }

    out = np.zeros((len(X), len(OCCUPIED_CLASSES)), dtype=float)

    for j, zone in enumerate(OCCUPIED_CLASSES):
        if zone in class_to_col:
            out[:, j] = raw[:, class_to_col[zone]]

    row_sums = out.sum(axis=1, keepdims=True)
    out = np.divide(
        out,
        row_sums,
        out=np.full_like(out, 1.0 / len(OCCUPIED_CLASSES)),
        where=row_sums > 0,
    )
    return out


def empty_probability(empty_model, X):
    raw = empty_model.predict_proba(X)
    class_to_col = {
        str(c): i
        for i, c in enumerate(empty_model.classes_)
    }

    if "EMPTY" not in class_to_col:
        raise RuntimeError("Binary model has no EMPTY class.")

    return raw[:, class_to_col["EMPTY"]]


def weighted_xy(zone_probs):
    centers = np.asarray(
        [ZONE_CENTERS[z] for z in OCCUPIED_CLASSES],
        dtype=float,
    )
    return zone_probs @ centers


def predict_hierarchical(
    empty_model,
    zone_model,
    X,
    empty_threshold=0.5,
):
    p_empty = empty_probability(empty_model, X)
    zone_probs = occupied_probabilities(zone_model, X)
    xy = weighted_xy(zone_probs)

    pred = []

    for i in range(len(X)):
        if p_empty[i] >= empty_threshold:
            pred.append("EMPTY")
        else:
            pred.append(nearest_zone(xy[i]))

    return (
        np.asarray(pred, dtype=object),
        p_empty,
        zone_probs,
        xy,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    train_rows,
    test_rows,
    feature_names,
    seed,
    empty_threshold,
):
    empty_model, zone_model = fit_hierarchical(
        train_rows,
        feature_names,
        seed,
    )

    X_test, y_test = feature_matrix(
        test_rows,
        feature_names,
    )

    pred_window, p_empty, zone_probs, xy_window = predict_hierarchical(
        empty_model,
        zone_model,
        X_test,
        empty_threshold=empty_threshold,
    )

    # Window metrics
    window_exact = float(accuracy_score(y_test, pred_window))
    window_f1 = float(
        f1_score(
            y_test,
            pred_window,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )
    )

    # Binary occupied/empty metrics
    y_bin = np.where(y_test == "EMPTY", "EMPTY", "OCCUPIED")
    p_bin = np.where(pred_window == "EMPTY", "EMPTY", "OCCUPIED")

    binary_acc = float(accuracy_score(y_bin, p_bin))
    binary_f1 = float(
        f1_score(
            y_bin,
            p_bin,
            labels=["EMPTY", "OCCUPIED"],
            average="macro",
            zero_division=0,
        )
    )

    # Trial aggregation
    groups = np.asarray(
        [trial_group(r) for r in test_rows],
        dtype=object,
    )

    trial_rows = []
    trial_true = []
    trial_pred = []
    occupied_errors = []

    for group in sorted(set(groups)):
        idx = np.where(groups == group)[0]

        true_zone = str(y_test[idx[0]])

        mean_p_empty = float(np.mean(p_empty[idx]))
        mean_zone_probs = np.mean(zone_probs[idx], axis=0)
        mean_zone_probs = mean_zone_probs / mean_zone_probs.sum()

        xy = weighted_xy(mean_zone_probs[None, :])[0]

        if mean_p_empty >= empty_threshold:
            pred_zone = "EMPTY"
        else:
            pred_zone = nearest_zone(xy)

        err = physical_error_m(true_zone, xy)

        if err is not None:
            occupied_errors.append(err)

        sample = test_rows[int(idx[0])]

        trial_rows.append(
            {
                "trial_group": group,
                "true_zone": true_zone,
                "pred_zone": pred_zone,
                "trial": int(trial_number_from_row(sample)),
                "orientation": str(sample["orientation"]),
                "n_windows": int(len(idx)),
                "p_empty": mean_p_empty,
                "x_m": float(xy[0]),
                "y_m": float(xy[1]),
                "physical_error_m": err,
                "physical_error_ft": None if err is None else float(err * 3.28084),
                "correct": int(pred_zone == true_zone),
                "exact_or_adjacent": adjacent(true_zone, pred_zone),
            }
        )

        trial_true.append(true_zone)
        trial_pred.append(pred_zone)

    trial_true = np.asarray(trial_true, dtype=object)
    trial_pred = np.asarray(trial_pred, dtype=object)

    trial_exact = float(accuracy_score(trial_true, trial_pred))
    trial_f1 = float(
        f1_score(
            trial_true,
            trial_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )
    )

    occupied_trial_rows = [
        r for r in trial_rows
        if r["true_zone"] != "EMPTY"
    ]

    adjacent_acc = float(
        np.mean(
            [r["exact_or_adjacent"] for r in occupied_trial_rows]
        )
    )

    occupied_errors = np.asarray(occupied_errors, dtype=float)

    metrics = {
        "window_exact_accuracy": window_exact,
        "window_macro_f1": window_f1,
        "binary_empty_occupied_accuracy": binary_acc,
        "binary_empty_occupied_macro_f1": binary_f1,
        "trial_exact_accuracy": trial_exact,
        "trial_macro_f1": trial_f1,
        "trial_exact_or_adjacent_accuracy_occupied": adjacent_acc,
        "trial_median_xy_error_m_occupied": float(np.median(occupied_errors)),
        "trial_p90_xy_error_m_occupied": float(np.percentile(occupied_errors, 90)),
        "trial_median_xy_error_ft_occupied": float(np.median(occupied_errors) * 3.28084),
        "trial_p90_xy_error_ft_occupied": float(np.percentile(occupied_errors, 90) * 3.28084),
        "train_windows": len(train_rows),
        "test_windows": len(test_rows),
    }

    return metrics, trial_rows


def write_csv(path, rows):
    if not rows:
        return

    keys = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
    )
    parser.add_argument(
        "--window",
        type=float,
        default=WINDOW_SECONDS,
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=HOP_SECONDS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--empty-threshold",
        type=float,
        default=0.5,
        help="Fixed EMPTY probability threshold. Default 0.5.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("TriCSI HIERARCHICAL + SPATIAL RANDOM FOREST")
    print("=" * 78)
    print("Development sessions: S4, S5, S6")
    print("Stage 1: EMPTY vs OCCUPIED")
    print("Stage 2: occupied Z1..Z9")
    print("Output : probability-weighted continuous (x,y)")
    print(f"EMPTY threshold: {args.empty_threshold:.2f}")
    print("=" * 78)

    data = {
        session: load_rows(
            args.raw_dir,
            session,
            args.window,
            args.hop,
        )
        for session in SESSIONS
    }

    feature_names = sorted(data["S4"][0]["features"])

    for session in ("S5", "S6"):
        if sorted(data[session][0]["features"]) != feature_names:
            raise RuntimeError(
                f"{session} feature names do not match S4."
            )

    out_dir = RESULTS_DIR / "spatial_rf_s4_s6"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_trial_rows = []

    # Leave-one-session-out development evaluation
    for i, test_session in enumerate(SESSIONS):
        train_sessions = [
            s for s in SESSIONS
            if s != test_session
        ]

        train_rows = []
        for s in train_sessions:
            train_rows.extend(data[s])

        test_rows = data[test_session]

        print("\n" + "=" * 78)
        print(
            f"TRAIN {train_sessions[0]}+{train_sessions[1]} "
            f"-> TEST {test_session}"
        )
        print("=" * 78)

        metrics, trial_rows = evaluate_split(
            train_rows,
            test_rows,
            feature_names,
            args.seed + i * 100,
            args.empty_threshold,
        )

        result = {
            "test_session": test_session,
            "train_sessions": "+".join(train_sessions),
            **metrics,
        }
        all_results.append(result)

        for row in trial_rows:
            all_trial_rows.append(
                {
                    "test_session": test_session,
                    **row,
                }
            )

        print(
            f"Exact trial accuracy: "
            f"{metrics['trial_exact_accuracy']:.3f}"
        )
        print(
            f"Exact-or-adjacent occupied: "
            f"{metrics['trial_exact_or_adjacent_accuracy_occupied']:.3f}"
        )
        print(
            f"Median XY error: "
            f"{metrics['trial_median_xy_error_m_occupied']:.3f} m "
            f"({metrics['trial_median_xy_error_ft_occupied']:.2f} ft)"
        )
        print(
            f"P90 XY error: "
            f"{metrics['trial_p90_xy_error_m_occupied']:.3f} m "
            f"({metrics['trial_p90_xy_error_ft_occupied']:.2f} ft)"
        )
        print(
            f"EMPTY/OCCUPIED accuracy: "
            f"{metrics['binary_empty_occupied_accuracy']:.3f}"
        )

    # Aggregate LOSO summary
    exact = np.asarray(
        [r["trial_exact_accuracy"] for r in all_results],
        dtype=float,
    )
    adjacent_acc = np.asarray(
        [r["trial_exact_or_adjacent_accuracy_occupied"] for r in all_results],
        dtype=float,
    )
    mederr = np.asarray(
        [r["trial_median_xy_error_m_occupied"] for r in all_results],
        dtype=float,
    )
    p90err = np.asarray(
        [r["trial_p90_xy_error_m_occupied"] for r in all_results],
        dtype=float,
    )

    print("\n" + "=" * 78)
    print("SPATIAL RF LEAVE-ONE-SESSION-OUT SUMMARY")
    print("=" * 78)

    for r in all_results:
        print(
            f"{r['train_sessions']:5s} -> {r['test_session']}: "
            f"exact={r['trial_exact_accuracy']:.3f} | "
            f"adjacent={r['trial_exact_or_adjacent_accuracy_occupied']:.3f} | "
            f"median={r['trial_median_xy_error_m_occupied']:.3f} m | "
            f"p90={r['trial_p90_xy_error_m_occupied']:.3f} m"
        )

    print(
        f"\nMean exact trial accuracy: "
        f"{exact.mean():.3f}"
    )
    print(
        f"Mean exact-or-adjacent occupied: "
        f"{adjacent_acc.mean():.3f}"
    )
    print(
        f"Mean session median XY error: "
        f"{mederr.mean():.3f} m"
    )
    print(
        f"Mean session P90 XY error: "
        f"{p90err.mean():.3f} m"
    )

    # Train final model on all S4+S5+S6 for future S7.
    print("\nTraining final S4+S5+S6 hierarchical spatial RF for S7...")

    final_rows = data["S4"] + data["S5"] + data["S6"]

    empty_model, zone_model = fit_hierarchical(
        final_rows,
        feature_names,
        args.seed + 9999,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "spatial_rf_s4_s5_s6_for_s7.joblib"

    joblib.dump(
        {
            "empty_model": empty_model,
            "zone_model": zone_model,
            "feature_names": feature_names,
            "occupied_classes": list(OCCUPIED_CLASSES),
            "zone_centers": ZONE_CENTERS,
            "empty_threshold": args.empty_threshold,
            "train_sessions": ["S4", "S5", "S6"],
            "model_type": "hierarchical_spatial_rf",
        },
        model_path,
    )

    write_csv(
        out_dir / "leave_one_session_out_metrics.csv",
        all_results,
    )

    write_csv(
        out_dir / "trial_predictions.csv",
        all_trial_rows,
    )

    summary = {
        "leave_one_session_out": all_results,
        "mean_trial_exact_accuracy": float(exact.mean()),
        "mean_trial_exact_or_adjacent_accuracy_occupied": float(adjacent_acc.mean()),
        "mean_session_median_xy_error_m": float(mederr.mean()),
        "mean_session_p90_xy_error_m": float(p90err.mean()),
        "empty_threshold": args.empty_threshold,
        "final_model_for_s7": str(model_path),
        "note": (
            "S4-S6 are development sessions. "
            "Keep S7 untouched until first evaluation."
        ),
    }

    with (out_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(f"\nSaved development results: {out_dir}")
    print(f"Saved final S7 model: {model_path}")


if __name__ == "__main__":
    main()
