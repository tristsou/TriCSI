from __future__ import annotations

"""
TriCSI cross-session evaluation with EMPTY-room calibration.

Purpose
-------
The ordinary S1->S2 RF collapsed from ~68% same-session CV to ~31% cross-session.
A major clue was that S2 EMPTY was never recognized as EMPTY, which indicates
that the room/background CSI baseline shifted between sessions.

This script performs a deployment-style calibration:

    S1 windows -> normalize using S1 EMPTY baseline
    S2 windows -> normalize using S2 EMPTY baseline

Then:
    train RF on normalized S1
    test RF on normalized S2

The test session's zone labels are NEVER used for fitting the RF.
The only test-session information used before prediction is its EMPTY-room
calibration data. This corresponds to a practical startup procedure where the
room is empty briefly before localization begins.

Normalization
-------------
For every engineered feature separately:

    x_cal = (x - median(EMPTY_session)) / robust_scale(EMPTY_session)

robust_scale = 1.4826 * MAD

Very tiny EMPTY scales are protected by a training-derived floor so that
near-constant features do not explode numerically.

Run:
    python evaluate_cross_session_calibrated.py --train-session S1 --test-session S2

Reverse:
    python evaluate_cross_session_calibrated.py --train-session S2 --test-session S1
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from features import iter_trial_windows
from project_config import (
    CLASS_ORDER,
    HOP_SECONDS,
    MIN_PACKETS_PER_LINK,
    RAW_DATA_DIR,
    RESULTS_DIR,
    WINDOW_SECONDS,
    ZONE_CENTERS,
)


MAD_TO_SIGMA = 1.4826
EPS = 1e-9


def gather_trial_paths(session_dir: Path) -> list[Path]:
    return sorted(
        p for p in session_dir.rglob("*.csv")
        if not p.name.startswith("schedule_")
    )


def gather_windows(paths: list[Path], window_seconds: float, hop_seconds: float):
    rows = []
    for i, path in enumerate(paths, 1):
        print(f"[{i:03d}/{len(paths):03d}] features: {path.name}")
        rows.extend(
            iter_trial_windows(
                path,
                window_seconds=window_seconds,
                hop_seconds=hop_seconds,
                min_packets_per_link=MIN_PACKETS_PER_LINK,
            )
        )
    return rows


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def calc_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def matrix_from_rows(rows, feature_names):
    X = np.asarray(
        [[r["features"][name] for name in feature_names] for r in rows],
        dtype=np.float64,
    )
    y = np.asarray([r["zone"] for r in rows], dtype=object)
    return X, y


def empty_stats(X: np.ndarray, y: np.ndarray):
    mask = y == "EMPTY"
    if not np.any(mask):
        raise RuntimeError("No EMPTY windows found for session calibration.")

    empty = X[mask]
    center = np.median(empty, axis=0)
    mad = np.median(np.abs(empty - center), axis=0)
    scale = MAD_TO_SIGMA * mad
    return center, scale, int(np.sum(mask))


def calibrate_sessions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Normalize each session relative to its own EMPTY-room feature baseline.

    A common scale floor is derived from the training session's overall feature
    variation so that features that are almost constant during EMPTY do not
    produce enormous z-scores.
    """
    train_center, train_scale, n_train_empty = empty_stats(X_train, y_train)
    test_center, test_scale, n_test_empty = empty_stats(X_test, y_test)

    # Feature-wise robust scale across ALL training windows. This is used only
    # as a floor; it does not leak test-zone information.
    all_train_center = np.median(X_train, axis=0)
    all_train_mad = np.median(np.abs(X_train - all_train_center), axis=0)
    train_global_scale = MAD_TO_SIGMA * all_train_mad

    # Require at least 5% of the overall training feature scale, or EPS.
    scale_floor = np.maximum(0.05 * train_global_scale, EPS)

    train_scale_safe = np.maximum(train_scale, scale_floor)
    test_scale_safe = np.maximum(test_scale, scale_floor)

    X_train_cal = (X_train - train_center) / train_scale_safe
    X_test_cal = (X_test - test_center) / test_scale_safe

    calibration = {
        "n_train_empty_windows": n_train_empty,
        "n_test_empty_windows": n_test_empty,
        "train_empty_center": train_center,
        "test_empty_center": test_center,
        "train_empty_scale": train_scale_safe,
        "test_empty_scale": test_scale_safe,
        "scale_floor": scale_floor,
    }
    return X_train_cal, X_test_cal, calibration


def probability_weighted_xy(probabilities: np.ndarray, classes: np.ndarray):
    indices, centers = [], []
    for i, cls in enumerate(classes):
        if cls in ZONE_CENTERS:
            indices.append(i)
            centers.append(ZONE_CENTERS[cls])

    centers = np.asarray(centers, dtype=np.float64)
    p = probabilities[:, indices]
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p @ centers


def xy_errors(y_true, xy_pred):
    errors = []
    for label, xy in zip(y_true, xy_pred):
        if label not in ZONE_CENTERS:
            errors.append(np.nan)
        else:
            target = np.asarray(ZONE_CENTERS[label], dtype=np.float64)
            errors.append(float(np.linalg.norm(xy - target)))
    return np.asarray(errors, dtype=np.float64)


def summarize_xy(errors):
    x = np.asarray(errors, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean_m": None, "median_m": None, "p90_m": None}
    return {
        "mean_m": float(np.mean(x)),
        "median_m": float(np.median(x)),
        "p90_m": float(np.percentile(x, 90)),
    }


def majority_vote_with_prob_tiebreak(preds, probs, classes):
    counts = Counter(preds.tolist())
    top_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == top_count]
    if len(tied) == 1:
        return tied[0]

    mean_prob = np.mean(probs, axis=0)
    idx = {label: i for i, label in enumerate(classes)}
    return max(tied, key=lambda label: mean_prob[idx[label]])


def save_confusion(y_true, y_pred, labels, path, title, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        shown = np.divide(
            cm.astype(float),
            denom,
            out=np.zeros_like(cm, dtype=float),
            where=denom != 0,
        )
    else:
        shown = cm

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(shown)
    ax.set_title(title)
    ax.set_xlabel("Predicted zone")
    ax.set_ylabel("True zone")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{shown[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return cm


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-session", default="S1")
    p.add_argument("--test-session", default="S2")
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--window", type=float, default=WINDOW_SECONDS)
    p.add_argument("--hop", type=float, default=HOP_SECONDS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_dir = args.raw_dir / args.train_session
    test_dir = args.raw_dir / args.test_session

    if not train_dir.exists():
        raise SystemExit(f"Missing training session: {train_dir}")
    if not test_dir.exists():
        raise SystemExit(f"Missing test session: {test_dir}")

    train_paths = gather_trial_paths(train_dir)
    test_paths = gather_trial_paths(test_dir)

    print("\n" + "=" * 78)
    print("TriCSI EMPTY-CALIBRATED CROSS-SESSION EVALUATION")
    print("=" * 78)
    print(f"TRAIN : {args.train_session} ({len(train_paths)} whole trials)")
    print(f"TEST  : {args.test_session} ({len(test_paths)} whole trials)")
    print("Calibration: each session is normalized using its own EMPTY windows.")
    print("No occupied-zone labels from the test session are used for model fitting.")
    print("=" * 78)

    print("\nExtracting TRAIN features...")
    train_rows = gather_windows(train_paths, args.window, args.hop)

    print("\nExtracting TEST features...")
    test_rows = gather_windows(test_paths, args.window, args.hop)

    feature_names = sorted(train_rows[0]["features"])
    if feature_names != sorted(test_rows[0]["features"]):
        raise SystemExit("Train/test feature sets do not match.")

    X_train, y_train = matrix_from_rows(train_rows, feature_names)
    X_test, y_test = matrix_from_rows(test_rows, feature_names)

    print("\nCALIBRATION")
    X_train_cal, X_test_cal, calibration = calibrate_sessions(
        X_train, y_train, X_test, y_test
    )
    print(f"  {args.train_session} EMPTY windows: {calibration['n_train_empty_windows']}")
    print(f"  {args.test_session} EMPTY windows : {calibration['n_test_empty_windows']}")
    print("  Robust feature centering/scaling complete.")

    labels = [c for c in CLASS_ORDER if c in set(y_train) and c in set(y_test)]

    print("\nWINDOW COUNTS")
    print(f"  Train windows: {len(X_train_cal)}")
    print(f"  Test windows : {len(X_test_cal)}")

    print("\nTraining calibrated Random Forest...")
    model = make_rf(args.seed)
    model.fit(X_train_cal, y_train)

    pred = model.predict(X_test_cal)
    probs = model.predict_proba(X_test_cal)

    window_metrics = calc_metrics(y_test, pred)
    pred_xy = probability_weighted_xy(probs, model.classes_)
    window_xy = xy_errors(y_test, pred_xy)
    window_xy_summary = summarize_xy(window_xy)

    # Trial aggregation.
    groups = np.asarray([r["trial_group"] for r in test_rows], dtype=object)
    trial_rows_out = []
    trial_true, trial_majority, trial_mean_prob = [], [], []
    trial_xy_values = []

    for group in sorted(set(groups)):
        idxs = np.where(groups == group)[0]
        true_label = str(y_test[idxs[0]])

        gp = probs[idxs]
        gpred = pred[idxs]
        mean_prob = np.mean(gp, axis=0)

        maj = majority_vote_with_prob_tiebreak(gpred, gp, model.classes_)
        prob_pred = str(model.classes_[int(np.argmax(mean_prob))])

        xy = probability_weighted_xy(mean_prob.reshape(1, -1), model.classes_)[0]
        xyerr = xy_errors(
            np.asarray([true_label], dtype=object),
            xy.reshape(1, 2),
        )[0]

        first = test_rows[idxs[0]]
        row = {
            "trial_group": group,
            "source_file": first["source_file"],
            "zone_true": true_label,
            "trial": first["trial"],
            "orientation": first["orientation"],
            "n_windows": len(idxs),
            "pred_majority": maj,
            "pred_mean_probability": prob_pred,
            "correct_majority": int(maj == true_label),
            "correct_mean_probability": int(prob_pred == true_label),
            "x_pred_m": float(xy[0]),
            "y_pred_m": float(xy[1]),
            "xy_error_m": None if not np.isfinite(xyerr) else float(xyerr),
        }

        class_index = {c: i for i, c in enumerate(model.classes_)}
        for cls in labels:
            row[f"p_{cls}"] = float(mean_prob[class_index[cls]])

        trial_rows_out.append(row)
        trial_true.append(true_label)
        trial_majority.append(maj)
        trial_mean_prob.append(prob_pred)
        if np.isfinite(xyerr):
            trial_xy_values.append(float(xyerr))

    trial_true = np.asarray(trial_true, dtype=object)
    trial_majority = np.asarray(trial_majority, dtype=object)
    trial_mean_prob = np.asarray(trial_mean_prob, dtype=object)

    trial_majority_metrics = calc_metrics(trial_true, trial_majority)
    trial_mean_prob_metrics = calc_metrics(trial_true, trial_mean_prob)
    trial_xy_summary = summarize_xy(trial_xy_values)

    out_dir = RESULTS_DIR / (
        f"cross_session_calibrated_"
        f"{args.train_session.lower()}_to_{args.test_session.lower()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_window = save_confusion(
        y_test, pred, labels,
        out_dir / "confusion_matrix_window.png",
        f"RF EMPTY-calibrated {args.train_session}->{args.test_session} (window)",
    )
    save_confusion(
        y_test, pred, labels,
        out_dir / "confusion_matrix_window_normalized.png",
        f"RF EMPTY-calibrated {args.train_session}->{args.test_session} normalized",
        normalize=True,
    )
    cm_trial = save_confusion(
        trial_true, trial_mean_prob, labels,
        out_dir / "confusion_matrix_trial_mean_probability.png",
        f"RF EMPTY-calibrated {args.train_session}->{args.test_session} (trial)",
    )

    zone_rows = []
    for i, label in enumerate(labels):
        ws = int(cm_window[i].sum())
        ts = int(cm_trial[i].sum())
        zone_rows.append({
            "zone": label,
            "window_correct": int(cm_window[i, i]),
            "window_support": ws,
            "window_recall": float(cm_window[i, i] / ws) if ws else None,
            "trial_correct": int(cm_trial[i, i]),
            "trial_support": ts,
            "trial_recall": float(cm_trial[i, i] / ts) if ts else None,
        })
    write_csv(out_dir / "per_zone_recall.csv", zone_rows)
    write_csv(out_dir / "trial_predictions.csv", trial_rows_out)

    # Orientation analysis.
    test_orient = np.asarray([r["orientation"] for r in test_rows], dtype=object)
    orientation_rows = []
    for orientation in sorted(set(test_orient)):
        mask = test_orient == orientation
        trial_idx = [
            i for i, r in enumerate(trial_rows_out)
            if r["orientation"] == orientation
        ]
        orientation_rows.append({
            "orientation": orientation,
            "window_count": int(np.sum(mask)),
            "window_accuracy": float(accuracy_score(y_test[mask], pred[mask])),
            "window_macro_f1": float(
                f1_score(y_test[mask], pred[mask], average="macro", zero_division=0)
            ),
            "trial_count": len(trial_idx),
            "trial_accuracy_mean_probability": (
                float(
                    accuracy_score(
                        trial_true[trial_idx],
                        trial_mean_prob[trial_idx],
                    )
                ) if trial_idx else None
            ),
        })
    write_csv(out_dir / "orientation_metrics.csv", orientation_rows)

    (out_dir / "classification_report_window.txt").write_text(
        classification_report(y_test, pred, labels=labels, zero_division=0),
        encoding="utf-8",
    )
    (out_dir / "classification_report_trial.txt").write_text(
        classification_report(
            trial_true, trial_mean_prob, labels=labels, zero_division=0
        ),
        encoding="utf-8",
    )

    # Save feature-level calibration diagnostics, but not huge raw arrays in JSON.
    calibration_rows = []
    for i, name in enumerate(feature_names):
        calibration_rows.append({
            "feature": name,
            "train_empty_center": float(calibration["train_empty_center"][i]),
            "test_empty_center": float(calibration["test_empty_center"][i]),
            "train_empty_scale": float(calibration["train_empty_scale"][i]),
            "test_empty_scale": float(calibration["test_empty_scale"][i]),
            "scale_floor": float(calibration["scale_floor"][i]),
            "center_shift_abs": float(abs(
                calibration["test_empty_center"][i]
                - calibration["train_empty_center"][i]
            )),
        })
    write_csv(out_dir / "calibration_feature_shifts.csv", calibration_rows)

    summary = {
        "train_session": args.train_session,
        "test_session": args.test_session,
        "calibration": "session-specific EMPTY median + robust MAD scale",
        "test_session_information_used": (
            "EMPTY-room calibration windows only; occupied-zone labels are not "
            "used for fitting or normalization."
        ),
        "train_trials": len(train_paths),
        "test_trials": len(test_paths),
        "train_windows": len(X_train_cal),
        "test_windows": len(X_test_cal),
        "train_empty_windows": calibration["n_train_empty_windows"],
        "test_empty_windows": calibration["n_test_empty_windows"],
        "window_metrics": window_metrics,
        "window_xy": window_xy_summary,
        "trial_majority_vote_metrics": trial_majority_metrics,
        "trial_mean_probability_metrics": trial_mean_prob_metrics,
        "trial_xy": trial_xy_summary,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("EMPTY-CALIBRATED CROSS-SESSION RESULTS")
    print("=" * 78)
    print(
        f"Window level: accuracy={window_metrics['accuracy']:.3f} "
        f"macroF1={window_metrics['macro_f1']:.3f} "
        f"medianXY={window_xy_summary['median_m']:.3f} m "
        f"p90XY={window_xy_summary['p90_m']:.3f} m"
    )
    print(
        f"Trial majority vote: accuracy={trial_majority_metrics['accuracy']:.3f} "
        f"macroF1={trial_majority_metrics['macro_f1']:.3f}"
    )
    print(
        f"Trial mean probability: accuracy={trial_mean_prob_metrics['accuracy']:.3f} "
        f"macroF1={trial_mean_prob_metrics['macro_f1']:.3f}"
    )
    print(f"\nSaved results: {out_dir}")
    print("\nCompare these directly with the uncalibrated cross-session results.")


if __name__ == "__main__":
    main()
