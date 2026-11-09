from __future__ import annotations

"""
TriCSI 5-fold whole-trial cross-validation for the S1 localization dataset.

What this script does
---------------------
1. Loads ONLY one requested session (default: S1).
2. Extracts the exact same engineered features used by train_localizer.py.
3. Uses trial-number folds:
       fold 1 -> test trial 1 from every class
       fold 2 -> test trial 2 from every class
       ...
       fold 5 -> test trial 5 from every class
   Every test trial is completely absent from that fold's training set.
4. Trains a NEW Random Forest from scratch for every fold.
5. Produces:
   - per-fold metrics
   - aggregate out-of-fold (OOF) window predictions
   - aggregate confusion matrix
   - per-zone recall
   - trial-level majority-vote predictions (50 independent trials)
   - trial-level confusion matrix
   - orientation-specific performance
   - cross-validated feature importance
   - XY localization error
6. Finally trains a deployment RF on ALL S1 data and saves it separately.
   This full-data model is NOT used to calculate cross-validation metrics.

Run from the TriCSI project directory with the virtual environment active:
    python train_crossval.py --session S1
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
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
    MODEL_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    ROOM_HEIGHT_M,
    ROOM_WIDTH_M,
    WINDOW_SECONDS,
    ZONE_CENTERS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def trial_number_from_row(row: dict[str, Any]) -> int:
    try:
        return int(str(row["trial"]).strip())
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Could not interpret trial number {row.get('trial')!r} "
            f"from {row.get('source_file', '<unknown file>')}"
        ) from exc


def gather_windows(
    paths: list[Path],
    *,
    window_seconds: float,
    hop_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index, path in enumerate(paths, 1):
        print(f"[{index:03d}/{len(paths):03d}] features: {path.name}")
        rows.extend(
            iter_trial_windows(
                path,
                window_seconds=window_seconds,
                hop_seconds=hop_seconds,
                min_packets_per_link=MIN_PACKETS_PER_LINK,
            )
        )

    return rows


def probability_weighted_xy(
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """
    Turn class probabilities into a continuous XY estimate using occupied
    zone centers. EMPTY probability is excluded and occupied probabilities
    are renormalized.
    """
    centers = []
    indices = []

    for index, cls in enumerate(classes):
        if cls in ZONE_CENTERS:
            indices.append(index)
            centers.append(ZONE_CENTERS[cls])

    if not indices:
        return np.full((len(probabilities), 2), np.nan, dtype=np.float64)

    centers_arr = np.asarray(centers, dtype=np.float64)
    p = probabilities[:, indices]
    denom = np.sum(p, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    p = p / denom
    return p @ centers_arr


def xy_error_for_label(label: str, pred_xy: np.ndarray) -> float:
    if label not in ZONE_CENTERS:
        return float("nan")
    true_xy = np.asarray(ZONE_CENTERS[label], dtype=np.float64)
    return float(np.linalg.norm(pred_xy - true_xy))


def summarize_xy_errors(errors: np.ndarray) -> dict[str, float | None]:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return {
            "xy_mean_m": None,
            "xy_median_m": None,
            "xy_p90_m": None,
        }
    return {
        "xy_mean_m": float(np.mean(errors)),
        "xy_median_m": float(np.median(errors)),
        "xy_p90_m": float(np.percentile(errors, 90)),
    }


def standard_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def make_rf(seed: int) -> RandomForestClassifier:
    """Exact RF settings used by the original TriCSI baseline."""
    return RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def save_confusion_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
    *,
    normalize: bool = False,
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        shown = np.divide(
            cm.astype(np.float64),
            denom,
            out=np.zeros_like(cm, dtype=np.float64),
            where=denom != 0,
        )
    else:
        shown = cm

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    image = ax.imshow(shown)
    ax.set_title(title)
    ax.set_xlabel("Predicted zone")
    ax.set_ylabel("True zone")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if normalize:
                text = f"{shown[i, j]:.2f}"
            else:
                text = str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return cm


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def majority_vote_with_probability_tiebreak(
    hard_predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> str:
    counts = Counter(hard_predictions.tolist())
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]

    if len(tied) == 1:
        return tied[0]

    mean_p = np.mean(probabilities, axis=0)
    class_to_index = {label: i for i, label in enumerate(classes)}
    return max(tied, key=lambda label: mean_p[class_to_index[label]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="5-fold whole-trial CV for TriCSI localization."
    )
    parser.add_argument(
        "--session",
        default="S1",
        help="Session folder to evaluate. Default: S1",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root raw-data directory. Default comes from project_config.py",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--hop", type=float, default=HOP_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-full-model",
        action="store_true",
        help="Do not train/save the final all-session deployment model.",
    )
    args = parser.parse_args()

    if args.folds < 2:
        raise SystemExit("--folds must be at least 2.")

    session_dir = args.raw_dir / args.session
    if not session_dir.exists():
        raise SystemExit(
            f"Session directory does not exist:\n  {session_dir}\n"
            "Check --session and --raw-dir."
        )

    trial_paths = sorted(
        p
        for p in session_dir.rglob("*.csv")
        if not p.name.startswith("schedule_")
    )

    if not trial_paths:
        raise SystemExit(f"No trial CSV files found under {session_dir}")

    expected_classes = list(CLASS_ORDER)

    print("\n" + "=" * 78)
    print("TriCSI 5-FOLD WHOLE-TRIAL CROSS-VALIDATION")
    print("=" * 78)
    print(f"Session       : {args.session}")
    print(f"Session path  : {session_dir}")
    print(f"Trial files   : {len(trial_paths)}")
    print(f"Window / hop  : {args.window:.2f} s / {args.hop:.2f} s")
    print(f"Folds         : {args.folds}")
    print("Split rule    : test trial number = fold number")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    windows = gather_windows(
        trial_paths,
        window_seconds=args.window,
        hop_seconds=args.hop,
    )
    if not windows:
        raise SystemExit("No valid dual-link windows were created.")

    feature_names = sorted(windows[0]["features"])

    X = np.asarray(
        [[row["features"][name] for name in feature_names] for row in windows],
        dtype=np.float64,
    )
    y = np.asarray([row["zone"] for row in windows], dtype=object)
    trial_numbers = np.asarray(
        [trial_number_from_row(row) for row in windows],
        dtype=np.int64,
    )
    groups = np.asarray([row["trial_group"] for row in windows], dtype=object)
    orientations = np.asarray([row["orientation"] for row in windows], dtype=object)

    labels_present = [label for label in expected_classes if label in set(y)]

    missing_classes = set(expected_classes) - set(labels_present)
    if missing_classes:
        raise SystemExit(
            "S1 is missing expected classes: "
            + ", ".join(sorted(missing_classes))
        )

    # Validate that every class has exactly the expected fold trial numbers.
    print("\nTRIAL/FOLD VALIDATION")
    validation_errors: list[str] = []

    for label in labels_present:
        label_mask = y == label
        class_groups = sorted(set(groups[label_mask]))

        # One trial_group can contain many overlapping windows; inspect each
        # group's trial number once.
        seen_trials: dict[int, set[str]] = defaultdict(set)
        for group in class_groups:
            idx = np.where(groups == group)[0][0]
            seen_trials[int(trial_numbers[idx])].add(group)

        observed = sorted(seen_trials)
        expected = list(range(1, args.folds + 1))

        print(
            f"  {label:5s}: independent trials={len(class_groups):2d} "
            f"trial_numbers={observed}"
        )

        if observed != expected:
            validation_errors.append(
                f"{label}: expected trial numbers {expected}, found {observed}"
            )

        for trial_no, trial_groups in seen_trials.items():
            if len(trial_groups) != 1:
                validation_errors.append(
                    f"{label} trial {trial_no}: expected 1 recording, "
                    f"found {len(trial_groups)} groups"
                )

    if validation_errors:
        raise SystemExit(
            "\nFold validation failed:\n  - "
            + "\n  - ".join(validation_errors)
        )

    print("\nWINDOW COUNTS")
    for label in labels_present:
        print(f"  {label:5s}: {int(np.sum(y == label))}")

    output_dir = RESULTS_DIR / f"crossval_{args.session.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Five folds
    # ------------------------------------------------------------------
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    fold_importances: list[np.ndarray] = []

    all_oof_true: list[str] = []
    all_oof_pred: list[str] = []
    all_oof_xy_errors: list[float] = []

    all_trial_true: list[str] = []
    all_trial_pred: list[str] = []
    all_trial_prob_pred: list[str] = []
    all_trial_xy_errors: list[float] = []

    for fold in range(1, args.folds + 1):
        print("\n" + "-" * 78)
        print(f"FOLD {fold}/{args.folds}: hold out trial {fold} from EVERY class")
        print("-" * 78)

        test_mask = trial_numbers == fold
        train_mask = ~test_mask

        train_groups = sorted(set(groups[train_mask]))
        test_groups = sorted(set(groups[test_mask]))

        # Strong leakage guards.
        overlap = set(train_groups) & set(test_groups)
        if overlap:
            raise RuntimeError(
                f"Leakage detected in fold {fold}: {sorted(overlap)[:3]}"
            )

        train_classes = set(y[train_mask])
        test_classes = set(y[test_mask])
        if train_classes != set(labels_present) or test_classes != set(labels_present):
            raise RuntimeError(
                f"Fold {fold} class mismatch. "
                f"train={sorted(train_classes)}, test={sorted(test_classes)}"
            )

        if len(test_groups) != len(labels_present):
            raise RuntimeError(
                f"Fold {fold}: expected {len(labels_present)} held-out whole trials, "
                f"found {len(test_groups)}."
            )

        expected_train_groups = len(labels_present) * (args.folds - 1)
        if len(train_groups) != expected_train_groups:
            raise RuntimeError(
                f"Fold {fold}: expected {expected_train_groups} train trials, "
                f"found {len(train_groups)}."
            )

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        test_indices = np.where(test_mask)[0]

        held_out_orientations = sorted(
            set(
                str(orientations[i])
                for i in test_indices
                if str(y[i]) != "EMPTY"
            )
        )

        print(
            f"Train: {len(X_train):4d} windows from {len(train_groups):2d} whole trials"
        )
        print(
            f"Test : {len(X_test):4d} windows from {len(test_groups):2d} whole trials"
        )
        print(
            "Held-out occupied orientation(s): "
            + ", ".join(held_out_orientations)
        )

        model = make_rf(args.seed)
        model.fit(X_train, y_train)
        fold_importances.append(model.feature_importances_.copy())

        pred = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        xy_pred = probability_weighted_xy(probabilities, model.classes_)

        window_xy_errors = np.asarray(
            [
                xy_error_for_label(str(label), xy)
                for label, xy in zip(y_test, xy_pred)
            ],
            dtype=np.float64,
        )

        fold_metrics = standard_metrics(y_test, pred)
        fold_metrics.update(summarize_xy_errors(window_xy_errors))

        # Trial-level predictions for this fold.
        fold_trial_true: list[str] = []
        fold_trial_pred: list[str] = []
        fold_trial_prob_pred: list[str] = []
        fold_trial_xy_errors: list[float] = []

        for group in test_groups:
            global_indices = np.where((groups == group) & test_mask)[0]
            local_indices = np.asarray(
                [np.where(test_indices == gi)[0][0] for gi in global_indices],
                dtype=np.int64,
            )

            true_labels = y[global_indices]
            if len(set(true_labels)) != 1:
                raise RuntimeError(f"Mixed labels inside trial group {group}")
            true_label = str(true_labels[0])

            group_pred = pred[local_indices]
            group_prob = probabilities[local_indices]
            mean_prob = np.mean(group_prob, axis=0)

            majority_pred = majority_vote_with_probability_tiebreak(
                group_pred,
                group_prob,
                model.classes_,
            )
            probability_pred = str(model.classes_[int(np.argmax(mean_prob))])

            group_xy = probability_weighted_xy(
                mean_prob.reshape(1, -1),
                model.classes_,
            )[0]
            group_xy_error = xy_error_for_label(true_label, group_xy)

            first = windows[int(global_indices[0])]
            trial_row = {
                "fold": fold,
                "session": first["session"],
                "zone_true": true_label,
                "trial": int(trial_numbers[int(global_indices[0])]),
                "orientation": first["orientation"],
                "trial_group": group,
                "source_file": first["source_file"],
                "n_windows": len(global_indices),
                "pred_majority": majority_pred,
                "pred_mean_probability": probability_pred,
                "correct_majority": int(majority_pred == true_label),
                "correct_mean_probability": int(probability_pred == true_label),
                "x_pred_m": safe_float(group_xy[0]),
                "y_pred_m": safe_float(group_xy[1]),
                "xy_error_m": safe_float(group_xy_error),
            }

            class_to_index = {cls: i for i, cls in enumerate(model.classes_)}
            for cls in labels_present:
                trial_row[f"p_{cls}"] = (
                    float(mean_prob[class_to_index[cls]])
                    if cls in class_to_index
                    else 0.0
                )

            trial_rows.append(trial_row)
            fold_trial_true.append(true_label)
            fold_trial_pred.append(majority_pred)
            fold_trial_prob_pred.append(probability_pred)
            if np.isfinite(group_xy_error):
                fold_trial_xy_errors.append(float(group_xy_error))

        fold_trial_true_arr = np.asarray(fold_trial_true, dtype=object)
        fold_trial_pred_arr = np.asarray(fold_trial_pred, dtype=object)

        trial_metrics = standard_metrics(
            fold_trial_true_arr,
            fold_trial_pred_arr,
        )
        trial_metrics.update(
            summarize_xy_errors(np.asarray(fold_trial_xy_errors, dtype=np.float64))
        )

        fold_row = {
            "fold": fold,
            "held_out_trial_number": fold,
            "held_out_occupied_orientations": ",".join(held_out_orientations),
            "train_trials": len(train_groups),
            "test_trials": len(test_groups),
            "train_windows": len(X_train),
            "test_windows": len(X_test),
            "window_accuracy": fold_metrics["accuracy"],
            "window_balanced_accuracy": fold_metrics["balanced_accuracy"],
            "window_macro_f1": fold_metrics["macro_f1"],
            "window_xy_mean_m": fold_metrics["xy_mean_m"],
            "window_xy_median_m": fold_metrics["xy_median_m"],
            "window_xy_p90_m": fold_metrics["xy_p90_m"],
            "trial_accuracy": trial_metrics["accuracy"],
            "trial_balanced_accuracy": trial_metrics["balanced_accuracy"],
            "trial_macro_f1": trial_metrics["macro_f1"],
            "trial_xy_mean_m": trial_metrics["xy_mean_m"],
            "trial_xy_median_m": trial_metrics["xy_median_m"],
            "trial_xy_p90_m": trial_metrics["xy_p90_m"],
        }
        fold_rows.append(fold_row)

        print(
            f"Window: accuracy={fold_metrics['accuracy']:.3f} "
            f"macroF1={fold_metrics['macro_f1']:.3f} "
            f"medianXY={fold_metrics['xy_median_m']:.3f} m "
            f"p90XY={fold_metrics['xy_p90_m']:.3f} m"
        )
        print(
            f"Trial : accuracy={trial_metrics['accuracy']:.3f} "
            f"macroF1={trial_metrics['macro_f1']:.3f}"
        )

        # Save one OOF record per test window.
        class_to_index = {cls: i for i, cls in enumerate(model.classes_)}
        for local_i, global_i in enumerate(test_indices):
            row = windows[int(global_i)]
            true_label = str(y[global_i])
            pred_label = str(pred[local_i])

            oof = {
                "fold": fold,
                "session": row["session"],
                "zone_true": true_label,
                "zone_pred": pred_label,
                "correct": int(true_label == pred_label),
                "trial": int(trial_numbers[global_i]),
                "orientation": row["orientation"],
                "trial_group": row["trial_group"],
                "source_file": row["source_file"],
                "window_index": row["window_index"],
                "window_start_s": row["window_start_s"],
                "left_packets": row["left_packets"],
                "right_packets": row["right_packets"],
                "x_pred_m": safe_float(xy_pred[local_i, 0]),
                "y_pred_m": safe_float(xy_pred[local_i, 1]),
                "xy_error_m": safe_float(window_xy_errors[local_i]),
            }

            for cls in labels_present:
                oof[f"p_{cls}"] = (
                    float(probabilities[local_i, class_to_index[cls]])
                    if cls in class_to_index
                    else 0.0
                )

            oof_rows.append(oof)

        all_oof_true.extend([str(v) for v in y_test])
        all_oof_pred.extend([str(v) for v in pred])
        all_oof_xy_errors.extend(
            [float(v) for v in window_xy_errors if np.isfinite(v)]
        )

        all_trial_true.extend(fold_trial_true)
        all_trial_pred.extend(fold_trial_pred)
        all_trial_prob_pred.extend(fold_trial_prob_pred)
        all_trial_xy_errors.extend(fold_trial_xy_errors)

    # ------------------------------------------------------------------
    # Aggregate out-of-fold metrics
    # ------------------------------------------------------------------
    y_oof = np.asarray(all_oof_true, dtype=object)
    pred_oof = np.asarray(all_oof_pred, dtype=object)
    y_trial = np.asarray(all_trial_true, dtype=object)
    pred_trial = np.asarray(all_trial_pred, dtype=object)
    pred_trial_prob = np.asarray(all_trial_prob_pred, dtype=object)

    if len(oof_rows) != len(windows):
        raise RuntimeError(
            f"OOF coverage error: {len(oof_rows)} OOF rows for "
            f"{len(windows)} total windows. Every window should be tested once."
        )

    if len(trial_rows) != len(labels_present) * args.folds:
        raise RuntimeError(
            f"Trial OOF coverage error: {len(trial_rows)} trial predictions; "
            f"expected {len(labels_present) * args.folds}."
        )

    aggregate_window = standard_metrics(y_oof, pred_oof)
    aggregate_window.update(
        summarize_xy_errors(np.asarray(all_oof_xy_errors, dtype=np.float64))
    )

    aggregate_trial_majority = standard_metrics(y_trial, pred_trial)
    aggregate_trial_majority.update(
        summarize_xy_errors(np.asarray(all_trial_xy_errors, dtype=np.float64))
    )

    aggregate_trial_probability = standard_metrics(y_trial, pred_trial_prob)

    fold_summary = {
        "window_accuracy": mean_std([r["window_accuracy"] for r in fold_rows]),
        "window_balanced_accuracy": mean_std(
            [r["window_balanced_accuracy"] for r in fold_rows]
        ),
        "window_macro_f1": mean_std([r["window_macro_f1"] for r in fold_rows]),
        "trial_accuracy": mean_std([r["trial_accuracy"] for r in fold_rows]),
        "trial_balanced_accuracy": mean_std(
            [r["trial_balanced_accuracy"] for r in fold_rows]
        ),
        "trial_macro_f1": mean_std([r["trial_macro_f1"] for r in fold_rows]),
    }

    print("\n" + "=" * 78)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 78)

    for row in fold_rows:
        print(
            f"Fold {row['fold']}: "
            f"window acc={row['window_accuracy']:.3f}, "
            f"macroF1={row['window_macro_f1']:.3f}, "
            f"trial acc={row['trial_accuracy']:.3f}, "
            f"held-out orientation={row['held_out_occupied_orientations']}"
        )

    print("\nMean ± SD across folds")
    print(
        f"  Window accuracy : "
        f"{fold_summary['window_accuracy']['mean']:.3f} ± "
        f"{fold_summary['window_accuracy']['std']:.3f}"
    )
    print(
        f"  Window macro F1 : "
        f"{fold_summary['window_macro_f1']['mean']:.3f} ± "
        f"{fold_summary['window_macro_f1']['std']:.3f}"
    )
    print(
        f"  Trial accuracy  : "
        f"{fold_summary['trial_accuracy']['mean']:.3f} ± "
        f"{fold_summary['trial_accuracy']['std']:.3f}"
    )

    print("\nAggregate OOF — every sample tested exactly once")
    print(
        f"  Windows ({len(y_oof)}): accuracy={aggregate_window['accuracy']:.3f}, "
        f"macroF1={aggregate_window['macro_f1']:.3f}, "
        f"medianXY={aggregate_window['xy_median_m']:.3f} m, "
        f"p90XY={aggregate_window['xy_p90_m']:.3f} m"
    )
    print(
        f"  Trials  ({len(y_trial)}): majority-vote accuracy="
        f"{aggregate_trial_majority['accuracy']:.3f}, "
        f"macroF1={aggregate_trial_majority['macro_f1']:.3f}"
    )
    print(
        f"  Trials  ({len(y_trial)}): mean-probability accuracy="
        f"{aggregate_trial_probability['accuracy']:.3f}"
    )

    # ------------------------------------------------------------------
    # Save fold metrics
    # ------------------------------------------------------------------
    fold_metric_fields = list(fold_rows[0].keys())
    write_csv(
        output_dir / "crossval_fold_metrics.csv",
        fold_metric_fields,
        fold_rows,
    )

    # ------------------------------------------------------------------
    # Save OOF window predictions
    # ------------------------------------------------------------------
    oof_fields = [
        "fold",
        "session",
        "zone_true",
        "zone_pred",
        "correct",
        "trial",
        "orientation",
        "trial_group",
        "source_file",
        "window_index",
        "window_start_s",
        "left_packets",
        "right_packets",
        "x_pred_m",
        "y_pred_m",
        "xy_error_m",
        *[f"p_{cls}" for cls in labels_present],
    ]
    write_csv(
        output_dir / "crossval_oof_predictions.csv",
        oof_fields,
        oof_rows,
    )

    # ------------------------------------------------------------------
    # Save trial predictions
    # ------------------------------------------------------------------
    trial_fields = [
        "fold",
        "session",
        "zone_true",
        "trial",
        "orientation",
        "trial_group",
        "source_file",
        "n_windows",
        "pred_majority",
        "pred_mean_probability",
        "correct_majority",
        "correct_mean_probability",
        "x_pred_m",
        "y_pred_m",
        "xy_error_m",
        *[f"p_{cls}" for cls in labels_present],
    ]
    write_csv(
        output_dir / "crossval_trial_predictions.csv",
        trial_fields,
        trial_rows,
    )

    # ------------------------------------------------------------------
    # Confusion matrices + reports
    # ------------------------------------------------------------------
    cm_window = save_confusion_plot(
        y_oof,
        pred_oof,
        labels_present,
        output_dir / "crossval_confusion_matrix.png",
        f"RF 5-fold OOF confusion matrix — {args.session} (window level)",
    )
    save_confusion_plot(
        y_oof,
        pred_oof,
        labels_present,
        output_dir / "crossval_confusion_matrix_normalized.png",
        f"RF 5-fold OOF normalized confusion — {args.session} (window level)",
        normalize=True,
    )
    cm_trial = save_confusion_plot(
        y_trial,
        pred_trial,
        labels_present,
        output_dir / "crossval_trial_confusion_matrix.png",
        f"RF 5-fold OOF confusion matrix — {args.session} (trial majority vote)",
    )

    window_report = classification_report(
        y_oof,
        pred_oof,
        labels=labels_present,
        zero_division=0,
    )
    (output_dir / "crossval_classification_report_window.txt").write_text(
        window_report,
        encoding="utf-8",
    )

    trial_report = classification_report(
        y_trial,
        pred_trial,
        labels=labels_present,
        zero_division=0,
    )
    (output_dir / "crossval_classification_report_trial.txt").write_text(
        trial_report,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Per-zone recall
    # ------------------------------------------------------------------
    per_zone_rows = []
    for i, label in enumerate(labels_present):
        window_support = int(np.sum(cm_window[i]))
        trial_support = int(np.sum(cm_trial[i]))
        window_recall = (
            float(cm_window[i, i] / window_support)
            if window_support
            else float("nan")
        )
        trial_recall = (
            float(cm_trial[i, i] / trial_support)
            if trial_support
            else float("nan")
        )
        per_zone_rows.append(
            {
                "zone": label,
                "window_correct": int(cm_window[i, i]),
                "window_support": window_support,
                "window_recall": window_recall,
                "trial_correct": int(cm_trial[i, i]),
                "trial_support": trial_support,
                "trial_recall": trial_recall,
            }
        )

    write_csv(
        output_dir / "crossval_per_zone_recall.csv",
        [
            "zone",
            "window_correct",
            "window_support",
            "window_recall",
            "trial_correct",
            "trial_support",
            "trial_recall",
        ],
        per_zone_rows,
    )

    # ------------------------------------------------------------------
    # Orientation-specific results
    # ------------------------------------------------------------------
    orientation_rows: list[dict[str, Any]] = []
    oof_orientations = sorted(set(row["orientation"] for row in oof_rows))

    for orientation in oof_orientations:
        wr = [row for row in oof_rows if row["orientation"] == orientation]
        tr = [row for row in trial_rows if row["orientation"] == orientation]

        w_true = np.asarray([row["zone_true"] for row in wr], dtype=object)
        w_pred = np.asarray([row["zone_pred"] for row in wr], dtype=object)
        t_true = np.asarray([row["zone_true"] for row in tr], dtype=object)
        t_pred = np.asarray([row["pred_majority"] for row in tr], dtype=object)

        row_out: dict[str, Any] = {
            "orientation": orientation,
            "window_count": len(wr),
            "trial_count": len(tr),
            "window_accuracy": (
                float(accuracy_score(w_true, w_pred)) if len(wr) else None
            ),
            "window_macro_f1": (
                float(f1_score(w_true, w_pred, average="macro", zero_division=0))
                if len(wr)
                else None
            ),
            "trial_accuracy": (
                float(accuracy_score(t_true, t_pred)) if len(tr) else None
            ),
            "trial_macro_f1": (
                float(f1_score(t_true, t_pred, average="macro", zero_division=0))
                if len(tr)
                else None
            ),
        }
        orientation_rows.append(row_out)

    write_csv(
        output_dir / "crossval_orientation_metrics.csv",
        [
            "orientation",
            "window_count",
            "trial_count",
            "window_accuracy",
            "window_macro_f1",
            "trial_accuracy",
            "trial_macro_f1",
        ],
        orientation_rows,
    )

    # ------------------------------------------------------------------
    # Cross-validated feature importance
    # ------------------------------------------------------------------
    importance_matrix = np.vstack(fold_importances)
    importance_mean = np.mean(importance_matrix, axis=0)
    importance_std = np.std(importance_matrix, axis=0, ddof=1)
    order = np.argsort(importance_mean)[::-1]

    importance_rows = []
    for rank, feature_index in enumerate(order, 1):
        importance_rows.append(
            {
                "rank": rank,
                "feature": feature_names[feature_index],
                "mean_importance": float(importance_mean[feature_index]),
                "std_importance": float(importance_std[feature_index]),
            }
        )

    write_csv(
        output_dir / "crossval_feature_importance.csv",
        ["rank", "feature", "mean_importance", "std_importance"],
        importance_rows,
    )

    top = order[:25]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    positions = np.arange(len(top))
    ax.barh(positions, importance_mean[top][::-1])
    ax.set_yticks(
        positions,
        [feature_names[i] for i in top][::-1],
        fontsize=7,
    )
    ax.set_xlabel("Mean Random Forest importance across 5 folds")
    ax.set_title(f"Cross-validated top localization features — {args.session}")
    fig.tight_layout()
    fig.savefig(
        output_dir / "crossval_feature_importance_top25.png",
        dpi=180,
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # JSON summary
    # ------------------------------------------------------------------
    summary = {
        "session": args.session,
        "session_dir": str(session_dir),
        "folds": args.folds,
        "split_rule": "test_trial_number_equals_fold_number",
        "window_seconds": args.window,
        "hop_seconds": args.hop,
        "min_packets_per_link": MIN_PACKETS_PER_LINK,
        "classes": labels_present,
        "independent_trials": len(set(groups)),
        "total_windows": len(windows),
        "fold_mean_std": fold_summary,
        "aggregate_oof_window": aggregate_window,
        "aggregate_oof_trial_majority_vote": aggregate_trial_majority,
        "aggregate_oof_trial_mean_probability": aggregate_trial_probability,
        "fold_metrics": fold_rows,
    }

    with (output_dir / "crossval_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    # ------------------------------------------------------------------
    # Train final deployment model on ALL session data, without using it for
    # evaluation. This preserves CV validity while giving the user a model
    # that benefits from all 50 recordings.
    # ------------------------------------------------------------------
    full_model_path = MODEL_DIR / f"localizer_rf_{args.session.lower()}_full.joblib"

    if not args.skip_full_model:
        print("\nTraining final deployment RF on ALL session windows...")
        full_rf = make_rf(args.seed)
        full_rf.fit(X, y)

        bundle = {
            "model": full_rf,
            "feature_names": feature_names,
            "classes": list(full_rf.classes_),
            "zone_centers": ZONE_CENTERS,
            "window_seconds": args.window,
            "hop_seconds": args.hop,
            "min_packets_per_link": MIN_PACKETS_PER_LINK,
            "room_width_m": ROOM_WIDTH_M,
            "room_height_m": ROOM_HEIGHT_M,
            "training_session": args.session,
            "training_trials": int(len(set(groups))),
            "training_windows": int(len(X)),
            "evaluation_note": (
                "Deployment model trained on all session data AFTER 5-fold CV. "
                "Do not use training-set performance as an evaluation metric."
            ),
        }
        joblib.dump(bundle, full_model_path)

    print("\nSAVED")
    print(f"  Results folder : {output_dir}")
    print(f"  Fold metrics   : {output_dir / 'crossval_fold_metrics.csv'}")
    print(f"  Summary        : {output_dir / 'crossval_summary.json'}")
    print(f"  Window matrix  : {output_dir / 'crossval_confusion_matrix.png'}")
    print(f"  Trial matrix   : {output_dir / 'crossval_trial_confusion_matrix.png'}")
    print(f"  Zone recall    : {output_dir / 'crossval_per_zone_recall.csv'}")
    print(f"  OOF windows    : {output_dir / 'crossval_oof_predictions.csv'}")
    print(f"  OOF trials     : {output_dir / 'crossval_trial_predictions.csv'}")
    print(f"  Orientation    : {output_dir / 'crossval_orientation_metrics.csv'}")
    print(f"  Feature imp.   : {output_dir / 'crossval_feature_importance.csv'}")
    if not args.skip_full_model:
        print(f"  Full S1 model  : {full_model_path}")

    print("\nIMPORTANT:")
    print("  The cross-validation metrics come ONLY from held-out whole trials.")
    print("  The final full-session model is for deployment/live testing, not")
    print("  for calculating the reported CV accuracy.")


if __name__ == "__main__":
    main()
