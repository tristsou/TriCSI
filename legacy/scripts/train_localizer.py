from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def gather_windows(
    paths: list[Path],
    window_seconds: float,
    hop_seconds: float,
) -> list[dict]:
    rows = []
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


def stratified_trial_holdout(
    labels: np.ndarray,
    groups: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Put entire trials in train or test, while forcing every class to have
    held-out trials. This avoids leakage from overlapping windows.
    """
    rng = np.random.default_rng(seed)
    test_groups = set()

    for label in sorted(set(labels)):
        class_groups = sorted(set(groups[labels == label]))
        if len(class_groups) < 2:
            raise ValueError(
                f"Class {label} has only {len(class_groups)} independent trial(s). "
                "Collect at least 2; 5+ is strongly recommended."
            )
        class_groups = np.asarray(class_groups, dtype=object)
        rng.shuffle(class_groups)
        n_test = max(1, int(round(len(class_groups) * test_fraction)))
        n_test = min(n_test, len(class_groups) - 1)
        test_groups.update(class_groups[:n_test].tolist())

    test_mask = np.asarray([g in test_groups for g in groups], dtype=bool)
    train_mask = ~test_mask
    return train_mask, test_mask


def probability_weighted_xy(
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    centers = []
    indices = []
    for index, cls in enumerate(classes):
        if cls in ZONE_CENTERS:
            indices.append(index)
            centers.append(ZONE_CENTERS[cls])

    centers = np.asarray(centers, dtype=np.float64)
    p = probabilities[:, indices]
    denom = np.sum(p, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    p = p / denom
    return p @ centers


def coordinate_errors(
    y_true: np.ndarray,
    xy_pred: np.ndarray,
) -> np.ndarray:
    errors = []
    for label, pred in zip(y_true, xy_pred):
        if label not in ZONE_CENTERS:
            continue
        true_xy = np.asarray(ZONE_CENTERS[label], dtype=np.float64)
        errors.append(float(np.linalg.norm(pred - true_xy)))
    return np.asarray(errors, dtype=np.float64)


def evaluate_model(name, model, X_test, y_test, labels, out_dir: Path) -> dict:
    pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }

    xy_pred = probability_weighted_xy(probabilities, model.classes_)
    errors = coordinate_errors(y_test, xy_pred)
    if len(errors):
        metrics.update(
            {
                "xy_mean_m": float(np.mean(errors)),
                "xy_median_m": float(np.median(errors)),
                "xy_p90_m": float(np.percentile(errors, 90)),
            }
        )

    report = classification_report(
        y_test,
        pred,
        labels=labels,
        zero_division=0,
    )
    (out_dir / f"classification_report_{name}.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm)
    ax.set_title(f"{name} confusion matrix")
    ax.set_xlabel("Predicted zone")
    ax.set_ylabel("True zone")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{name}.png", dpi=170)
    plt.close(fig)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TriCSI 3x3 localization baselines.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--window", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--hop", type=float, default=HOP_SECONDS)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-session",
        default=None,
        help="Optional stronger evaluation: hold out one whole session, e.g. S2.",
    )
    args = parser.parse_args()

    trial_paths = sorted(
        p for p in args.raw_dir.rglob("*.csv") if not p.name.startswith("schedule_")
    )
    if not trial_paths:
        raise SystemExit(f"No raw trials found under {args.raw_dir}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    windows = gather_windows(trial_paths, args.window, args.hop)
    if not windows:
        raise SystemExit("No valid dual-link windows were created.")

    feature_names = sorted(windows[0]["features"])
    X = np.asarray(
        [[row["features"][name] for name in feature_names] for row in windows],
        dtype=np.float64,
    )
    y = np.asarray([row["zone"] for row in windows], dtype=object)
    groups = np.asarray([row["trial_group"] for row in windows], dtype=object)
    sessions = np.asarray([row["session"] for row in windows], dtype=object)

    labels_present = [label for label in CLASS_ORDER if label in set(y)]

    print("\nWINDOW COUNTS")
    for label in labels_present:
        print(f"  {label:5s}: {int(np.sum(y == label))}")

    if args.test_session:
        test_mask = sessions == args.test_session
        train_mask = ~test_mask
        if not np.any(test_mask):
            raise SystemExit(f"No windows found for test session {args.test_session}")
        missing_train = set(labels_present) - set(y[train_mask])
        missing_test = set(labels_present) - set(y[test_mask])
        if missing_train or missing_test:
            raise SystemExit(
                f"Session split missing classes. train missing={missing_train}; "
                f"test missing={missing_test}"
            )
    else:
        train_mask, test_mask = stratified_trial_holdout(
            y, groups, args.test_fraction, args.seed
        )

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\nTrain windows: {len(X_train)} from {len(set(groups[train_mask]))} whole trials")
    print(f"Test windows : {len(X_test)} from {len(set(groups[test_mask]))} whole trials")

    rf = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    k = max(3, min(9, int(math.sqrt(len(X_train))) // 2 * 2 + 1))
    knn = Pipeline(
        [
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ]
    )
    knn.fit(X_train, y_train)

    metrics = {
        "random_forest": evaluate_model(
            "random_forest", rf, X_test, y_test, labels_present, RESULTS_DIR
        ),
        "knn": evaluate_model(
            "knn", knn, X_test, y_test, labels_present, RESULTS_DIR
        ),
        "split": {
            "test_session": args.test_session,
            "test_fraction": args.test_fraction,
            "train_windows": int(len(X_train)),
            "test_windows": int(len(X_test)),
            "train_trials": int(len(set(groups[train_mask]))),
            "test_trials": int(len(set(groups[test_mask]))),
        },
    }

    print("\nRESULTS")
    for model_name in ("random_forest", "knn"):
        m = metrics[model_name]
        print(
            f"{model_name:14s} accuracy={m['accuracy']:.3f} "
            f"macroF1={m['macro_f1']:.3f} "
            f"median_xy={m.get('xy_median_m', float('nan')):.3f} m "
            f"p90_xy={m.get('xy_p90_m', float('nan')):.3f} m"
        )

    # Save RF as the primary live model. We report kNN as a baseline rather
    # than selecting a model based on the held-out test set.
    bundle = {
        "model": rf,
        "feature_names": feature_names,
        "classes": list(rf.classes_),
        "zone_centers": ZONE_CENTERS,
        "window_seconds": args.window,
        "hop_seconds": args.hop,
        "min_packets_per_link": MIN_PACKETS_PER_LINK,
        "room_width_m": ROOM_WIDTH_M,
        "room_height_m": ROOM_HEIGHT_M,
    }
    joblib.dump(bundle, MODEL_DIR / "localizer_rf.joblib")
    joblib.dump(
        {
            **bundle,
            "model": knn,
            "classes": list(knn.classes_),
        },
        MODEL_DIR / "localizer_knn.joblib",
    )

    # Feature importance is scientifically useful: it tells you whether RSSI,
    # static subcarrier shape, or motion descriptors are carrying the result.
    importance = rf.feature_importances_
    order = np.argsort(importance)[::-1]
    with (RESULTS_DIR / "feature_importance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "feature", "importance"])
        for rank, index in enumerate(order, 1):
            writer.writerow([rank, feature_names[index], float(importance[index])])

    top = order[:25]
    fig, ax = plt.subplots(figsize=(10, 7))
    positions = np.arange(len(top))
    ax.barh(positions, importance[top][::-1])
    ax.set_yticks(positions, [feature_names[i] for i in top][::-1], fontsize=7)
    ax.set_xlabel("Random Forest importance")
    ax.set_title("Top localization features")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance_top25.png", dpi=170)
    plt.close(fig)

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    # Reproducible window-level feature table.
    feature_table = RESULTS_DIR / "window_features.csv"
    metadata_cols = [
        "zone", "x_m", "y_m", "session", "trial", "orientation",
        "trial_group", "source_file", "window_index", "window_start_s",
        "left_packets", "right_packets",
    ]
    with feature_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_cols + feature_names)
        writer.writeheader()
        for row in windows:
            out = {key: row[key] for key in metadata_cols}
            out.update(row["features"])
            writer.writerow(out)

    print(f"\nSaved primary model: {MODEL_DIR / 'localizer_rf.joblib'}")
    print(f"Saved metrics      : {RESULTS_DIR / 'metrics.json'}")
    print(f"Saved confusion    : {RESULTS_DIR / 'confusion_matrix_random_forest.png'}")
    print(f"Saved feature table: {feature_table}")
    print("\nNext:")
    print("  py -3.14 live_localize.py")


if __name__ == "__main__":
    main()
