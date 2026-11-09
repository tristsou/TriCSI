from __future__ import annotations

"""
Analyze spatial errors for the already-frozen S4+S5 -> S6 RF model.

This script DOES NOT retrain anything.

It loads:
    models/rf_s4_s5_to_s6.joblib

and evaluates S6 to produce:
- window confusion matrix
- trial-level confusion matrix
- per-zone trial accuracy
- median / p90 physical localization error
- exact-or-adjacent-zone accuracy
- detailed trial prediction CSV

Run:
    python analyze_s6_errors.py
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from project_config import (
    CLASS_ORDER,
    RAW_DATA_DIR,
    RESULTS_DIR,
    MODEL_DIR,
    WINDOW_SECONDS,
    HOP_SECONDS,
)

from train_crossval import gather_windows, trial_number_from_row
from train_csi_cnn import trial_paths_for_session


# ---------------------------------------------------------------------------
# 3x3 grid geometry
# ---------------------------------------------------------------------------

# Existing tape/grid centers, meters.
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

# Grid indices for "one-zone away" calculation.
GRID_POS = {
    "Z1": (0, 0),
    "Z2": (1, 0),
    "Z3": (2, 0),
    "Z4": (0, 1),
    "Z5": (1, 1),
    "Z6": (2, 1),
    "Z7": (0, 2),
    "Z8": (1, 2),
    "Z9": (2, 2),
}


def load_s6_rows(raw_dir: Path, window: float, hop: float):
    paths = trial_paths_for_session(raw_dir, "S6")
    if not paths:
        raise SystemExit("No S6 trial CSVs found.")

    rows = gather_windows(
        paths,
        window_seconds=window,
        hop_seconds=hop,
    )

    for r in rows:
        r["_session"] = "S6"

    print(f"S6: {len(paths)} trials, {len(rows)} windows")
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
        f"S6|{r['zone']}|"
        f"{trial_number_from_row(r)}|{r['orientation']}"
    )


def physical_error_m(true_zone: str, pred_zone: str):
    if true_zone not in ZONE_CENTERS or pred_zone not in ZONE_CENTERS:
        return None

    a = np.asarray(ZONE_CENTERS[true_zone], dtype=float)
    b = np.asarray(ZONE_CENTERS[pred_zone], dtype=float)
    return float(np.linalg.norm(a - b))


def exact_or_adjacent(true_zone: str, pred_zone: str):
    """
    8-neighbor adjacency:
    exact zone or any horizontally/vertically/diagonally adjacent grid cell.
    EMPTY is not treated as spatially adjacent to occupied zones.
    """
    if true_zone == "EMPTY":
        return int(pred_zone == "EMPTY")

    if true_zone not in GRID_POS or pred_zone not in GRID_POS:
        return 0

    tx, ty = GRID_POS[true_zone]
    px, py = GRID_POS[pred_zone]

    return int(max(abs(tx - px), abs(ty - py)) <= 1)


def save_confusion(y_true, y_pred, path: Path, title: str):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    denom = cm.sum(axis=1, keepdims=True)
    norm = np.divide(
        cm.astype(float),
        denom,
        out=np.zeros_like(cm, dtype=float),
        where=denom != 0,
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(
        range(len(CLASS_ORDER)),
        labels=CLASS_ORDER,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(
        range(len(CLASS_ORDER)),
        labels=CLASS_ORDER,
    )

    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows):
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
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "rf_s4_s5_to_s6.joblib",
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

    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Frozen model not found: {args.model}"
        )

    print("\n" + "=" * 78)
    print("S6 SPATIAL ERROR ANALYSIS — FROZEN S4+S5 RF")
    print("=" * 78)
    print(f"Model: {args.model}")
    print("Retraining: NO")
    print("=" * 78)

    bundle = joblib.load(args.model)

    rf = bundle["model"]
    feature_names = bundle["feature_names"]

    rows = load_s6_rows(
        args.raw_dir,
        args.window,
        args.hop,
    )

    X, y_true_window = feature_matrix(
        rows,
        feature_names,
    )

    pred_window = rf.predict(X)
    prob_window = rf.predict_proba(X)

    # ------------------------------------------------------------------
    # Trial aggregation
    # ------------------------------------------------------------------

    groups = np.asarray(
        [trial_group(r) for r in rows],
        dtype=object,
    )

    trial_rows = []
    trial_true = []
    trial_pred = []

    for group in sorted(set(groups)):
        idx = np.where(groups == group)[0]

        true_zone = str(y_true_window[idx[0]])
        mean_prob = prob_window[idx].mean(axis=0)
        pred_zone = str(
            rf.classes_[int(np.argmax(mean_prob))]
        )

        sample = rows[int(idx[0])]

        err_m = physical_error_m(
            true_zone,
            pred_zone,
        )

        trial_rows.append(
            {
                "trial_group": group,
                "true_zone": true_zone,
                "pred_zone": pred_zone,
                "trial": int(
                    trial_number_from_row(sample)
                ),
                "orientation": str(
                    sample["orientation"]
                ),
                "n_windows": int(len(idx)),
                "correct": int(
                    pred_zone == true_zone
                ),
                "exact_or_adjacent": exact_or_adjacent(
                    true_zone,
                    pred_zone,
                ),
                "physical_error_m": err_m,
                "physical_error_ft": (
                    None
                    if err_m is None
                    else err_m * 3.28084
                ),
                "confidence": float(
                    np.max(mean_prob)
                ),
            }
        )

        trial_true.append(true_zone)
        trial_pred.append(pred_zone)

    trial_true = np.asarray(
        trial_true,
        dtype=object,
    )
    trial_pred = np.asarray(
        trial_pred,
        dtype=object,
    )

    # ------------------------------------------------------------------
    # Spatial metrics
    # ------------------------------------------------------------------

    occupied_trial_errors = np.asarray(
        [
            r["physical_error_m"]
            for r in trial_rows
            if r["physical_error_m"] is not None
        ],
        dtype=float,
    )

    window_errors = []

    for true_zone, pred_zone in zip(
        y_true_window,
        pred_window,
    ):
        e = physical_error_m(
            str(true_zone),
            str(pred_zone),
        )

        if e is not None:
            window_errors.append(e)

    window_errors = np.asarray(
        window_errors,
        dtype=float,
    )

    trial_exact_acc = float(
        np.mean(
            [
                r["correct"]
                for r in trial_rows
            ]
        )
    )

    trial_adjacent_acc = float(
        np.mean(
            [
                r["exact_or_adjacent"]
                for r in trial_rows
            ]
        )
    )

    # Occupied-only adjacent accuracy, since EMPTY has no coordinate.
    occupied_trial_rows = [
        r for r in trial_rows
        if r["true_zone"] != "EMPTY"
    ]

    occupied_adjacent_acc = float(
        np.mean(
            [
                r["exact_or_adjacent"]
                for r in occupied_trial_rows
            ]
        )
    )

    # ------------------------------------------------------------------
    # Per-zone trial accuracy
    # ------------------------------------------------------------------

    per_zone = []

    for zone in CLASS_ORDER:
        zrows = [
            r for r in trial_rows
            if r["true_zone"] == zone
        ]

        if not zrows:
            continue

        per_zone.append(
            {
                "zone": zone,
                "n_trials": len(zrows),
                "trial_accuracy": float(
                    np.mean(
                        [
                            r["correct"]
                            for r in zrows
                        ]
                    )
                ),
                "exact_or_adjacent_accuracy": float(
                    np.mean(
                        [
                            r["exact_or_adjacent"]
                            for r in zrows
                        ]
                    )
                ),
                "median_physical_error_m": (
                    None
                    if zone == "EMPTY"
                    else float(
                        np.median(
                            [
                                r["physical_error_m"]
                                for r in zrows
                                if r["physical_error_m"] is not None
                            ]
                        )
                    )
                ),
            }
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    out_dir = RESULTS_DIR / "s6_error_analysis"
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        out_dir / "trial_predictions_spatial.csv",
        trial_rows,
    )

    write_csv(
        out_dir / "per_zone_trial_metrics.csv",
        per_zone,
    )

    save_confusion(
        y_true_window,
        pred_window,
        out_dir / "confusion_windows.png",
        "S6 window confusion — frozen S4+S5 RF",
    )

    save_confusion(
        trial_true,
        trial_pred,
        out_dir / "confusion_trials.png",
        "S6 trial confusion — frozen S4+S5 RF",
    )

    summary = {
        "trial_exact_accuracy": trial_exact_acc,
        "trial_exact_or_adjacent_accuracy_all_classes": trial_adjacent_acc,
        "trial_exact_or_adjacent_accuracy_occupied_only": occupied_adjacent_acc,
        "trial_median_physical_error_m_occupied_only": float(
            np.median(
                occupied_trial_errors
            )
        ),
        "trial_p90_physical_error_m_occupied_only": float(
            np.percentile(
                occupied_trial_errors,
                90,
            )
        ),
        "trial_median_physical_error_ft_occupied_only": float(
            np.median(
                occupied_trial_errors
            )
            * 3.28084
        ),
        "trial_p90_physical_error_ft_occupied_only": float(
            np.percentile(
                occupied_trial_errors,
                90,
            )
            * 3.28084
        ),
        "window_median_physical_error_m_occupied_only": float(
            np.median(
                window_errors
            )
        ),
        "window_p90_physical_error_m_occupied_only": float(
            np.percentile(
                window_errors,
                90,
            )
        ),
        "per_zone_trial_metrics": per_zone,
        "note": (
            "Physical-error metrics exclude EMPTY because EMPTY "
            "has no spatial coordinate. Adjacent-zone metric uses "
            "8-neighbor adjacency: exact, horizontal, vertical, or diagonal."
        ),
    }

    with (
        out_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Print concise final summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("S6 SPATIAL ERROR SUMMARY")
    print("=" * 78)

    print(
        f"Exact trial accuracy: "
        f"{trial_exact_acc:.3f}"
    )

    print(
        f"Exact-or-adjacent trial accuracy "
        f"(occupied zones): "
        f"{occupied_adjacent_acc:.3f}"
    )

    print(
        f"Median trial localization error: "
        f"{np.median(occupied_trial_errors):.3f} m "
        f"({np.median(occupied_trial_errors)*3.28084:.2f} ft)"
    )

    print(
        f"90th-percentile trial error: "
        f"{np.percentile(occupied_trial_errors, 90):.3f} m "
        f"({np.percentile(occupied_trial_errors, 90)*3.28084:.2f} ft)"
    )

    print("\nPer-zone TRIAL accuracy:")

    for r in per_zone:
        med = (
            "NA"
            if r["median_physical_error_m"] is None
            else f"{r['median_physical_error_m']:.3f} m"
        )

        print(
            f"  {r['zone']:5s}: "
            f"exact={r['trial_accuracy']:.3f} | "
            f"adjacent={r['exact_or_adjacent_accuracy']:.3f} | "
            f"median error={med}"
        )

    print(f"\nSaved: {out_dir}")
    print(
        "Open confusion_trials.png first to see where the incorrect "
        "S6 trials are actually going."
    )


if __name__ == "__main__":
    main()
