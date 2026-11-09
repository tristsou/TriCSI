from __future__ import annotations

"""
Train the existing engineered-feature RF on S1+S2 and evaluate once on unseen S3.

Run:
    python evaluate_s1s2_to_s3_rf.py

Expected:
    data/raw/S1/*.csv
    data/raw/S2/*.csv
    data/raw/S3/*.csv

IMPORTANT:
Do not tune preprocessing/features/hyperparameters using S3 before this run
if you want S3 to remain an untouched final test session.
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_config import CLASS_ORDER, RAW_DATA_DIR, RESULTS_DIR, MODEL_DIR, WINDOW_SECONDS, HOP_SECONDS
from train_crossval import gather_windows, make_rf, trial_number_from_row
from train_csi_cnn import trial_paths_for_session


def load_rows(raw_dir: Path, session: str, window: float, hop: float):
    paths = trial_paths_for_session(raw_dir, session)
    if not paths:
        raise SystemExit(f"No CSVs found for {session}")

    print(f"Loading {session}: {len(paths)} trials")
    rows = gather_windows(paths, window_seconds=window, hop_seconds=hop)

    for r in rows:
        r["_session"] = session

    print(f"  windows: {len(rows)}")
    return rows


def feature_matrix(rows, names):
    X = np.asarray(
        [[float(r["features"][n]) for n in names] for r in rows],
        dtype=np.float64,
    )
    y = np.asarray([str(r["zone"]) for r in rows], dtype=object)
    return X, y


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
    }


def trial_group(r):
    return (
        f"{r['_session']}|{r['zone']}|"
        f"{trial_number_from_row(r)}|{r['orientation']}"
    )


def aggregate_trials(test_rows, pred, prob, classes):
    groups = np.asarray([trial_group(r) for r in test_rows], dtype=object)
    y = np.asarray([str(r["zone"]) for r in test_rows], dtype=object)

    rows = []
    yt, maj, mp = [], [], []

    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        true = str(y[idx[0]])

        vals, counts = np.unique(pred[idx], return_counts=True)
        majority = str(vals[np.argmax(counts)])

        mean_prob = prob[idx].mean(axis=0)
        mean_pred = str(classes[int(np.argmax(mean_prob))])

        sample = test_rows[int(idx[0])]

        rows.append(
            {
                "trial_group": g,
                "zone_true": true,
                "trial": int(trial_number_from_row(sample)),
                "orientation": str(sample["orientation"]),
                "n_windows": int(len(idx)),
                "pred_majority": majority,
                "pred_mean_probability": mean_pred,
                "majority_correct": int(majority == true),
                "mean_probability_correct": int(mean_pred == true),
                "mean_probability_confidence": float(np.max(mean_prob)),
            }
        )

        yt.append(true)
        maj.append(majority)
        mp.append(mean_pred)

    return (
        rows,
        np.asarray(yt, dtype=object),
        np.asarray(maj, dtype=object),
        np.asarray(mp, dtype=object),
    )


def save_confusion(y_true, y_pred, path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    denom = cm.sum(axis=1, keepdims=True)
    shown = np.divide(
        cm.astype(float),
        denom,
        out=np.zeros_like(cm, dtype=float),
        where=denom != 0,
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(shown)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)

    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            ax.text(j, i, f"{shown[i,j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--window", type=float, default=WINDOW_SECONDS)
    p.add_argument("--hop", type=float, default=HOP_SECONDS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("\n" + "=" * 78)
    print("TriCSI S1+S2 -> S3 UNSEEN-SESSION RANDOM FOREST")
    print("=" * 78)

    s1 = load_rows(args.raw_dir, "S1", args.window, args.hop)
    s2 = load_rows(args.raw_dir, "S2", args.window, args.hop)
    s3 = load_rows(args.raw_dir, "S3", args.window, args.hop)

    train_rows = s1 + s2
    test_rows = s3

    feature_names = sorted(train_rows[0]["features"])
    if sorted(test_rows[0]["features"]) != feature_names:
        raise RuntimeError("S3 feature names differ from S1/S2.")

    Xtr, ytr = feature_matrix(train_rows, feature_names)
    Xte, yte = feature_matrix(test_rows, feature_names)

    print(f"\nTrain windows: {len(train_rows)}")
    print(f"Test windows : {len(test_rows)}")
    print(f"Features     : {len(feature_names)}")

    print("\nTraining RF on ALL S1+S2 windows...")
    model = make_rf(args.seed)
    model.fit(Xtr, ytr)

    print("Evaluating ONCE on S3...")
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)

    window_m = metrics(yte, pred)

    trial_rows, trial_true, trial_majority, trial_mean_prob = aggregate_trials(
        test_rows,
        pred,
        prob,
        model.classes_,
    )

    majority_m = metrics(trial_true, trial_majority)
    mean_prob_m = metrics(trial_true, trial_mean_prob)

    zone_rows = []
    for zone in CLASS_ORDER:
        idx = np.where(yte == zone)[0]
        recall = float(np.mean(pred[idx] == zone)) if len(idx) else None
        zone_rows.append(
            {
                "zone": zone,
                "n_windows": int(len(idx)),
                "recall": recall,
            }
        )

    out_dir = RESULTS_DIR / "rf_s1_s2_to_s3"
    out_dir.mkdir(parents=True, exist_ok=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "rf_s1_s2_to_s3.joblib"

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "train_sessions": ["S1", "S2"],
            "test_session": "S3",
            "seed": args.seed,
        },
        model_path,
    )

    write_csv(out_dir / "trial_predictions.csv", trial_rows)
    write_csv(out_dir / "per_zone_recall.csv", zone_rows)

    save_confusion(
        yte,
        pred,
        out_dir / "confusion_windows.png",
        "RF S1+S2 -> S3 window predictions",
    )

    save_confusion(
        trial_true,
        trial_mean_prob,
        out_dir / "confusion_trials.png",
        "RF S1+S2 -> S3 trial mean-probability predictions",
    )

    summary = {
        "train_sessions": ["S1", "S2"],
        "test_session": "S3",
        "train_windows": len(train_rows),
        "test_windows": len(test_rows),
        "n_features": len(feature_names),
        "window_metrics": window_m,
        "trial_majority_metrics": majority_m,
        "trial_mean_probability_metrics": mean_prob_m,
        "per_zone_recall": zone_rows,
        "model_path": str(model_path),
        "note": (
            "S3 is an untouched final test only if no preprocessing/model "
            "choices were changed using S3 before this run."
        ),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("S1+S2 -> S3 UNSEEN-SESSION RF RESULTS")
    print("=" * 78)
    print(
        f"Window level: accuracy={window_m['accuracy']:.3f} "
        f"macroF1={window_m['macro_f1']:.3f}"
    )
    print(
        f"Trial majority: accuracy={majority_m['accuracy']:.3f} "
        f"macroF1={majority_m['macro_f1']:.3f}"
    )
    print(
        f"Trial mean probability: accuracy={mean_prob_m['accuracy']:.3f} "
        f"macroF1={mean_prob_m['macro_f1']:.3f}"
    )

    print("\nPer-zone window recall:")
    for r in zone_rows:
        value = "NA" if r["recall"] is None else f"{r['recall']:.3f}"
        print(f"  {r['zone']:5s}: {value}")

    print(f"\nSaved model  : {model_path}")
    print(f"Saved results: {out_dir}")
    print(
        "\nIMPORTANT: after you inspect this result, do not tune against S3 "
        "unless you are willing to treat S3 as development data and reserve S4."
    )


if __name__ == "__main__":
    main()
