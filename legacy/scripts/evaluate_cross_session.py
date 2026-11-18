from __future__ import annotations

"""
TriCSI cross-session evaluator.

Default experiment:
    TRAIN = S1
    TEST  = S2

Run:
    python evaluate_cross_session.py --train-session S1 --test-session S2
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
    out = []
    for label, xy in zip(y_true, xy_pred):
        if label not in ZONE_CENTERS:
            out.append(np.nan)
        else:
            true_xy = np.asarray(ZONE_CENTERS[label], dtype=np.float64)
            out.append(float(np.linalg.norm(xy - true_xy)))
    return np.asarray(out, dtype=np.float64)


def summarize_xy(errors: np.ndarray):
    x = errors[np.isfinite(errors)]
    if len(x) == 0:
        return {"mean_m": None, "median_m": None, "p90_m": None}
    return {
        "mean_m": float(np.mean(x)),
        "median_m": float(np.median(x)),
        "p90_m": float(np.percentile(x, 90)),
    }


def majority_vote_with_prob_tiebreak(preds, probs, classes):
    counts = Counter(preds.tolist())
    best_count = max(counts.values())
    tied = [k for k, v in counts.items() if v == best_count]
    if len(tied) == 1:
        return tied[0]
    mean_prob = np.mean(probs, axis=0)
    idx = {c: i for i, c in enumerate(classes)}
    return max(tied, key=lambda c: mean_prob[idx[c]])


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
            txt = f"{shown[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

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
        raise SystemExit(f"Missing training session directory: {train_dir}")
    if not test_dir.exists():
        raise SystemExit(f"Missing test session directory: {test_dir}")

    train_paths = gather_trial_paths(train_dir)
    test_paths = gather_trial_paths(test_dir)

    print("\n" + "=" * 78)
    print("TriCSI CROSS-SESSION EVALUATION")
    print("=" * 78)
    print(f"TRAIN session : {args.train_session} ({len(train_paths)} trials)")
    print(f"TEST session  : {args.test_session} ({len(test_paths)} trials)")
    print("No test-session samples are used for training.")
    print("=" * 78)

    print("\nExtracting TRAIN features...")
    train_rows = gather_windows(train_paths, args.window, args.hop)

    print("\nExtracting TEST features...")
    test_rows = gather_windows(test_paths, args.window, args.hop)

    if not train_rows or not test_rows:
        raise SystemExit("No usable windows were created.")

    feature_names = sorted(train_rows[0]["features"])
    if feature_names != sorted(test_rows[0]["features"]):
        raise SystemExit("Training and test feature sets do not match.")

    X_train = np.asarray(
        [[r["features"][f] for f in feature_names] for r in train_rows],
        dtype=np.float64,
    )
    y_train = np.asarray([r["zone"] for r in train_rows], dtype=object)

    X_test = np.asarray(
        [[r["features"][f] for f in feature_names] for r in test_rows],
        dtype=np.float64,
    )
    y_test = np.asarray([r["zone"] for r in test_rows], dtype=object)

    labels = [c for c in CLASS_ORDER if c in set(y_train) and c in set(y_test)]

    print("\nWINDOW COUNTS")
    print(f"  Train windows: {len(X_train)}")
    print(f"  Test windows : {len(X_test)}")

    print("\nTraining Random Forest on ALL training-session windows...")
    model = make_rf(args.seed)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)

    window_metrics = calc_metrics(y_test, pred)
    pred_xy = probability_weighted_xy(probs, model.classes_)
    window_xy_errors = xy_errors(y_test, pred_xy)
    window_xy_summary = summarize_xy(window_xy_errors)

    groups = np.asarray([r["trial_group"] for r in test_rows], dtype=object)
    trial_rows = []
    trial_true, trial_pred_majority, trial_pred_prob = [], [], []
    trial_xy_error_values = []

    for group in sorted(set(groups)):
        idxs = np.where(groups == group)[0]
        true_label = str(y_test[idxs[0]])
        group_preds = pred[idxs]
        group_probs = probs[idxs]
        mean_prob = np.mean(group_probs, axis=0)

        majority_pred = majority_vote_with_prob_tiebreak(
            group_preds, group_probs, model.classes_
        )
        mean_prob_pred = str(model.classes_[int(np.argmax(mean_prob))])

        group_xy = probability_weighted_xy(
            mean_prob.reshape(1, -1), model.classes_
        )[0]
        group_xy_err = xy_errors(
            np.asarray([true_label], dtype=object),
            group_xy.reshape(1, 2),
        )[0]

        first = test_rows[idxs[0]]
        row = {
            "trial_group": group,
            "source_file": first["source_file"],
            "zone_true": true_label,
            "trial": first["trial"],
            "orientation": first["orientation"],
            "n_windows": len(idxs),
            "pred_majority": majority_pred,
            "pred_mean_probability": mean_prob_pred,
            "correct_majority": int(majority_pred == true_label),
            "correct_mean_probability": int(mean_prob_pred == true_label),
            "x_pred_m": float(group_xy[0]),
            "y_pred_m": float(group_xy[1]),
            "xy_error_m": None if not np.isfinite(group_xy_err) else float(group_xy_err),
        }

        class_index = {c: i for i, c in enumerate(model.classes_)}
        for cls in labels:
            row[f"p_{cls}"] = float(mean_prob[class_index[cls]])

        trial_rows.append(row)
        trial_true.append(true_label)
        trial_pred_majority.append(majority_pred)
        trial_pred_prob.append(mean_prob_pred)
        if np.isfinite(group_xy_err):
            trial_xy_error_values.append(float(group_xy_err))

    trial_true = np.asarray(trial_true, dtype=object)
    trial_pred_majority = np.asarray(trial_pred_majority, dtype=object)
    trial_pred_prob = np.asarray(trial_pred_prob, dtype=object)

    trial_majority_metrics = calc_metrics(trial_true, trial_pred_majority)
    trial_prob_metrics = calc_metrics(trial_true, trial_pred_prob)
    trial_xy_summary = summarize_xy(
        np.asarray(trial_xy_error_values, dtype=np.float64)
    )

    out_dir = RESULTS_DIR / (
        f"cross_session_{args.train_session.lower()}_to_{args.test_session.lower()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_window = save_confusion(
        y_test,
        pred,
        labels,
        out_dir / "confusion_matrix_window.png",
        f"RF {args.train_session}->{args.test_session} confusion matrix (window)",
    )
    save_confusion(
        y_test,
        pred,
        labels,
        out_dir / "confusion_matrix_window_normalized.png",
        f"RF {args.train_session}->{args.test_session} normalized confusion (window)",
        normalize=True,
    )
    cm_trial = save_confusion(
        trial_true,
        trial_pred_prob,
        labels,
        out_dir / "confusion_matrix_trial_mean_probability.png",
        f"RF {args.train_session}->{args.test_session} confusion matrix (trial mean probability)",
    )

    zone_rows = []
    for i, label in enumerate(labels):
        ws, ts = int(cm_window[i].sum()), int(cm_trial[i].sum())
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

    test_orientations = np.asarray([r["orientation"] for r in test_rows], dtype=object)
    orientation_rows = []
    for orientation in sorted(set(test_orientations)):
        wm = test_orientations == orientation
        tr_idx = [i for i, r in enumerate(trial_rows) if r["orientation"] == orientation]
        orientation_rows.append({
            "orientation": orientation,
            "window_count": int(np.sum(wm)),
            "window_accuracy": float(accuracy_score(y_test[wm], pred[wm])) if np.any(wm) else None,
            "window_macro_f1": float(
                f1_score(y_test[wm], pred[wm], average="macro", zero_division=0)
            ) if np.any(wm) else None,
            "trial_count": len(tr_idx),
            "trial_accuracy_mean_probability": (
                float(accuracy_score(trial_true[tr_idx], trial_pred_prob[tr_idx]))
                if tr_idx else None
            ),
        })
    write_csv(out_dir / "orientation_metrics.csv", orientation_rows)

    window_rows = []
    class_index = {c: i for i, c in enumerate(model.classes_)}
    for i, row in enumerate(test_rows):
        out = {
            "source_file": row["source_file"],
            "trial_group": row["trial_group"],
            "zone_true": str(y_test[i]),
            "zone_pred": str(pred[i]),
            "correct": int(y_test[i] == pred[i]),
            "trial": row["trial"],
            "orientation": row["orientation"],
            "window_index": row["window_index"],
            "window_start_s": row["window_start_s"],
            "x_pred_m": float(pred_xy[i, 0]),
            "y_pred_m": float(pred_xy[i, 1]),
            "xy_error_m": None if not np.isfinite(window_xy_errors[i]) else float(window_xy_errors[i]),
        }
        for cls in labels:
            out[f"p_{cls}"] = float(probs[i, class_index[cls]])
        window_rows.append(out)

    write_csv(out_dir / "window_predictions.csv", window_rows)
    write_csv(out_dir / "trial_predictions.csv", trial_rows)

    (out_dir / "classification_report_window.txt").write_text(
        classification_report(y_test, pred, labels=labels, zero_division=0),
        encoding="utf-8",
    )
    (out_dir / "classification_report_trial.txt").write_text(
        classification_report(trial_true, trial_pred_prob, labels=labels, zero_division=0),
        encoding="utf-8",
    )

    summary = {
        "train_session": args.train_session,
        "test_session": args.test_session,
        "train_trials": len(train_paths),
        "test_trials": len(test_paths),
        "train_windows": len(X_train),
        "test_windows": len(X_test),
        "window_metrics": window_metrics,
        "window_xy": window_xy_summary,
        "trial_majority_vote_metrics": trial_majority_metrics,
        "trial_mean_probability_metrics": trial_prob_metrics,
        "trial_xy": trial_xy_summary,
        "note": "All model fitting uses only the training session. The test session is completely held out.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("CROSS-SESSION RESULTS")
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
        f"Trial mean probability: accuracy={trial_prob_metrics['accuracy']:.3f} "
        f"macroF1={trial_prob_metrics['macro_f1']:.3f}"
    )
    print(f"\nSaved results: {out_dir}")


if __name__ == "__main__":
    main()
