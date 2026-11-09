from __future__ import annotations

"""
Plain pooled S1+S2 Random Forest benchmark.

Combines S1 and S2, but splits by WHOLE independent trials, never by
overlapping 1-second windows.

For each (session, class), the 5 trials are randomly permuted and one is
assigned to each of 5 folds. Each fold therefore tests 20 trials:
2 sessions x 10 classes x 1 trial.

Run:
    python evaluate_pooled_rf.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from project_config import CLASS_ORDER, HOP_SECONDS, RAW_DATA_DIR, RESULTS_DIR, WINDOW_SECONDS
from train_crossval import gather_windows, make_rf, trial_number_from_row
from train_csi_cnn import trial_paths_for_session

SESSIONS = ("S1", "S2")


def make_assignment(seed: int):
    rng = np.random.default_rng(seed)
    out = {}
    for session in SESSIONS:
        for zone in CLASS_ORDER:
            trials = np.arange(1, 6)
            rng.shuffle(trials)
            for fold, trial in enumerate(trials, 1):
                out[(session, zone, int(trial))] = fold
    return out


def load_rows(raw_dir: Path, session: str, window: float, hop: float):
    paths = trial_paths_for_session(raw_dir, session)
    print(f"Loading RF {session}: {len(paths)} trials")
    rows = gather_windows(paths, window_seconds=window, hop_seconds=hop)
    for r in rows:
        r["_session"] = session
    return rows


def feature_matrix(rows, feature_names):
    X = np.asarray(
        [[float(r["features"][f]) for f in feature_names] for r in rows],
        dtype=np.float64,
    )
    y = np.asarray([str(r["zone"]) for r in rows], dtype=object)
    return X, y


def trial_mean_prob(test_rows, probs, classes):
    groups = np.asarray(
        [
            f"{r['_session']}|{r['zone']}|{trial_number_from_row(r)}"
            for r in test_rows
        ],
        dtype=object,
    )
    y = np.asarray([str(r["zone"]) for r in test_rows], dtype=object)

    yt, yp = [], []
    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        yt.append(str(y[idx[0]]))
        yp.append(str(classes[int(np.argmax(probs[idx].mean(axis=0)))]))
    return float(accuracy_score(yt, yp))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--window", type=float, default=WINDOW_SECONDS)
    p.add_argument("--hop", type=float, default=HOP_SECONDS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = RESULTS_DIR / "pooled_s1s2_plain_rf"
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment = make_assignment(args.seed)

    rows = []
    for session in SESSIONS:
        rows.extend(load_rows(args.raw_dir, session, args.window, args.hop))

    feature_names = sorted(rows[0]["features"])
    fold_rows = []

    print("\n" + "=" * 78)
    print("PLAIN POOLED S1+S2 RANDOM FOREST")
    print("=" * 78)
    print(f"Windows: {len(rows)}")
    print("Split: shuffled WHOLE trials, 5 folds")
    print("Each fold: 80 train trials / 20 test trials")
    print("=" * 78)

    for fold in range(1, 6):
        test_rows = [
            r for r in rows
            if assignment[
                (
                    str(r["_session"]),
                    str(r["zone"]),
                    int(trial_number_from_row(r)),
                )
            ] == fold
        ]
        train_rows = [r for r in rows if r not in test_rows]

        Xtr, ytr = feature_matrix(train_rows, feature_names)
        Xte, yte = feature_matrix(test_rows, feature_names)

        rf = make_rf(args.seed + fold)
        rf.fit(Xtr, ytr)

        pred = rf.predict(Xte)
        prob = rf.predict_proba(Xte)

        acc = float(accuracy_score(yte, pred))
        f1 = float(
            f1_score(
                yte,
                pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        )
        trial_acc = trial_mean_prob(test_rows, prob, rf.classes_)

        row = {
            "fold": fold,
            "window_accuracy": acc,
            "window_macro_f1": f1,
            "trial_mean_probability_accuracy": trial_acc,
            "train_windows": len(train_rows),
            "test_windows": len(test_rows),
        }
        fold_rows.append(row)

        print(
            f"Fold {fold}: window={acc:.3f} "
            f"F1={f1:.3f} trial mean-prob={trial_acc:.3f}"
        )

    accs = np.asarray([r["window_accuracy"] for r in fold_rows])
    f1s = np.asarray([r["window_macro_f1"] for r in fold_rows])
    trials = np.asarray([r["trial_mean_probability_accuracy"] for r in fold_rows])

    summary = {
        "window_accuracy_mean": float(accs.mean()),
        "window_accuracy_std": float(accs.std(ddof=1)),
        "window_macro_f1_mean": float(f1s.mean()),
        "window_macro_f1_std": float(f1s.std(ddof=1)),
        "trial_mean_probability_accuracy_mean": float(trials.mean()),
        "trial_mean_probability_accuracy_std": float(trials.std(ddof=1)),
        "folds": fold_rows,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        w.writerows(fold_rows)

    with (out_dir / "fold_assignment.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["fold", "session", "zone", "trial"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (session, zone, trial), fold in sorted(assignment.items(), key=lambda x: (x[1], x[0])):
            w.writerow({"fold": fold, "session": session, "zone": zone, "trial": trial})

    print("\n" + "=" * 78)
    print("POOLED S1+S2 RF SUMMARY")
    print("=" * 78)
    print(f"Window accuracy: {accs.mean():.3f} ± {accs.std(ddof=1):.3f}")
    print(f"Window macro F1: {f1s.mean():.3f} ± {f1s.std(ddof=1):.3f}")
    print(f"Trial mean probability: {trials.mean():.3f} ± {trials.std(ddof=1):.3f}")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
