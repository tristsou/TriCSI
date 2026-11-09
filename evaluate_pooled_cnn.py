from __future__ import annotations

"""
Plain pooled S1+S2 raw-CSI CNN benchmark.

Combines S1 and S2 and evaluates the original CNN V1 with NO domain loss,
NO alignment loss, and NO calibration.

The split is by WHOLE independent trials. For each (session, class), the
5 trials are randomly permuted and one is assigned to each of 5 folds.

Run:
    python evaluate_pooled_cnn.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from project_config import (
    CLASS_ORDER,
    HOP_SECONDS,
    MIN_PACKETS_PER_LINK,
    RAW_DATA_DIR,
    RESULTS_DIR,
    WINDOW_SECONDS,
)

from train_csi_cnn import (
    TrainingConfig,
    combine_raw_sets,
    gather_raw_windows,
    label_to_index_map,
    resolve_device,
    run_outer_experiment,
    set_seed,
    trial_paths_for_session,
)

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


def load_session(
    raw_dir,
    session,
    *,
    window,
    hop,
    target_steps,
    min_packets,
):
    paths = trial_paths_for_session(raw_dir, session)
    print(f"Loading CNN {session}: {len(paths)} trials")
    return gather_raw_windows(
        paths,
        window_seconds=window,
        hop_seconds=hop,
        target_steps=target_steps,
        min_packets_per_link=min_packets,
    )


def fold_mask(data, assignment, fold):
    return np.asarray(
        [
            assignment[(str(s), str(z), int(t))] == fold
            for s, z, t in zip(
                data.session,
                data.y_text,
                data.trial_number,
            )
        ],
        dtype=bool,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--window", type=float, default=WINDOW_SECONDS)
    p.add_argument("--hop", type=float, default=HOP_SECONDS)
    p.add_argument("--target-steps", type=int, default=32)
    p.add_argument("--min-packets", type=int, default=MIN_PACKETS_PER_LINK)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--augmentation-strength", type=float, default=1.0)
    p.add_argument("--no-rssi", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = RESULTS_DIR / "pooled_s1s2_plain_cnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment = make_assignment(args.seed)

    s1 = load_session(
        args.raw_dir,
        "S1",
        window=args.window,
        hop=args.hop,
        target_steps=args.target_steps,
        min_packets=args.min_packets,
    )
    s2 = load_session(
        args.raw_dir,
        "S2",
        window=args.window,
        hop=args.hop,
        target_steps=args.target_steps,
        min_packets=args.min_packets,
    )
    data = combine_raw_sets(s1, s2)

    device = resolve_device(args.device)
    cfg = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        augmentation_strength=args.augmentation_strength,
        use_rssi=not args.no_rssi,
        device=device,
        seed=args.seed,
    )

    set_seed(args.seed)
    labels, label_map = label_to_index_map(data.y_text)

    print("\n" + "=" * 78)
    print("PLAIN POOLED S1+S2 RAW-CSI CNN")
    print("=" * 78)
    print(f"Windows: {len(data.y_text)}")
    print(f"Trials : {len(set(data.group))}")
    print("Split  : shuffled WHOLE trials, 5 folds")
    print("Each fold: 80 train trials / 20 test trials")
    print("Domain/adaptation losses: NONE")
    print("=" * 78)

    fold_rows = []

    for fold in range(1, 6):
        test_mask = fold_mask(data, assignment, fold)
        train_mask = ~test_mask

        overlap = set(data.group[train_mask]) & set(data.group[test_mask])
        if overlap:
            raise RuntimeError(f"Fold {fold}: trial leakage detected: {overlap}")

        print("\n" + "=" * 78)
        print(f"POOLED CNN FOLD {fold}/5")
        print("=" * 78)

        result = run_outer_experiment(
            data,
            train_mask,
            test_mask,
            labels=labels,
            label_map=label_map,
            cfg=cfg,
            seed=args.seed + fold,
            run_name=f"pooled_s1s2_fold_{fold}",
            save_model_path=None,
        )

        wm = result["window_metrics"]
        tm = result["trial_mean_probability_metrics"]

        row = {
            "fold": fold,
            "window_accuracy": float(wm["accuracy"]),
            "window_macro_f1": float(wm["macro_f1"]),
            "trial_mean_probability_accuracy": float(tm["accuracy"]),
            "selected_epochs": int(result["selected_epochs"]),
            "train_windows": int(result["train_windows"]),
            "test_windows": int(result["test_windows"]),
        }
        fold_rows.append(row)

        print(
            f"Fold {fold}: window={row['window_accuracy']:.3f} "
            f"F1={row['window_macro_f1']:.3f} "
            f"trial mean-prob={row['trial_mean_probability_accuracy']:.3f}"
        )

    acc = np.asarray([r["window_accuracy"] for r in fold_rows])
    f1 = np.asarray([r["window_macro_f1"] for r in fold_rows])
    trial = np.asarray([r["trial_mean_probability_accuracy"] for r in fold_rows])

    summary = {
        "window_accuracy_mean": float(acc.mean()),
        "window_accuracy_std": float(acc.std(ddof=1)),
        "window_macro_f1_mean": float(f1.mean()),
        "window_macro_f1_std": float(f1.std(ddof=1)),
        "trial_mean_probability_accuracy_mean": float(trial.mean()),
        "trial_mean_probability_accuracy_std": float(trial.std(ddof=1)),
        "folds": fold_rows,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        w.writerows(fold_rows)

    print("\n" + "=" * 78)
    print("POOLED S1+S2 CNN SUMMARY")
    print("=" * 78)
    print(f"Window accuracy: {acc.mean():.3f} ± {acc.std(ddof=1):.3f}")
    print(f"Window macro F1: {f1.mean():.3f} ± {f1.std(ddof=1):.3f}")
    print(f"Trial mean probability: {trial.mean():.3f} ± {trial.std(ddof=1):.3f}")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
