from __future__ import annotations

"""
TriCSI N-only diagnostic.

Goal
----
Separate ORIENTATION effects from SESSION effects by using only the repeated
North-facing trials:

    trial 1 = N
    trial 5 = N

For occupied zones, only orientation == "N" is used.
For EMPTY, trial 1 / trial 5 are retained even though orientation is "NA".

The script runs BOTH:
1) Original engineered-feature Random Forest
2) Raw-CSI CNN V1

Comparisons
-----------
Same-session repeatability:
    S1 trial01_N -> S1 trial05_N
    S1 trial05_N -> S1 trial01_N
    S2 trial01_N -> S2 trial05_N
    S2 trial05_N -> S2 trial01_N

Cross-session with orientation fixed:
    S1 N1+N5 -> S2 N1+N5
    S2 N1+N5 -> S1 N1+N5

Important CNN detail
--------------------
For the one-trial-per-class same-session comparisons, a clean inner
whole-trial validation split is impossible: there is only ONE training N
recording per class. Therefore those four CNN runs use a FIXED epoch count
(default 25) and never look at the test trial to select epochs.

For the cross-session N1+N5 comparisons, there are two training N recordings
per class, so the existing clean inner whole-trial epoch-selection procedure
from train_csi_cnn.py is used.

This is a DIAGNOSTIC experiment, not the final model-selection benchmark.

Run
---
    python evaluate_n_only.py

Optional:
    python evaluate_n_only.py --pairwise-epochs 25
    python evaluate_n_only.py --skip-cnn
    python evaluate_n_only.py --skip-rf
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from project_config import (
    CLASS_ORDER,
    HOP_SECONDS,
    MIN_PACKETS_PER_LINK,
    RAW_DATA_DIR,
    RESULTS_DIR,
    WINDOW_SECONDS,
)

# Original engineered RF pipeline.
from train_crossval import (
    gather_windows as gather_engineered_windows,
    make_rf,
    trial_number_from_row,
)

# Existing raw-CNN V1 pipeline.
from train_csi_cnn import (
    TrainingConfig,
    aggregate_trials,
    classification_metrics,
    combine_raw_sets,
    encode_labels,
    fit_full_training_data,
    gather_raw_windows,
    index_to_text,
    label_to_index_map,
    predict_arrays,
    resolve_device,
    run_outer_experiment,
    set_seed,
    trial_paths_for_session,
)


N_TRIALS = {1, 5}


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def is_n_sample(zone: str, orientation: str, trial: int) -> bool:
    """
    Occupied zones: require N orientation and trial 1 or 5.
    EMPTY: keep trial 1 or 5 despite orientation == NA.
    """
    if int(trial) not in N_TRIALS:
        return False

    if str(zone) == "EMPTY":
        return True

    return str(orientation).strip().upper() == "N"


def standard_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_result_row(row: dict[str, Any]) -> None:
    print(
        f"{row['model']:4s} | "
        f"{row['comparison']:28s} | "
        f"window={row['window_accuracy']:.3f} "
        f"F1={row['window_macro_f1']:.3f} | "
        f"trial mean-prob={row['trial_mean_probability_accuracy']:.3f}"
    )


# ---------------------------------------------------------------------------
# RF
# ---------------------------------------------------------------------------

def load_rf_session(
    raw_dir: Path,
    session: str,
    *,
    window_seconds: float,
    hop_seconds: float,
) -> list[dict[str, Any]]:
    paths = trial_paths_for_session(raw_dir, session)

    print(f"\nLoading engineered RF features for {session}...")
    rows = gather_engineered_windows(
        paths,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )

    filtered = []

    for row in rows:
        t = trial_number_from_row(row)
        if is_n_sample(
            str(row["zone"]),
            str(row["orientation"]),
            t,
        ):
            filtered.append(row)

    validate_rf_n_rows(filtered, session)
    return filtered


def validate_rf_n_rows(rows: list[dict[str, Any]], session: str) -> None:
    for label in CLASS_ORDER:
        class_rows = [r for r in rows if str(r["zone"]) == label]
        trials = sorted(
            {
                trial_number_from_row(r)
                for r in class_rows
            }
        )

        if trials != [1, 5]:
            raise RuntimeError(
                f"{session} / {label}: expected N trials [1, 5], "
                f"found {trials}"
            )


def rf_feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]):
    X = np.asarray(
        [
            [float(row["features"][name]) for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.asarray([str(row["zone"]) for row in rows], dtype=object)
    return X, y


def rf_trial_mean_probability(
    test_rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    groups = np.asarray(
        [str(r["trial_group"]) for r in test_rows],
        dtype=object,
    )
    y = np.asarray([str(r["zone"]) for r in test_rows], dtype=object)

    trial_true = []
    trial_pred = []
    trial_rows = []

    for group in sorted(set(groups)):
        idx = np.where(groups == group)[0]
        true_label = str(y[idx[0]])
        mean_prob = probabilities[idx].mean(axis=0)
        pred = str(classes[int(np.argmax(mean_prob))])

        trial_true.append(true_label)
        trial_pred.append(pred)

        trial_rows.append(
            {
                "trial_group": group,
                "zone_true": true_label,
                "pred_mean_probability": pred,
                "correct": int(pred == true_label),
                "n_windows": int(len(idx)),
            }
        )

    return (
        float(accuracy_score(trial_true, trial_pred)),
        trial_rows,
    )


def run_rf_comparison(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    comparison: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_names = sorted(train_rows[0]["features"])

    if sorted(test_rows[0]["features"]) != feature_names:
        raise RuntimeError("RF train/test feature names differ.")

    X_train, y_train = rf_feature_matrix(train_rows, feature_names)
    X_test, y_test = rf_feature_matrix(test_rows, feature_names)

    if set(y_train) != set(CLASS_ORDER):
        raise RuntimeError(
            f"{comparison}: RF training split missing classes: "
            f"{sorted(set(CLASS_ORDER) - set(y_train))}"
        )

    if set(y_test) != set(CLASS_ORDER):
        raise RuntimeError(
            f"{comparison}: RF test split missing classes."
        )

    model = make_rf(seed)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    wm = standard_metrics(y_test, pred)
    trial_acc, trial_rows = rf_trial_mean_probability(
        test_rows,
        prob,
        model.classes_,
    )

    row = {
        "model": "RF",
        "comparison": comparison,
        "train_windows": int(len(train_rows)),
        "test_windows": int(len(test_rows)),
        "window_accuracy": wm["accuracy"],
        "window_macro_f1": wm["macro_f1"],
        "trial_mean_probability_accuracy": trial_acc,
        "epoch_rule": "not_applicable",
    }

    return row, trial_rows


def select_rf(
    rows: list[dict[str, Any]],
    *,
    session: str,
    trials: set[int],
) -> list[dict[str, Any]]:
    out = [
        r
        for r in rows
        if str(r.get("session", session)) == session
        and trial_number_from_row(r) in trials
    ]

    # Some feature-pipeline rows may not preserve session explicitly.
    if not out:
        out = [
            r
            for r in rows
            if trial_number_from_row(r) in trials
        ]

    return out


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

def load_cnn_session(
    raw_dir: Path,
    session: str,
    *,
    window_seconds: float,
    hop_seconds: float,
    target_steps: int,
    min_packets_per_link: int,
):
    paths = trial_paths_for_session(raw_dir, session)

    print(f"\nLoading raw CNN windows for {session}...")
    data = gather_raw_windows(
        paths,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        target_steps=target_steps,
        min_packets_per_link=min_packets_per_link,
    )

    mask = np.asarray(
        [
            is_n_sample(
                str(zone),
                str(orientation),
                int(trial),
            )
            for zone, orientation, trial
            in zip(
                data.y_text,
                data.orientation,
                data.trial_number,
            )
        ],
        dtype=bool,
    )

    filtered = subset_raw(data, mask)
    validate_cnn_n_data(filtered, session)
    return filtered


def subset_raw(data, mask):
    """
    Preserve the RawWindowSet type without duplicating its definition.
    """
    return type(data)(
        X=data.X[mask],
        rssi=data.rssi[mask],
        y_text=data.y_text[mask],
        session=data.session[mask],
        trial_number=data.trial_number[mask],
        orientation=data.orientation[mask],
        group=data.group[mask],
        source_file=data.source_file[mask],
        window_index=data.window_index[mask],
        window_start_s=data.window_start_s[mask],
        left_packets=data.left_packets[mask],
        right_packets=data.right_packets[mask],
    )


def validate_cnn_n_data(data, session: str) -> None:
    for label in CLASS_ORDER:
        trials = sorted(
            set(
                data.trial_number[
                    data.y_text == label
                ].tolist()
            )
        )
        if trials != [1, 5]:
            raise RuntimeError(
                f"{session} / {label}: expected N trials [1, 5], "
                f"found {trials}"
            )


def fixed_epoch_cnn_comparison(
    data,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    comparison: str,
    cfg: TrainingConfig,
    epochs: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Used only where there is ONE training N trial per class.

    There is no honest inner whole-trial validation split in that situation,
    so use a predeclared fixed epoch count and never inspect test performance
    to select it.
    """
    labels, label_map = label_to_index_map(data.y_text)
    y_index = encode_labels(data.y_text, label_map)

    train_groups = set(data.group[train_mask])
    test_groups = set(data.group[test_mask])

    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"{comparison}: CNN group leakage: {overlap}")

    print(
        f"\n[{comparison}] CNN fixed-epoch diagnostic: "
        f"{epochs} epochs"
    )

    model, normalizer, _ = fit_full_training_data(
        data.X[train_mask],
        data.rssi[train_mask],
        y_index[train_mask],
        n_classes=len(labels),
        epochs=epochs,
        cfg=cfg,
        seed=seed,
    )

    y_return, pred_index, probabilities = predict_arrays(
        model,
        normalizer,
        data.X[test_mask],
        data.rssi[test_mask],
        y_index[test_mask],
        cfg=cfg,
    )

    if not np.array_equal(y_return, y_index[test_mask]):
        raise RuntimeError("CNN test order changed.")

    y_true = data.y_text[test_mask]
    pred_text = index_to_text(pred_index, labels)

    wm = classification_metrics(y_true, pred_text)

    (
        trial_rows,
        trial_true,
        _trial_majority,
        trial_mean_prob,
    ) = aggregate_trials(
        y_true,
        pred_text,
        probabilities,
        data.group[test_mask],
        data.orientation[test_mask],
        data.source_file[test_mask],
        data.trial_number[test_mask],
        labels,
    )

    trial_metrics = classification_metrics(
        trial_true,
        trial_mean_prob,
    )

    row = {
        "model": "CNN",
        "comparison": comparison,
        "train_windows": int(np.sum(train_mask)),
        "test_windows": int(np.sum(test_mask)),
        "window_accuracy": float(wm["accuracy"]),
        "window_macro_f1": float(wm["macro_f1"]),
        "trial_mean_probability_accuracy": float(
            trial_metrics["accuracy"]
        ),
        "epoch_rule": f"fixed_{epochs}",
    }

    return row, trial_rows


def clean_cross_session_cnn_comparison(
    train_data,
    test_data,
    *,
    comparison: str,
    cfg: TrainingConfig,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Two N training recordings/class are available, so use the existing
    scientific inner whole-trial validation for epoch selection.
    """
    combined = combine_raw_sets(train_data, test_data)

    train_session = str(train_data.session[0])
    test_session = str(test_data.session[0])

    train_mask = combined.session == train_session
    test_mask = combined.session == test_session

    labels, label_map = label_to_index_map(combined.y_text)

    result = run_outer_experiment(
        combined,
        train_mask,
        test_mask,
        labels=labels,
        label_map=label_map,
        cfg=cfg,
        seed=seed,
        run_name=comparison,
        save_model_path=None,
    )

    wm = result["window_metrics"]
    tm = result["trial_mean_probability_metrics"]

    row = {
        "model": "CNN",
        "comparison": comparison,
        "train_windows": int(result["train_windows"]),
        "test_windows": int(result["test_windows"]),
        "window_accuracy": float(wm["accuracy"]),
        "window_macro_f1": float(wm["macro_f1"]),
        "trial_mean_probability_accuracy": float(tm["accuracy"]),
        "epoch_rule": f"inner_selected_{result['selected_epochs']}",
    }

    return row, result["trial_rows"]


# ---------------------------------------------------------------------------
# Run all comparisons
# ---------------------------------------------------------------------------

def rf_splits(s1: list[dict[str, Any]], s2: list[dict[str, Any]]):
    return [
        (
            "S1_N1_to_N5",
            [r for r in s1 if trial_number_from_row(r) == 1],
            [r for r in s1 if trial_number_from_row(r) == 5],
        ),
        (
            "S1_N5_to_N1",
            [r for r in s1 if trial_number_from_row(r) == 5],
            [r for r in s1 if trial_number_from_row(r) == 1],
        ),
        (
            "S2_N1_to_N5",
            [r for r in s2 if trial_number_from_row(r) == 1],
            [r for r in s2 if trial_number_from_row(r) == 5],
        ),
        (
            "S2_N5_to_N1",
            [r for r in s2 if trial_number_from_row(r) == 5],
            [r for r in s2 if trial_number_from_row(r) == 1],
        ),
        (
            "S1_N_to_S2_N",
            s1,
            s2,
        ),
        (
            "S2_N_to_S1_N",
            s2,
            s1,
        ),
    ]


def cnn_pairwise_specs(s1, s2):
    specs = []

    for session, data in (("S1", s1), ("S2", s2)):
        for train_trial, test_trial in ((1, 5), (5, 1)):
            specs.append(
                (
                    f"{session}_N{train_trial}_to_N{test_trial}",
                    data,
                    data.trial_number == train_trial,
                    data.trial_number == test_trial,
                )
            )

    return specs


def add_repeatability_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add convenience summary rows for the two directions within each session.
    """
    extras = []

    for model in ("RF", "CNN"):
        for session in ("S1", "S2"):
            wanted = {
                f"{session}_N1_to_N5",
                f"{session}_N5_to_N1",
            }

            subset = [
                r for r in rows
                if r["model"] == model
                and r["comparison"] in wanted
            ]

            if len(subset) != 2:
                continue

            extras.append(
                {
                    "model": model,
                    "comparison": f"{session}_N_repeatability_MEAN",
                    "train_windows": "",
                    "test_windows": "",
                    "window_accuracy": float(
                        np.mean([r["window_accuracy"] for r in subset])
                    ),
                    "window_macro_f1": float(
                        np.mean([r["window_macro_f1"] for r in subset])
                    ),
                    "trial_mean_probability_accuracy": float(
                        np.mean(
                            [
                                r["trial_mean_probability_accuracy"]
                                for r in subset
                            ]
                        )
                    ),
                    "epoch_rule": "mean_of_two_directions",
                }
            )

    return extras


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--window", type=float, default=WINDOW_SECONDS)
    p.add_argument("--hop", type=float, default=HOP_SECONDS)
    p.add_argument("--target-steps", type=int, default=32)
    p.add_argument(
        "--min-packets",
        type=int,
        default=MIN_PACKETS_PER_LINK,
    )

    p.add_argument(
        "--pairwise-epochs",
        type=int,
        default=25,
        help=(
            "Fixed CNN epochs for N1->N5 / N5->N1 diagnostics. "
            "Default 25. Do not tune this from the held-out pair results."
        ),
    )

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--augmentation-strength", type=float, default=1.0)
    p.add_argument("--no-rssi", action="store_true")
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--skip-rf", action="store_true")
    p.add_argument("--skip-cnn", action="store_true")

    args = p.parse_args()

    if args.skip_rf and args.skip_cnn:
        raise SystemExit("Nothing to run: both --skip-rf and --skip-cnn set.")

    output_dir = RESULTS_DIR / "n_only_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("TriCSI N-ONLY ORIENTATION-CONTROLLED DIAGNOSTIC")
    print("=" * 78)
    print("Occupied data used : trial01_N and trial05_N only")
    print("EMPTY data used    : trial01 and trial05")
    print("Comparisons        : same-session repeatability + cross-session")
    print(
        f"CNN pairwise epochs: {args.pairwise_epochs} "
        "(fixed; no test-based early stopping)"
    )
    print("=" * 78)

    start = time.time()
    result_rows: list[dict[str, Any]] = []
    detailed_trial_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # RF
    # ------------------------------------------------------------------
    if not args.skip_rf:
        rf_s1 = load_rf_session(
            args.raw_dir,
            "S1",
            window_seconds=args.window,
            hop_seconds=args.hop,
        )
        rf_s2 = load_rf_session(
            args.raw_dir,
            "S2",
            window_seconds=args.window,
            hop_seconds=args.hop,
        )

        print("\n" + "=" * 78)
        print("ORIGINAL ENGINEERED-FEATURE RF")
        print("=" * 78)

        for i, (name, train_rows, test_rows) in enumerate(
            rf_splits(rf_s1, rf_s2),
            1,
        ):
            row, trial_rows = run_rf_comparison(
                train_rows,
                test_rows,
                comparison=name,
                seed=args.seed + i,
            )
            result_rows.append(row)

            for tr in trial_rows:
                detailed_trial_rows.append(
                    {
                        "model": "RF",
                        "comparison": name,
                        **tr,
                    }
                )

            print_result_row(row)

    # ------------------------------------------------------------------
    # CNN
    # ------------------------------------------------------------------
    if not args.skip_cnn:
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

        cnn_s1 = load_cnn_session(
            args.raw_dir,
            "S1",
            window_seconds=args.window,
            hop_seconds=args.hop,
            target_steps=args.target_steps,
            min_packets_per_link=args.min_packets,
        )
        cnn_s2 = load_cnn_session(
            args.raw_dir,
            "S2",
            window_seconds=args.window,
            hop_seconds=args.hop,
            target_steps=args.target_steps,
            min_packets_per_link=args.min_packets,
        )

        print("\n" + "=" * 78)
        print("RAW-CSI CNN V1 — SAME-SESSION N REPEATABILITY")
        print("=" * 78)

        for i, (name, data, train_mask, test_mask) in enumerate(
            cnn_pairwise_specs(cnn_s1, cnn_s2),
            1,
        ):
            row, trial_rows = fixed_epoch_cnn_comparison(
                data,
                train_mask,
                test_mask,
                comparison=name,
                cfg=cfg,
                epochs=args.pairwise_epochs,
                seed=args.seed + 100 + i,
            )
            result_rows.append(row)

            for tr in trial_rows:
                detailed_trial_rows.append(
                    {
                        "model": "CNN",
                        "comparison": name,
                        **{
                            k: v
                            for k, v in tr.items()
                            if not k.startswith("p_")
                        },
                    }
                )

            print_result_row(row)

        print("\n" + "=" * 78)
        print("RAW-CSI CNN V1 — N-ONLY CROSS-SESSION")
        print("=" * 78)

        for i, (train_data, test_data, name) in enumerate(
            [
                (cnn_s1, cnn_s2, "S1_N_to_S2_N"),
                (cnn_s2, cnn_s1, "S2_N_to_S1_N"),
            ],
            1,
        ):
            row, trial_rows = clean_cross_session_cnn_comparison(
                train_data,
                test_data,
                comparison=name,
                cfg=cfg,
                seed=args.seed + 500 + i,
            )
            result_rows.append(row)

            for tr in trial_rows:
                detailed_trial_rows.append(
                    {
                        "model": "CNN",
                        "comparison": name,
                        **{
                            k: v
                            for k, v in tr.items()
                            if not k.startswith("p_")
                        },
                    }
                )

            print_result_row(row)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    result_rows.extend(add_repeatability_means(result_rows))

    order = {
        "S1_N1_to_N5": 1,
        "S1_N5_to_N1": 2,
        "S1_N_repeatability_MEAN": 3,
        "S2_N1_to_N5": 4,
        "S2_N5_to_N1": 5,
        "S2_N_repeatability_MEAN": 6,
        "S1_N_to_S2_N": 7,
        "S2_N_to_S1_N": 8,
    }

    result_rows.sort(
        key=lambda r: (
            0 if r["model"] == "RF" else 1,
            order.get(r["comparison"], 99),
        )
    )

    write_csv(
        output_dir / "n_only_results.csv",
        result_rows,
    )

    if detailed_trial_rows:
        # Different RF/CNN trial rows have slightly different keys, so build a
        # union of columns before writing.
        all_keys = []
        seen = set()
        for row in detailed_trial_rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    all_keys.append(key)

        with (output_dir / "trial_predictions.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=all_keys,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(detailed_trial_rows)

    summary = {
        "purpose": (
            "Control orientation by using only repeated North-facing trials "
            "1 and 5, then compare within-session repeatability against "
            "cross-session generalization."
        ),
        "cnn_pairwise_epoch_rule": (
            f"Fixed {args.pairwise_epochs} epochs for same-session pairwise "
            "tests because only one training N trial/class remains; "
            "cross-session N tests use inner whole-trial validation."
        ),
        "results": result_rows,
    }

    with (output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("N-ONLY DIAGNOSTIC SUMMARY")
    print("=" * 78)

    for row in result_rows:
        print_result_row(row)

    print(
        "\nInterpretation target:"
        "\n  same-session N repeatability high + cross-session N low"
        "\n      => real session shift even after orientation is controlled"
        "\n  same-session N repeatability low"
        "\n      => recording repeatability/setup stability is the bigger issue"
    )

    print(f"\nSaved: {output_dir}")
    print(
        f"Total elapsed: {(time.time() - start) / 60.0:.1f} min"
    )


if __name__ == "__main__":
    main()
