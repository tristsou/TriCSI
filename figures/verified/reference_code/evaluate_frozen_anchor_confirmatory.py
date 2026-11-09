from __future__ import annotations

"""
Evaluate the EXACT SAVED TriCSI-Anchor model on S8/S9/S10 without retraining.

This is the confirmatory evaluator.

Frozen source model:
    models/tricsi_anchor_s4_s5_s6_s7.pt

Source prototypes:
    recomputed from S4-S7 with the frozen encoder.

Target-session information used by the model:
    EMPTY trial01 only.

No labeled occupied target-session sample is used to train, tune, calibrate,
or construct prototypes.

Supported test layouts
----------------------
S8:
    occupied trials03-06 are the original untouched final test
    EMPTY trials02-05 are test

S9 / S10 confirmatory collector:
    occupied trials01-04 are all test
    EMPTY trial01 is background calibration
    EMPTY trials02-05 are test
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from project_config import (
    CLASS_ORDER,
    MODEL_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    ZONE_CENTERS,
)

import train_tricsi_anchor_final as anchor


LABELS = list(CLASS_ORDER)
OCCUPIED = [z for z in LABELS if z != "EMPTY"]


def torch_load_full(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def load_frozen_model(checkpoint_path):
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Frozen checkpoint not found:\n{checkpoint_path}\n"
            "Do NOT retrain it. Copy the exact S8-era checkpoint into models/."
        )

    ckpt = torch_load_full(
        checkpoint_path
    )

    model = anchor.TriCSIAdaptNet(
        n_classes=len(LABELS),
        embed_dim=64,
    )
    model.load_state_dict(
        ckpt["state_dict"]
    )
    model.eval()

    n = ckpt["norm"]

    norm = anchor.Norm(
        csi_mean=np.asarray(
            n["csi_mean"],
            dtype=np.float32,
        ),
        csi_std=np.asarray(
            n["csi_std"],
            dtype=np.float32,
        ),
        rssi_mean=np.asarray(
            n["rssi_mean"],
            dtype=np.float32,
        ),
        rssi_std=np.asarray(
            n["rssi_std"],
            dtype=np.float32,
        ),
    )

    return model, norm, ckpt


def build_source_prototypes(
    model,
    norm,
    raw_dir,
):
    prepared = []

    for session in ("S4", "S5", "S6", "S7"):
        raw = anchor.load_session(
            raw_dir,
            session,
        )
        prepared.append(
            anchor.prepare_source_session(
                raw
            )
        )

    source = anchor.combine(
        prepared
    )

    embedded = anchor.embed_raw(
        model,
        source,
        norm,
    )

    occupied_mask = (
        source.y_text != "EMPTY"
    )

    prototypes = anchor.make_prototypes(
        embedded["embedding"][
            occupied_mask
        ],
        source.y_text[
            occupied_mask
        ],
    )

    return prototypes


def trial_group(raw, i):
    return (
        f"{raw.session[i]}|"
        f"{raw.y_text[i]}|"
        f"{int(raw.trial_number[i])}|"
        f"{raw.orientation[i]}"
    )


def test_mask_for_session(
    raw,
    session,
):
    is_empty = (
        raw.y_text == "EMPTY"
    )
    is_occ = ~is_empty

    if session == "S8":
        # Original frozen S8 final-test split.
        return (
            (
                is_occ
                & (
                    raw.trial_number >= 3
                )
            )
            |
            (
                is_empty
                & (
                    raw.trial_number >= 2
                )
            )
        )

    # S9/S10 confirmatory design:
    # ALL occupied trials are test; EMPTY trial01 alone is calibration.
    return (
        is_occ
        |
        (
            is_empty
            & (
                raw.trial_number >= 2
            )
        )
    )


def physical_error(
    true_zone,
    probs,
):
    if true_zone == "EMPTY":
        return None

    centers = np.asarray(
        [
            ZONE_CENTERS[z]
            for z in OCCUPIED
        ],
        dtype=np.float64,
    )

    xy = (
        np.asarray(
            probs,
            dtype=np.float64,
        )
        @ centers
    )

    truth = np.asarray(
        ZONE_CENTERS[
            true_zone
        ],
        dtype=np.float64,
    )

    return float(
        np.linalg.norm(
            xy - truth
        )
    )


def wilson_interval(
    successes,
    n,
    z=1.959963984540054,
):
    if n <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    p = successes / n
    denom = 1.0 + z * z / n

    center = (
        p
        + z * z / (2.0 * n)
    ) / denom

    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / n
            + z * z / (4.0 * n * n)
        )
        / denom
    )

    return (
        max(0.0, center - half),
        min(1.0, center + half),
    )


def evaluate_one(
    model,
    norm,
    source_prototypes,
    raw_dir,
    session,
):
    raw = anchor.load_session(
        raw_dir,
        session,
    )

    target = anchor.prepare_target_session(
        raw
    )

    embedded = anchor.embed_raw(
        model,
        target,
        norm,
    )

    mask = test_mask_for_session(
        target,
        session,
    )

    test_idx = np.where(
        mask
    )[0]

    if len(test_idx) == 0:
        raise RuntimeError(
            f"No evaluation windows found for {session}."
        )

    zone_probs = anchor.prototype_probabilities(
        embedded["embedding"][
            test_idx
        ],
        source_prototypes,
    )

    occ_prob = embedded[
        "occ_prob"
    ][test_idx]

    gates = embedded[
        "link_weights"
    ][test_idx]

    groups = np.asarray(
        [
            trial_group(
                target,
                int(i),
            )
            for i in test_idx
        ],
        dtype=object,
    )

    trial_rows = []
    errors = []
    adjacent_hits = []

    for group in sorted(set(groups)):
        local = np.where(
            groups == group
        )[0]

        global_i = int(
            test_idx[
                local[0]
            ]
        )

        true_zone = str(
            target.y_text[
                global_i
            ]
        )

        mean_occ = float(
            np.mean(
                occ_prob[
                    local
                ]
            )
        )

        mean_zone = np.mean(
            zone_probs[
                local
            ],
            axis=0,
        )
        mean_zone = (
            mean_zone
            / np.maximum(
                mean_zone.sum(),
                1e-12,
            )
        )

        if mean_occ < 0.5:
            pred = "EMPTY"
        else:
            pred = OCCUPIED[
                int(
                    np.argmax(
                        mean_zone
                    )
                )
            ]

        err = physical_error(
            true_zone,
            mean_zone,
        )

        if err is not None:
            errors.append(
                err
            )

            adjacent_hits.append(
                int(
                    pred != "EMPTY"
                    and anchor.adjacent(
                        true_zone,
                        pred,
                    )
                )
            )

        gate = np.mean(
            gates[
                local
            ],
            axis=0,
        )

        row = {
            "session": session,
            "group": group,
            "true_zone": true_zone,
            "pred_zone": pred,
            "correct": int(
                pred == true_zone
            ),
            "adjacent_or_exact": (
                ""
                if true_zone == "EMPTY"
                else int(
                    pred != "EMPTY"
                    and anchor.adjacent(
                        true_zone,
                        pred,
                    )
                )
            ),
            "mean_occ_prob": mean_occ,
            "left_gate": float(
                gate[0]
            ),
            "right_gate": float(
                gate[1]
            ),
            "weighted_xy_error_m": (
                ""
                if err is None
                else err
            ),
        }

        for j, zone in enumerate(
            OCCUPIED
        ):
            row[
                f"p_{zone}"
            ] = float(
                mean_zone[j]
            )

        trial_rows.append(
            row
        )

    y_true = np.asarray(
        [
            r["true_zone"]
            for r in trial_rows
        ],
        dtype=object,
    )

    y_pred = np.asarray(
        [
            r["pred_zone"]
            for r in trial_rows
        ],
        dtype=object,
    )

    correct = int(
        np.sum(
            y_true == y_pred
        )
    )

    total = len(
        trial_rows
    )

    low, high = wilson_interval(
        correct,
        total,
    )

    errs = np.asarray(
        errors,
        dtype=np.float64,
    )

    metrics = {
        "session": session,
        "n_trials": total,
        "correct_trials": correct,
        "exact_accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "exact_accuracy_wilson95_low": low,
        "exact_accuracy_wilson95_high": high,
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "occupied_exact_or_adjacent": (
            float(
                np.mean(
                    adjacent_hits
                )
            )
            if adjacent_hits
            else float("nan")
        ),
        "median_weighted_xy_error_m": (
            float(
                np.median(
                    errs
                )
            )
            if len(errs)
            else float("nan")
        ),
        "p90_weighted_xy_error_m": (
            float(
                np.percentile(
                    errs,
                    90,
                )
            )
            if len(errs)
            else float("nan")
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        ).tolist(),
        "per_zone_recall": {},
    }

    for zone in LABELS:
        zone_mask = (
            y_true == zone
        )
        if np.any(
            zone_mask
        ):
            metrics[
                "per_zone_recall"
            ][zone] = float(
                np.mean(
                    y_pred[
                        zone_mask
                    ]
                    == zone
                )
            )

    return metrics, trial_rows


def print_metrics(metrics):
    print("\n" + "=" * 78)
    print(
        f"FROZEN CONFIRMATORY EVALUATION — {metrics['session']}"
    )
    print("=" * 78)

    print(
        f"exact trial accuracy : "
        f"{metrics['exact_accuracy']:.3f} "
        f"({metrics['correct_trials']}/{metrics['n_trials']})"
    )

    print(
        "95% Wilson CI        : "
        f"[{metrics['exact_accuracy_wilson95_low']:.3f}, "
        f"{metrics['exact_accuracy_wilson95_high']:.3f}]"
    )

    print(
        f"macro F1             : "
        f"{metrics['macro_f1']:.3f}"
    )

    print(
        f"exact-or-adjacent    : "
        f"{metrics['occupied_exact_or_adjacent']:.3f}"
    )

    print(
        f"median weighted XY   : "
        f"{metrics['median_weighted_xy_error_m']:.3f} m"
    )

    print(
        f"P90 weighted XY      : "
        f"{metrics['p90_weighted_xy_error_m']:.3f} m"
    )

    print("\nper-zone recall")

    for zone in LABELS:
        if zone in metrics[
            "per_zone_recall"
        ]:
            print(
                f"  {zone:5s} "
                f"{metrics['per_zone_recall'][zone]:.3f}"
            )


def pooled_summary(
    all_metrics,
    all_rows,
):
    y_true = np.asarray(
        [
            r["true_zone"]
            for r in all_rows
        ],
        dtype=object,
    )

    y_pred = np.asarray(
        [
            r["pred_zone"]
            for r in all_rows
        ],
        dtype=object,
    )

    correct = int(
        np.sum(
            y_true == y_pred
        )
    )

    total = len(
        all_rows
    )

    low, high = wilson_interval(
        correct,
        total,
    )

    occupied_rows = [
        r
        for r in all_rows
        if r["true_zone"] != "EMPTY"
    ]

    adjacent_vals = [
        int(
            r[
                "adjacent_or_exact"
            ]
        )
        for r in occupied_rows
        if str(
            r[
                "adjacent_or_exact"
            ]
        )
        != ""
    ]

    errors = [
        float(
            r[
                "weighted_xy_error_m"
            ]
        )
        for r in occupied_rows
        if str(
            r[
                "weighted_xy_error_m"
            ]
        )
        != ""
    ]

    session_acc = [
        float(
            m[
                "exact_accuracy"
            ]
        )
        for m in all_metrics
    ]

    summary = {
        "sessions": [
            m["session"]
            for m in all_metrics
        ],
        "n_sessions": len(
            all_metrics
        ),
        "n_trials": total,
        "correct_trials": correct,
        "pooled_exact_accuracy": (
            correct / total
            if total
            else float("nan")
        ),
        "pooled_exact_accuracy_wilson95_low": low,
        "pooled_exact_accuracy_wilson95_high": high,
        "mean_session_accuracy": float(
            np.mean(
                session_acc
            )
        ),
        "std_session_accuracy": float(
            np.std(
                session_acc,
                ddof=1,
            )
        )
        if len(session_acc) > 1
        else 0.0,
        "pooled_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "pooled_occupied_exact_or_adjacent": float(
            np.mean(
                adjacent_vals
            )
        )
        if adjacent_vals
        else float("nan"),
        "pooled_median_weighted_xy_error_m": float(
            np.median(
                errors
            )
        )
        if errors
        else float("nan"),
        "pooled_p90_weighted_xy_error_m": float(
            np.percentile(
                errors,
                90,
            )
        )
        if errors
        else float("nan"),
        "per_session": {
            m["session"]: {
                "exact_accuracy": m[
                    "exact_accuracy"
                ],
                "macro_f1": m[
                    "macro_f1"
                ],
                "occupied_exact_or_adjacent": m[
                    "occupied_exact_or_adjacent"
                ],
            }
            for m in all_metrics
        },
    }

    print("\n" + "=" * 78)
    print("POOLED FROZEN CONFIRMATORY SUMMARY")
    print("=" * 78)

    print(
        "sessions              : "
        + ", ".join(
            summary["sessions"]
        )
    )

    print(
        f"pooled exact          : "
        f"{summary['pooled_exact_accuracy']:.3f} "
        f"({correct}/{total})"
    )

    print(
        "95% Wilson CI         : "
        f"[{low:.3f}, {high:.3f}]"
    )

    print(
        f"mean session accuracy : "
        f"{summary['mean_session_accuracy']:.3f}"
    )

    if len(
        all_metrics
    ) > 1:
        print(
            f"session SD            : "
            f"{summary['std_session_accuracy']:.3f}"
        )

    print(
        f"pooled macro F1       : "
        f"{summary['pooled_macro_f1']:.3f}"
    )

    print(
        f"pooled adjacent       : "
        f"{summary['pooled_occupied_exact_or_adjacent']:.3f}"
    )

    print(
        f"pooled median XY      : "
        f"{summary['pooled_median_weighted_xy_error_m']:.3f} m"
    )

    print(
        f"pooled P90 XY         : "
        f"{summary['pooled_p90_weighted_xy_error_m']:.3f} m"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sessions",
        nargs="+",
        default=["S8", "S9", "S10"],
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            MODEL_DIR
            / "tricsi_anchor_s4_s5_s6_s7.pt"
        ),
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
    )

    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("TriCSI-Anchor FROZEN CONFIRMATORY EVALUATOR")
    print("=" * 78)
    print(f"Checkpoint: {args.checkpoint}")
    print(
        "NO neural training and NO target-zone calibration will occur."
    )
    print("=" * 78)

    model, norm, ckpt = load_frozen_model(
        args.checkpoint
    )

    print(
        "Loaded frozen method: "
        f"{ckpt.get('method', '<unknown>')}"
    )

    source_prototypes = build_source_prototypes(
        model,
        norm,
        args.raw_dir,
    )

    out_dir = (
        RESULTS_DIR
        / "tricsi_anchor_confirmatory"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics = []
    all_rows = []

    for session in args.sessions:
        metrics, rows = evaluate_one(
            model,
            norm,
            source_prototypes,
            args.raw_dir,
            session,
        )

        print_metrics(
            metrics
        )

        all_metrics.append(
            metrics
        )
        all_rows.extend(
            rows
        )

        with (
            out_dir
            / f"{session}_metrics.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metrics,
                f,
                indent=2,
            )

        with (
            out_dir
            / f"{session}_trial_predictions.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                rows
            )

    summary = pooled_summary(
        all_metrics,
        all_rows,
    )

    with (
        out_dir
        / "pooled_confirmatory_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    with (
        out_dir
        / "all_confirmatory_trial_predictions.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                all_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            all_rows
        )

    print(
        f"\nSaved confirmatory results under:\n{out_dir}"
    )


if __name__ == "__main__":
    main()
