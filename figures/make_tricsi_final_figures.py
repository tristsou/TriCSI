from __future__ import annotations

"""
Create paper/STEM-competition figures from the frozen confirmatory results.

Run after S9 and S10 have been collected and evaluated.

Inputs:
    results/tricsi_anchor_confirmatory/
        S8_trial_predictions.csv
        S9_trial_predictions.csv
        S10_trial_predictions.csv
        pooled_confirmatory_summary.json

Outputs:
    results/tricsi_anchor_report_figures/
        session_accuracy.png
        pooled_confusion_matrix.png
        per_zone_recall.png
        mean_link_gate_by_zone.png
        report_metrics.csv

No model training occurs.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_config import CLASS_ORDER, RESULTS_DIR


LABELS = list(CLASS_ORDER)


def load_predictions(
    input_dir,
    sessions,
):
    frames = []

    for session in sessions:
        path = (
            input_dir
            / f"{session}_trial_predictions.csv"
        )

        if not path.exists():
            raise SystemExit(
                f"Missing predictions:\n{path}\n"
                "Run evaluate_frozen_anchor_confirmatory.py first."
            )

        df = pd.read_csv(
            path
        )
        frames.append(
            df
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def save_session_accuracy(
    df,
    output_dir,
    sessions,
):
    vals = []

    for session in sessions:
        s = df[
            df[
                "session"
            ]
            == session
        ]

        vals.append(
            float(
                np.mean(
                    s[
                        "true_zone"
                    ]
                    == s[
                        "pred_zone"
                    ]
                )
            )
        )

    fig, ax = plt.subplots(
        figsize=(6.5, 4.2)
    )

    ax.bar(
        sessions,
        vals,
    )

    ax.set_ylim(
        0,
        1.0,
    )
    ax.set_ylabel(
        "Exact trial accuracy"
    )
    ax.set_xlabel(
        "Held-out session"
    )
    ax.set_title(
        "Frozen cross-session localization accuracy"
    )

    for i, value in enumerate(
        vals
    ):
        ax.text(
            i,
            min(
                0.98,
                value + 0.025,
            ),
            f"{100*value:.1f}%",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(
        output_dir
        / "session_accuracy.png",
        dpi=220,
    )
    plt.close(
        fig
    )


def save_confusion(
    df,
    output_dir,
):
    matrix = np.zeros(
        (
            len(LABELS),
            len(LABELS),
        ),
        dtype=int,
    )

    lookup = {
        z: i
        for i, z in enumerate(
            LABELS
        )
    }

    for _, row in df.iterrows():
        truth = str(
            row[
                "true_zone"
            ]
        )
        pred = str(
            row[
                "pred_zone"
            ]
        )

        if (
            truth in lookup
            and pred in lookup
        ):
            matrix[
                lookup[
                    truth
                ],
                lookup[
                    pred
                ],
            ] += 1

    fig, ax = plt.subplots(
        figsize=(7.2, 6.4)
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(
                LABELS
            )
        )
    )
    ax.set_yticks(
        np.arange(
            len(
                LABELS
            )
        )
    )
    ax.set_xticklabels(
        LABELS,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(
        LABELS
    )
    ax.set_xlabel(
        "Predicted zone"
    )
    ax.set_ylabel(
        "True zone"
    )
    ax.set_title(
        "Pooled frozen-session confusion matrix"
    )

    for i in range(
        len(
            LABELS
        )
    ):
        for j in range(
            len(
                LABELS
            )
        ):
            if matrix[
                i,
                j,
            ]:
                ax.text(
                    j,
                    i,
                    str(
                        matrix[
                            i,
                            j,
                        ]
                    ),
                    ha="center",
                    va="center",
                )

    fig.colorbar(
        im,
        ax=ax,
        label="Number of trials",
    )

    fig.tight_layout()
    fig.savefig(
        output_dir
        / "pooled_confusion_matrix.png",
        dpi=220,
    )
    plt.close(
        fig
    )


def save_per_zone_recall(
    df,
    output_dir,
):
    recalls = []

    for zone in LABELS:
        z = df[
            df[
                "true_zone"
            ]
            == zone
        ]

        if len(
            z
        ):
            recalls.append(
                float(
                    np.mean(
                        z[
                            "pred_zone"
                        ]
                        == zone
                    )
                )
            )
        else:
            recalls.append(
                np.nan
            )

    fig, ax = plt.subplots(
        figsize=(8.0, 4.2)
    )

    ax.bar(
        LABELS,
        recalls,
    )

    ax.set_ylim(
        0,
        1.0,
    )

    ax.set_ylabel(
        "Recall"
    )

    ax.set_xlabel(
        "True zone"
    )

    ax.set_title(
        "Pooled per-zone recall"
    )

    fig.tight_layout()
    fig.savefig(
        output_dir
        / "per_zone_recall.png",
        dpi=220,
    )
    plt.close(
        fig
    )


def save_gate_plot(
    df,
    output_dir,
):
    occupied = df[
        df[
            "true_zone"
        ]
        != "EMPTY"
    ]

    left = []
    right = []

    zones = [
        z
        for z in LABELS
        if z != "EMPTY"
    ]

    for zone in zones:
        z = occupied[
            occupied[
                "true_zone"
            ]
            == zone
        ]

        left.append(
            float(
                z[
                    "left_gate"
                ].mean()
            )
        )

        right.append(
            float(
                z[
                    "right_gate"
                ].mean()
            )
        )

    x = np.arange(
        len(
            zones
        )
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(9.0, 4.4)
    )

    ax.bar(
        x - width / 2,
        left,
        width,
        label="LEFT link",
    )

    ax.bar(
        x + width / 2,
        right,
        width,
        label="RIGHT link",
    )

    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        zones
    )

    ax.set_ylim(
        0,
        1.0,
    )

    ax.set_ylabel(
        "Mean learned gate weight"
    )

    ax.set_xlabel(
        "True zone"
    )

    ax.set_title(
        "How the model uses the two CSI links"
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir
        / "mean_link_gate_by_zone.png",
        dpi=220,
    )
    plt.close(
        fig
    )


def save_metrics_table(
    df,
    output_dir,
    sessions,
):
    rows = []

    for session in sessions:
        s = df[
            df[
                "session"
            ]
            == session
        ]

        occupied = s[
            s[
                "true_zone"
            ]
            != "EMPTY"
        ]

        rows.append(
            {
                "session": session,
                "n_trials": len(
                    s
                ),
                "exact_accuracy": float(
                    np.mean(
                        s[
                            "true_zone"
                        ]
                        == s[
                            "pred_zone"
                        ]
                    )
                ),
                "occupied_exact_or_adjacent": float(
                    pd.to_numeric(
                        occupied[
                            "adjacent_or_exact"
                        ],
                        errors="coerce",
                    ).mean()
                ),
                "median_weighted_xy_error_m": float(
                    pd.to_numeric(
                        occupied[
                            "weighted_xy_error_m"
                        ],
                        errors="coerce",
                    ).median()
                ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        output_dir
        / "report_metrics.csv",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sessions",
        nargs="+",
        default=[
            "S8",
            "S9",
            "S10",
        ],
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=(
            RESULTS_DIR
            / "tricsi_anchor_confirmatory"
        ),
    )

    args = parser.parse_args()

    output_dir = (
        RESULTS_DIR
        / "tricsi_anchor_report_figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_predictions(
        args.input_dir,
        args.sessions,
    )

    save_session_accuracy(
        df,
        output_dir,
        args.sessions,
    )

    save_confusion(
        df,
        output_dir,
    )

    save_per_zone_recall(
        df,
        output_dir,
    )

    save_gate_plot(
        df,
        output_dir,
    )

    save_metrics_table(
        df,
        output_dir,
        args.sessions,
    )

    print(
        f"Saved report figures and metrics under:\n{output_dir}"
    )


if __name__ == "__main__":
    main()
