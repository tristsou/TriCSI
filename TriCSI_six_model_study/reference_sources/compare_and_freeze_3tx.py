from __future__ import annotations

"""
Compare every one-, two-, and three-transmitter subset using session holdout,
then freeze the predeclared model set trained on S1+S2 without looking at S3.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tricsi_3tx_modeling import (
    CLASS_ORDER,
    LINK_SUBSETS,
    combine_links,
    combine_sessions,
    evaluate_classifier,
    extract_session_features,
    make_classifier,
    session_data_sha256,
    stratified_paired_bootstrap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", type=Path, default=Path("data_3tx/raw/3TX_S1"))
    parser.add_argument("--s2", type=Path, default=Path("data_3tx/raw/3TX_S2"))
    parser.add_argument("--output", type=Path, default=Path("results/3tx_dev_comparison"))
    parser.add_argument("--window-packets", type=int, default=64)
    parser.add_argument("--windows-per-trial", type=int, default=7)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--freeze-subset",
        choices=("auto", *LINK_SUBSETS.keys()),
        default="auto",
        help="auto uses highest bidirectional trial accuracy, then lowest position error, then fewer links",
    )
    return parser.parse_args()


def save_comparison_plot(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values(["link_count", "subset"]).reset_index(drop=True)
    colors = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(ordered))
    bar_colors = [colors[int(value)] for value in ordered["link_count"]]
    axes[0].bar(x, 100 * ordered["mean_trial_accuracy"], color=bar_colors)
    axes[0].set_ylabel("Trial accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Bidirectional cross-session accuracy")
    axes[1].bar(x, ordered["mean_position_error_m"], color=bar_colors)
    axes[1].set_ylabel("Mean occupied-zone error (m)")
    axes[1].set_title("Lower is better")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(ordered["subset"], rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("Extracting fixed, equal-count windows from S1 and S2...")
    s1 = extract_session_features(
        args.s1, window_packets=args.window_packets, windows_per_trial=args.windows_per_trial
    )
    s2 = extract_session_features(
        args.s2, window_packets=args.window_packets, windows_per_trial=args.windows_per_trial
    )
    if s1.session_name == s2.session_name:
        raise ValueError("S1 and S2 resolved to the same session name")
    if s1.zone_coords != s2.zone_coords:
        raise ValueError("S1 and S2 zone coordinates differ; do not compare them as the same grid")

    fold_specs = [(s1, s2), (s2, s1)]
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for subset_name, links in LINK_SUBSETS.items():
        print(f"Testing {subset_name}: {', '.join(links)}")
        for fold_index, (train_set, test_set) in enumerate(fold_specs):
            model = make_classifier(seed=args.seed + fold_index, n_estimators=args.trees)
            train_matrix = combine_links(train_set, links)
            test_matrix = combine_links(test_set, links)
            model.fit(train_matrix, train_set.metadata["zone"])
            metrics, predictions, _ = evaluate_classifier(
                model, test_matrix, test_set.metadata, test_set.zone_coords
            )
            fold_name = f"{train_set.session_name}_to_{test_set.session_name}"
            metric_rows.append(
                {
                    "subset": subset_name,
                    "links": "+".join(links),
                    "link_count": len(links),
                    "fold": fold_name,
                    "train_session": train_set.session_name,
                    "test_session": test_set.session_name,
                    **metrics,
                }
            )
            predictions.insert(0, "fold", fold_name)
            predictions.insert(0, "link_count", len(links))
            predictions.insert(0, "links", "+".join(links))
            predictions.insert(0, "subset", subset_name)
            prediction_frames.append(predictions)

    fold_metrics = pd.DataFrame(metric_rows)
    trial_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = (
        fold_metrics.groupby(["subset", "links", "link_count"], as_index=False)
        .agg(
            mean_trial_accuracy=("trial_accuracy", "mean"),
            min_fold_trial_accuracy=("trial_accuracy", "min"),
            mean_occupied_trial_accuracy=("occupied_trial_accuracy", "mean"),
            mean_empty_recall=("empty_recall", "mean"),
            mean_window_accuracy=("window_accuracy", "mean"),
            mean_position_error_m=("mean_position_error_m", "mean"),
            max_fold_position_error_m=("mean_position_error_m", "max"),
        )
        .sort_values(["mean_trial_accuracy", "mean_position_error_m", "link_count"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    # Paired, stratified bootstrap: all-three versus the original outer pair.
    outer = trial_predictions[trial_predictions["subset"] == "2tx_outer"].copy()
    three = trial_predictions[trial_predictions["subset"] == "3tx_all"].copy()
    keys = ["fold", "trial_uid", "true_zone"]
    paired = outer.merge(three, on=keys, suffixes=("_outer", "_3tx"), validate="one_to_one")
    paired["accuracy_gain_3tx"] = paired["correct_3tx"] - paired["correct_outer"]
    paired["error_reduction_3tx_m"] = paired["position_error_m_outer"] - paired["position_error_m_3tx"]
    accuracy_gain = stratified_paired_bootstrap(
        paired, "accuracy_gain_3tx", iterations=5000, seed=args.seed
    )
    error_reduction = stratified_paired_bootstrap(
        paired[paired["true_zone"] != "EMPTY"],
        "error_reduction_3tx_m",
        iterations=5000,
        seed=args.seed + 1,
    )
    if accuracy_gain["ci95_low"] > 0:
        middle_conclusion = "The development sessions support an accuracy improvement from adding the middle link."
    elif accuracy_gain["ci95_high"] < 0:
        middle_conclusion = "The development sessions support an accuracy decrease from adding the middle link."
    else:
        middle_conclusion = "The accuracy effect of adding the middle link is inconclusive on S1/S2."

    if args.freeze_subset == "auto":
        winner = summary.sort_values(
            ["mean_trial_accuracy", "mean_position_error_m", "link_count"],
            ascending=[False, True, True],
        ).iloc[0]
        selected_subset = str(winner["subset"])
    else:
        selected_subset = args.freeze_subset
    selected_links = LINK_SUBSETS[selected_subset]

    print(f"Primary frozen subset: {selected_subset}: {', '.join(selected_links)}")
    frozen_models = {}
    for subset_index, (subset_name, links) in enumerate(LINK_SUBSETS.items()):
        final_matrix, final_metadata = combine_sessions([s1, s2], links)
        final_model = make_classifier(seed=args.seed + subset_index, n_estimators=args.trees)
        final_model.fit(final_matrix, final_metadata["zone"])
        frozen_models[subset_name] = final_model

    feature_config = {
        "window_packets": args.window_packets,
        "windows_per_trial": args.windows_per_trial,
    }
    bundle = {
        "format_version": 2,
        "models_by_subset": frozen_models,
        "subset_links": {name: list(links) for name, links in LINK_SUBSETS.items()},
        "primary_subset": selected_subset,
        "feature_config": feature_config,
        "classes": list(CLASS_ORDER),
        "zone_coords": s1.zone_coords,
        "training_sessions": [s1.session_name, s2.session_name],
        "training_directories": [s1.source_dir, s2.source_dir],
        "training_data_sha256": {
            s1.session_name: session_data_sha256(s1.source_dir),
            s2.session_name: session_data_sha256(s2.source_dir),
        },
        "seed": args.seed,
        "trees": args.trees,
        "selection_rule": (
            "Highest mean trial accuracy across S1->S2 and S2->S1; then lowest mean "
            "occupied-zone error; then fewer links. Override recorded if --freeze-subset was not auto."
        ),
        "selection_mode": args.freeze_subset,
    }
    model_path = output / "frozen_3tx_model.joblib"
    joblib.dump(bundle, model_path, compress=3)

    development_report = {
        "s1": s1.source_dir,
        "s2": s2.source_dir,
        "protocol": {
            "validation": "whole-session holdout in both directions",
            "window_packets": args.window_packets,
            "windows_per_trial": args.windows_per_trial,
            "trial_decision": "mean predicted probability across fixed windows",
            "classifier": "ExtraTreesClassifier",
            "trees": args.trees,
            "seed": args.seed,
        },
        "middle_link_test": {
            "contrast": "3tx_all minus 2tx_outer",
            "accuracy_gain": accuracy_gain,
            "position_error_reduction_m": error_reduction,
            "conclusion": middle_conclusion,
        },
        "primary_subset": selected_subset,
        "primary_links": list(selected_links),
        "selection_mode": args.freeze_subset,
        "frozen_model": str(model_path),
        "frozen_ablation_subsets": list(LINK_SUBSETS),
    }

    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "subset_summary.csv", index=False)
    trial_predictions.to_csv(output / "trial_predictions.csv", index=False)
    paired.to_csv(output / "paired_outer_vs_3tx.csv", index=False)
    with (output / "development_report.json").open("w", encoding="utf-8") as handle:
        json.dump(development_report, handle, indent=2)
    save_comparison_plot(summary, output / "subset_comparison.png")

    print("\nDevelopment comparison complete.")
    print(summary[["subset", "mean_trial_accuracy", "mean_position_error_m"]].to_string(index=False))
    print(f"\n{middle_conclusion}")
    print(
        "Accuracy gain, 3TX - outer 2TX: "
        f"{100 * accuracy_gain['estimate']:+.1f} percentage points "
        f"(95% CI {100 * accuracy_gain['ci95_low']:+.1f} to {100 * accuracy_gain['ci95_high']:+.1f})"
    )
    print(
        "Position-error reduction: "
        f"{error_reduction['estimate']:+.3f} m "
        f"(95% CI {error_reduction['ci95_low']:+.3f} to {error_reduction['ci95_high']:+.3f})"
    )
    print(f"Primary frozen subset: {selected_subset}")
    print(f"Frozen model set: {model_path}")
    print("Do not rerun model selection after opening S3 results.")


if __name__ == "__main__":
    main()
