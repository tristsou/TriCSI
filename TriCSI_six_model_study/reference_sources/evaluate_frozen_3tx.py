from __future__ import annotations

"""Evaluate the predeclared frozen TriCSI models on untouched 3TX_S3 once."""

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
    combine_links,
    evaluate_classifier,
    extract_session_features,
    session_data_sha256,
    stratified_paired_bootstrap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("results/3tx_dev_comparison/frozen_3tx_model.joblib"))
    parser.add_argument("--session", type=Path, default=Path("data_3tx/raw/3TX_S3"))
    parser.add_argument("--output", type=Path, default=Path("results/3tx_s3_final"))
    return parser.parse_args()


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 7.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(np.arange(len(CLASS_ORDER)), labels=CLASS_ORDER, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(CLASS_ORDER)), labels=CLASS_ORDER)
    axis.set_xlabel("Predicted zone")
    axis.set_ylabel("True zone")
    axis.set_title("Frozen-model S3 trial confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} already contains results. Use a new output directory so the final test remains auditable."
        )
    output.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    if bundle.get("format_version") != 2:
        raise ValueError("Unsupported frozen-model format")
    config = bundle["feature_config"]
    test_set = extract_session_features(
        args.session,
        window_packets=int(config["window_packets"]),
        windows_per_trial=int(config["windows_per_trial"]),
    )
    if test_set.session_name in set(bundle["training_sessions"]):
        raise ValueError(
            f"Refusing to call {test_set.session_name} a final test because it was used for training."
        )
    test_hash = session_data_sha256(test_set.source_dir)
    if test_hash in set(bundle.get("training_data_sha256", {}).values()):
        raise ValueError(
            "Refusing final evaluation because these trial CSV bytes duplicate a training session."
        )
    if test_set.zone_coords != bundle["zone_coords"]:
        raise ValueError("S3 grid coordinates do not match the frozen development grid")

    metric_rows = []
    prediction_frames = []
    confusion_by_subset = {}
    for subset_name, model in bundle["models_by_subset"].items():
        links = bundle["subset_links"][subset_name]
        matrix = combine_links(test_set, links)
        metrics, predictions, confusion = evaluate_classifier(
            model, matrix, test_set.metadata, test_set.zone_coords
        )
        metric_rows.append(
            {
                "subset": subset_name,
                "links": "+".join(links),
                "link_count": len(links),
                **metrics,
            }
        )
        predictions.insert(0, "link_count", len(links))
        predictions.insert(0, "links", "+".join(links))
        predictions.insert(0, "subset", subset_name)
        prediction_frames.append(predictions)
        confusion_by_subset[subset_name] = confusion

    subset_metrics = pd.DataFrame(metric_rows).sort_values(
        ["trial_accuracy", "mean_position_error_m", "link_count"], ascending=[False, True, True]
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    primary_subset = bundle["primary_subset"]
    primary_metrics = subset_metrics.loc[subset_metrics["subset"] == primary_subset].iloc[0].to_dict()
    primary_confusion = confusion_by_subset[primary_subset]

    outer = predictions[predictions["subset"] == "2tx_outer"].copy()
    three = predictions[predictions["subset"] == "3tx_all"].copy()
    paired = outer.merge(
        three,
        on=["trial_uid", "true_zone"],
        suffixes=("_outer", "_3tx"),
        validate="one_to_one",
    )
    paired["accuracy_gain_3tx"] = paired["correct_3tx"] - paired["correct_outer"]
    paired["error_reduction_3tx_m"] = paired["position_error_m_outer"] - paired["position_error_m_3tx"]
    accuracy_gain = stratified_paired_bootstrap(paired, "accuracy_gain_3tx", iterations=5000, seed=2026)
    error_reduction = stratified_paired_bootstrap(
        paired[paired["true_zone"] != "EMPTY"],
        "error_reduction_3tx_m",
        iterations=5000,
        seed=2027,
    )
    report = {
        "test_session": test_set.session_name,
        "test_directory": test_set.source_dir,
        "model_path": str(model_path),
        "trained_on": bundle["training_sessions"],
        "primary_subset": primary_subset,
        "primary_links": bundle["subset_links"][primary_subset],
        "primary_metrics": primary_metrics,
        "predeclared_ablation_subsets": list(bundle["models_by_subset"]),
        "s3_middle_link_contrast": {
            "contrast": "3tx_all minus 2tx_outer",
            "accuracy_gain": accuracy_gain,
            "position_error_reduction_m": error_reduction,
        },
        "note": "No fitting, tuning, subset selection, or fusion-weight learning was performed on S3.",
    }
    subset_metrics.to_csv(output / "s3_subset_metrics.csv", index=False)
    predictions.to_csv(output / "s3_trial_predictions.csv", index=False)
    paired.to_csv(output / "s3_paired_outer_vs_3tx.csv", index=False)
    pd.DataFrame(primary_confusion, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        output / "s3_primary_confusion_matrix.csv"
    )
    with (output / "s3_final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    save_confusion_matrix(primary_confusion, output / "s3_primary_confusion_matrix.png")

    print("Frozen-model final evaluation complete. No model was retrained on S3.")
    print(subset_metrics[["subset", "trial_accuracy", "mean_position_error_m"]].to_string(index=False))
    print(f"\nPrimary subset: {primary_subset} ({', '.join(bundle['subset_links'][primary_subset])})")
    print(f"Primary trial accuracy: {100 * float(primary_metrics['trial_accuracy']):.1f}%")
    print(f"Primary occupied-zone accuracy: {100 * float(primary_metrics['occupied_trial_accuracy']):.1f}%")
    print(f"Primary EMPTY recall: {100 * float(primary_metrics['empty_recall']):.1f}%")
    print(f"Primary mean occupied-zone error: {float(primary_metrics['mean_position_error_m']):.3f} m")
    print(
        "S3 accuracy gain, 3TX - outer 2TX: "
        f"{100 * accuracy_gain['estimate']:+.1f} percentage points "
        f"(95% CI {100 * accuracy_gain['ci95_low']:+.1f} to {100 * accuracy_gain['ci95_high']:+.1f})"
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
