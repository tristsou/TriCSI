"""
Analyze ESP32-CSI-Tool CSV files and generate RSSI/CSI graphs.

Examples:
    python analyze_csi.py cover_right_01.csv
    python analyze_csi.py variation_02.csv
    python analyze_csi.py foil_01.csv empty_01.csv empty_02.csv

The script accepts one or more filtered CSV files containing lines beginning with:
CSI_DATA,AP,<STA MAC>,...
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MAC = "02:00:00:00:00:03"


def parse_csi_file(path: Path, mac: str) -> pd.DataFrame:
    """Parse one filtered ESP32 CSI CSV file."""
    rows: list[dict[str, float]] = []
    expected_prefix = f"CSI_DATA,AP,{mac}"

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line.startswith(expected_prefix):
                continue

            try:
                prefix, vector_text = line.rsplit(",[", 1)
                fields = prefix.split(",")

                # CSI vector contains signed I/Q integer pairs.
                values = np.asarray(
                    [int(value) for value in re.findall(r"-?\d+", "[" + vector_text)],
                    dtype=float,
                )

                if values.size < 2:
                    continue

                if values.size % 2:
                    values = values[:-1]

                i_values = values[0::2]
                q_values = values[1::2]
                amplitudes = np.sqrt(i_values**2 + q_values**2)

                rows.append(
                    {
                        "rssi": float(fields[3]),
                        "timestamp": float(fields[23]),
                        "mean_csi_amplitude": float(np.mean(amplitudes)),
                        "median_csi_amplitude": float(np.median(amplitudes)),
                        "within_frame_csi_sd": float(np.std(amplitudes)),
                    }
                )
            except (IndexError, TypeError, ValueError) as exc:
                print(f"Skipped malformed row {line_number} in {path.name}: {exc}")

    if not rows:
        raise ValueError(
            f"No valid CSI rows found in {path}. "
            f"Check the MAC address and filtered-file format."
        )

    frame = pd.DataFrame(rows)
    frame["frame"] = np.arange(1, len(frame) + 1)
    frame["time_s"] = frame["timestamp"] - frame["timestamp"].iloc[0]

    frame["rssi_abs_change"] = frame["rssi"].diff().abs()
    frame["csi_abs_change"] = frame["mean_csi_amplitude"].diff().abs()

    rolling_window = max(5, min(15, len(frame) // 10))
    frame["rolling_rssi_mean"] = (
        frame["rssi"].rolling(rolling_window, center=True, min_periods=1).mean()
    )
    frame["rolling_rssi_sd"] = (
        frame["rssi"].rolling(rolling_window, center=True, min_periods=2).std()
    )
    frame["rolling_csi_mean"] = (
        frame["mean_csi_amplitude"]
        .rolling(rolling_window, center=True, min_periods=1)
        .mean()
    )
    frame["rolling_csi_sd"] = (
        frame["mean_csi_amplitude"]
        .rolling(rolling_window, center=True, min_periods=2)
        .std()
    )

    return frame


def summarize_trial(path: Path, frame: pd.DataFrame) -> dict[str, float | str]:
    """Compute one independent summary row per trial."""
    return {
        "file": path.name,
        "frames": int(len(frame)),
        "duration_s": float(frame["time_s"].iloc[-1]),
        "mean_rssi_dbm": float(frame["rssi"].mean()),
        "rssi_sd_db": float(frame["rssi"].std()),
        "largest_rssi_frame_change_db": float(frame["rssi_abs_change"].max()),
        "mean_csi_amplitude": float(frame["mean_csi_amplitude"].mean()),
        "csi_amplitude_sd_over_time": float(frame["mean_csi_amplitude"].std()),
        "mean_within_frame_csi_sd": float(frame["within_frame_csi_sd"].mean()),
        "mean_abs_csi_frame_change": float(frame["csi_abs_change"].mean()),
        "largest_csi_frame_change": float(frame["csi_abs_change"].max()),
    }


def save_individual_plot(path: Path, frame: pd.DataFrame, output_dir: Path) -> Path:
    """Graph RSSI, CSI amplitude, and event-style changes for one trial."""
    output_path = output_dir / f"{path.stem}_analysis.png"

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(frame["time_s"], frame["rssi"], alpha=0.55, label="Raw RSSI")
    axes[0].plot(
        frame["time_s"],
        frame["rolling_rssi_mean"],
        linewidth=2,
        label="Rolling mean",
    )
    axes[0].set_ylabel("RSSI (dBm)")
    axes[0].set_title(f"{path.name}: RSSI")
    axes[0].legend()

    axes[1].plot(
        frame["time_s"],
        frame["mean_csi_amplitude"],
        alpha=0.55,
        label="Raw mean CSI amplitude",
    )
    axes[1].plot(
        frame["time_s"],
        frame["rolling_csi_mean"],
        linewidth=2,
        label="Rolling mean",
    )
    axes[1].set_ylabel("Mean CSI amplitude")
    axes[1].set_title("CSI amplitude")
    axes[1].legend()

    axes[2].plot(
        frame["time_s"],
        frame["rssi_abs_change"],
        label="Absolute RSSI change",
    )
    axes[2].plot(
        frame["time_s"],
        frame["csi_abs_change"],
        label="Absolute CSI change",
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frame-to-frame change")
    axes[2].set_title("Transient-event features")
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def save_comparison_plots(
    parsed: dict[Path, pd.DataFrame], output_dir: Path
) -> list[Path]:
    """Compare several files after trimming them to the same number of frames."""
    if len(parsed) < 2:
        return []

    common_length = min(len(frame) for frame in parsed.values())
    outputs: list[Path] = []

    for metric, ylabel, filename in [
        ("rssi", "RSSI (dBm)", "comparison_rssi.png"),
        ("mean_csi_amplitude", "Mean CSI amplitude", "comparison_csi_amplitude.png"),
        ("csi_abs_change", "Absolute CSI frame change", "comparison_csi_change.png"),
    ]:
        figure = plt.figure(figsize=(11, 6))

        for path, frame in parsed.items():
            trimmed = frame.iloc[:common_length]
            plt.plot(
                trimmed["frame"],
                trimmed[metric],
                linewidth=1.5,
                label=path.stem,
            )

        plt.xlabel(f"Frame, first {common_length} matched rows")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel}: file comparison")
        plt.legend()
        plt.tight_layout()

        output_path = output_dir / filename
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze filtered ESP32 CSI CSV files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more CSI CSV files.",
    )
    parser.add_argument(
        "--mac",
        default=DEFAULT_MAC,
        help=f"STA MAC address. Default: {DEFAULT_MAC}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("csi_analysis"),
        help="Output directory. Default: csi_analysis",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    parsed: dict[Path, pd.DataFrame] = {}
    summaries: list[dict[str, float | str]] = []

    for input_path in args.files:
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        frame = parse_csi_file(input_path, args.mac)
        parsed[input_path] = frame
        summaries.append(summarize_trial(input_path, frame))

        processed_csv = args.output / f"{input_path.stem}_processed.csv"
        frame.drop(
            columns=[],
            errors="ignore",
        ).to_csv(processed_csv, index=False)

        plot_path = save_individual_plot(input_path, frame, args.output)
        print(f"Created {processed_csv}")
        print(f"Created {plot_path}")

    summary_frame = pd.DataFrame(summaries)
    summary_path = args.output / "trial_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    print(f"Created {summary_path}")

    for comparison_path in save_comparison_plots(parsed, args.output):
        print(f"Created {comparison_path}")

    print("\nTrial summary:")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
