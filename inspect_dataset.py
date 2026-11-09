from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from csi_io import load_trial_csv
from project_config import RAW_DATA_DIR, RESULTS_DIR


def packet_rate(times: np.ndarray) -> float:
    if len(times) < 3:
        return 0.0
    dt = np.diff(times)
    dt = dt[(dt > 0.002) & (dt < 1.0)]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-check collected localization trials.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--min-rate", type=float, default=15.0)
    args = parser.parse_args()

    paths = sorted(p for p in args.raw_dir.rglob("*.csv") if not p.name.startswith("schedule_"))
    if not paths:
        raise SystemExit(f"No trial CSVs found under {args.raw_dir}")

    rows = []
    bad = 0

    print("\nDATASET QUALITY")
    print("=" * 92)
    print(f"{'file':36s} {'zone':6s} {'L rows':>7s} {'R rows':>7s} {'L Hz':>7s} {'R Hz':>7s} status")

    for path in paths:
        try:
            trial = load_trial_csv(path)
        except Exception as exc:
            print(f"{path.name:36s} ERROR: {exc}")
            bad += 1
            continue

        left_n = len(trial["left"]["time"])
        right_n = len(trial["right"]["time"])
        left_hz = packet_rate(trial["left"]["time"])
        right_hz = packet_rate(trial["right"]["time"])
        zone = trial["metadata"]["zone"]

        status = "OK"
        if left_n == 0 or right_n == 0:
            status = "MISSING LINK"
        elif left_hz < args.min_rate or right_hz < args.min_rate:
            status = "LOW RATE"
        elif min(left_n, right_n) / max(left_n, right_n) < 0.65:
            status = "IMBALANCED"

        if status != "OK":
            bad += 1

        print(
            f"{path.name[:36]:36s} {zone:6s} {left_n:7d} {right_n:7d} "
            f"{left_hz:7.1f} {right_hz:7.1f} {status}"
        )

        rows.append(
            {
                "file": str(path),
                "session": trial["metadata"]["session"],
                "zone": zone,
                "trial": trial["metadata"]["trial"],
                "orientation": trial["metadata"]["orientation"],
                "left_rows": left_n,
                "right_rows": right_n,
                "left_hz": f"{left_hz:.3f}",
                "right_hz": f"{right_hz:.3f}",
                "status": status,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "dataset_quality.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 92)
    print(f"Trials checked: {len(rows)}")
    print(f"Flagged       : {bad}")
    print(f"Saved report  : {out}")

    if bad:
        print("\nRepeat flagged trials before treating the dataset as final.")
    else:
        print("\nAll trials passed the basic link/rate checks.")


if __name__ == "__main__":
    main()
