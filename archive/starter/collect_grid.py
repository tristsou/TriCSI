from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import serial

from csi_io import parse_csi_line, trial_csv_header, write_record
from project_config import (
    CLASS_ORDER,
    ORIENTATIONS,
    RAW_DATA_DIR,
    SERIAL_BAUD,
    SERIAL_PORT,
    ZONE_CENTERS,
)


def build_schedule(trials_per_zone: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = []

    for zone in CLASS_ORDER:
        for trial in range(1, trials_per_zone + 1):
            orientation = "NA" if zone == "EMPTY" else ORIENTATIONS[(trial - 1) % len(ORIENTATIONS)]
            items.append(
                {
                    "zone": zone,
                    "trial": trial,
                    "orientation": orientation,
                }
            )

    rng.shuffle(items)
    return items


def estimate_rate(times: list[float]) -> float:
    if len(times) < 3:
        return 0.0
    x = np.asarray(times, dtype=np.float64)
    dt = np.diff(x)
    dt = dt[(dt > 0.002) & (dt < 1.0)]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def countdown(seconds: float) -> None:
    whole = int(round(seconds))
    for remaining in range(whole, 0, -1):
        print(f"  settling... {remaining}", end="\r", flush=True)
        time.sleep(1)
    if whole:
        print(" " * 40, end="\r")


def zone_coordinates(zone: str) -> tuple[str | float, str | float]:
    if zone == "EMPTY":
        return "", ""
    return ZONE_CENTERS[zone]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 3x3 ESP32 CSI localization data collector."
    )
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--session", default="S1")
    parser.add_argument("--trials-per-zone", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    session_dir = RAW_DATA_DIR / args.session
    session_dir.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(args.trials_per_zone, args.seed)

    schedule_path = session_dir / f"schedule_{args.session}.csv"
    with schedule_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order", "zone", "trial", "orientation"])
        writer.writeheader()
        for order, item in enumerate(schedule, 1):
            writer.writerow({"order": order, **item})

    print("\nTriCSI GRID DATA COLLECTION")
    print("=" * 70)
    print(f"AP receiver : {args.port} @ {args.baud}")
    print(f"Session     : {args.session}")
    print(f"Trials/class: {args.trials_per_zone}")
    print(f"Recording   : {args.seconds:.1f} s per trial")
    print(f"Schedule    : {schedule_path}")
    print()
    print("IMPORTANT:")
    print("  - COM9 and COM10 stay powered; do not open their serial ports.")
    print("  - Do not move the three ESP32 boards, antennas, furniture, or cables.")
    print("  - At a Z1-Z9 trial, stand with both feet centered on the floor mark.")
    print("  - Be mostly still; natural breathing/small sway is fine.")
    print("  - For EMPTY, leave the sensing area.")
    print("  - N/E/S/W are body-facing directions relative to the room map.")
    print()

    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            timeout=0.05,
            write_timeout=1.0,
        )
        ser.dtr = False
        ser.rts = False
        time.sleep(0.3)
        ser.reset_input_buffer()
    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {args.port}: {exc}\n"
            "Close ESP-IDF Monitor and every other program using the AP port."
        ) from exc

    try:
        for order, item in enumerate(schedule, 1):
            zone = item["zone"]
            trial = item["trial"]
            orientation = item["orientation"]
            x_m, y_m = zone_coordinates(zone)

            filename = f"{zone}_trial{trial:02d}_{orientation}.csv"
            path = session_dir / filename

            if path.exists() and not args.overwrite:
                print(f"[{order:02d}/{len(schedule)}] SKIP existing {filename}")
                continue

            print("\n" + "-" * 70)
            print(f"[{order:02d}/{len(schedule)}] NEXT: {zone}  trial {trial}  facing {orientation}")
            if zone == "EMPTY":
                print("Leave the marked 3x3 area completely.")
            else:
                print(f"Stand at {zone} center ({x_m:.2f} m, {y_m:.2f} m), facing {orientation}.")

            response = input("Press ENTER when ready, or type q then ENTER to stop: ").strip().lower()
            if response == "q":
                print("Collection stopped. Existing files are safe.")
                break

            countdown(args.settle)
            ser.reset_input_buffer()

            counts = defaultdict(int)
            timestamps = {"left": [], "right": []}
            accepted = 0
            raw_lines = 0

            print(f"RECORDING {args.seconds:.1f} s — stay in position...")
            start = time.monotonic()
            end = start + args.seconds

            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=trial_csv_header())
                writer.writeheader()

                while time.monotonic() < end:
                    line = ser.readline()
                    if not line:
                        continue
                    raw_lines += 1

                    record = parse_csi_line(line)
                    if record is None:
                        continue

                    accepted += 1
                    counts[record.link] += 1
                    timestamps[record.link].append(record.device_time_s)

                    write_record(
                        writer,
                        record,
                        session=args.session,
                        zone=zone,
                        x_m=x_m,
                        y_m=y_m,
                        trial=trial,
                        orientation=orientation,
                    )

            left_rate = estimate_rate(timestamps["left"])
            right_rate = estimate_rate(timestamps["right"])

            print(f"SAVED: {path}")
            print(
                f"  left : {counts['left']:4d} packets  ~{left_rate:5.1f} pkt/s\n"
                f"  right: {counts['right']:4d} packets  ~{right_rate:5.1f} pkt/s\n"
                f"  accepted/raw serial lines: {accepted}/{raw_lines}"
            )

            if counts["left"] == 0 or counts["right"] == 0:
                print("  !!! BAD TRIAL: one link is missing. Repeat this trial.")
            elif left_rate < 15 or right_rate < 15:
                print("  !!! LOW RATE: aim for >20 pkt/s on both links before the final dataset.")

    finally:
        ser.close()

    print("\nDone. Next run:")
    print("  py -3.14 inspect_dataset.py")
    print("Then train:")
    print("  py -3.14 train_localizer.py")


if __name__ == "__main__":
    main()
