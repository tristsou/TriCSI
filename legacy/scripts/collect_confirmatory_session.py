from __future__ import annotations

"""
TriCSI confirmatory-session collector for S9 / S10.

Purpose
-------
Collect genuinely new sessions WITHOUT changing the frozen final model.

Model calibration allowed:
    EMPTY trial01 only (12 seconds).

Evaluation:
    4 occupied trials per zone, one each facing N/E/S/W.
    4 additional EMPTY trials.
    All evaluation trials are randomized.

Important:
- The zone labels are saved only so accuracy can be scored afterward.
- The frozen model does NOT use any labeled target-zone trial for adaptation.
- Keep the standardized S4+ hardware geometry unchanged.
"""

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import serial

from csi_io import parse_csi_line, trial_csv_header, write_record
from project_config import RAW_DATA_DIR, SERIAL_BAUD, SERIAL_PORT, ZONE_CENTERS


ZONES = [f"Z{i}" for i in range(1, 10)]

# Four independent evaluation trials per occupied zone.
TRIAL_ORIENTATIONS = {
    1: "N",
    2: "E",
    3: "S",
    4: "W",
}


def estimate_rate(times):
    if len(times) < 3:
        return 0.0
    x = np.asarray(times, dtype=np.float64)
    dt = np.diff(x)
    dt = dt[(dt > 0.002) & (dt < 1.0)]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def open_serial_safely(port, baud):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.02
    ser.write_timeout = 1.0
    ser.dtr = False
    ser.rts = False
    ser.open()

    try:
        ser.set_buffer_size(rx_size=262_144, tx_size=16_384)
    except (AttributeError, NotImplementedError, OSError):
        pass

    time.sleep(2.5)
    ser.reset_input_buffer()
    return ser


def read_available_lines(ser, buffer):
    waiting = ser.in_waiting
    chunk = ser.read(waiting if waiting > 0 else 4096)

    if not chunk:
        return []

    buffer.extend(chunk)
    lines = []

    while True:
        j = buffer.find(b"\n")
        if j < 0:
            break

        line = bytes(buffer[:j]).rstrip(b"\r")
        del buffer[: j + 1]
        lines.append(line)

    if len(buffer) > 1_000_000:
        buffer.clear()

    return lines


def preflight(ser, seconds=2.5):
    print(f"Checking both CSI links for {seconds:.1f} s...")

    ser.reset_input_buffer()
    buffer = bytearray()
    start = time.monotonic()

    counts = defaultdict(int)
    times = {"left": [], "right": []}

    while time.monotonic() - start < seconds:
        for line in read_available_lines(ser, buffer):
            rec = parse_csi_line(line)
            if rec is None:
                continue

            counts[rec.link] += 1
            times[rec.link].append(rec.device_time_s)

    lr = estimate_rate(times["left"])
    rr = estimate_rate(times["right"])

    print(
        f"Preflight: LEFT={counts['left']} (~{lr:.1f} Hz), "
        f"RIGHT={counts['right']} (~{rr:.1f} Hz)"
    )

    if counts["left"] == 0 or counts["right"] == 0:
        raise SystemExit(
            "Both links are not live. Check the two STAs, AP/receiver, "
            "and make sure ESP-IDF Monitor is closed."
        )


def countdown(seconds, message):
    end = time.monotonic() + seconds
    last = None

    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break

        shown = int(np.ceil(remaining))
        if shown != last:
            print(f"  {message} {shown}")
            last = shown

        time.sleep(0.05)


def record_trial(
    ser,
    path,
    *,
    session,
    zone,
    trial,
    orientation,
    seconds,
):
    ser.reset_input_buffer()

    buffer = bytearray()
    counts = defaultdict(int)
    times = {"left": [], "right": []}

    if zone == "EMPTY":
        x_m, y_m = "", ""
    else:
        x_m, y_m = ZONE_CENTERS[zone]

    end_time = time.monotonic() + seconds

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=trial_csv_header(),
        )
        writer.writeheader()

        while time.monotonic() < end_time:
            for line in read_available_lines(ser, buffer):
                rec = parse_csi_line(line)
                if rec is None:
                    continue

                counts[rec.link] += 1
                times[rec.link].append(rec.device_time_s)

                write_record(
                    writer,
                    rec,
                    session=session,
                    zone=zone,
                    trial=trial,
                    orientation=orientation,
                    x_m=x_m,
                    y_m=y_m,
                )

    lr = estimate_rate(times["left"])
    rr = estimate_rate(times["right"])

    print(
        f"  saved {path.name}: "
        f"L={counts['left']} ({lr:.1f} Hz), "
        f"R={counts['right']} ({rr:.1f} Hz)"
    )

    if counts["left"] < 100 or counts["right"] < 100:
        print(
            "  WARNING: unusually few packets. "
            "Repeat only this trial if necessary."
        )


def run_one(
    ser,
    *,
    path,
    session,
    zone,
    trial,
    orientation,
    seconds,
    settle,
    label,
):
    print("\n" + "-" * 78)
    print(label)

    if zone == "EMPTY":
        print(
            "After ENTER, leave the entire marked sensing area."
        )
    else:
        print(
            f"After ENTER, move to {zone} and face {orientation}."
        )

    response = input(
        f"Press ENTER now; you have {settle:.1f} s to move/settle "
        "(q + ENTER to stop): "
    ).strip().lower()

    if response == "q":
        return False

    countdown(
        settle,
        "move and settle...",
    )

    print(
        f"  RECORDING NOW for {seconds:.1f} s — remain still."
    )

    record_trial(
        ser,
        path,
        session=session,
        zone=zone,
        trial=trial,
        orientation=orientation,
        seconds=seconds,
    )

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--session",
        required=True,
        help="Confirmatory session name, e.g. S9 or S10.",
    )
    parser.add_argument(
        "--port",
        default=SERIAL_PORT,
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=SERIAL_BAUD,
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="Seconds for each evaluation trial.",
    )
    parser.add_argument(
        "--empty-baseline-seconds",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9102026,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.session in {"S4", "S5", "S6", "S7", "S8"}:
        raise SystemExit(
            "--session must be a NEW confirmatory session such as S9 or S10."
        )

    session_dir = RAW_DATA_DIR / args.session
    session_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    schedule = []

    # The ONLY target-session model calibration measurement.
    schedule.append(
        {
            "phase": "empty_background_calibration",
            "zone": "EMPTY",
            "trial": 1,
            "orientation": "NA",
            "seconds": args.empty_baseline_seconds,
        }
    )

    # Occupied evaluation trials: ALL labels are evaluation-only.
    evaluation = []

    for zone in ZONES:
        for trial, orientation in TRIAL_ORIENTATIONS.items():
            evaluation.append(
                {
                    "phase": "confirmatory_evaluation",
                    "zone": zone,
                    "trial": trial,
                    "orientation": orientation,
                    "seconds": args.seconds,
                }
            )

    # Four EMPTY evaluation trials, separate from trial01 calibration.
    for trial in range(2, 6):
        evaluation.append(
            {
                "phase": "confirmatory_evaluation",
                "zone": "EMPTY",
                "trial": trial,
                "orientation": "NA",
                "seconds": args.seconds,
            }
        )

    rng = random.Random(
        args.seed
        + sum(ord(c) for c in args.session)
    )
    rng.shuffle(evaluation)

    schedule.extend(evaluation)

    schedule_path = (
        session_dir
        / f"schedule_{args.session}_confirmatory.csv"
    )

    with schedule_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "order",
                "phase",
                "zone",
                "trial",
                "orientation",
                "seconds",
            ],
        )
        writer.writeheader()

        for i, row in enumerate(schedule, 1):
            writer.writerow(
                {
                    "order": i,
                    **row,
                }
            )

    print("\n" + "=" * 78)
    print("TriCSI-Anchor CONFIRMATORY SESSION")
    print("=" * 78)
    print(f"Session        : {args.session}")
    print(f"Serial         : {args.port} @ {args.baud}")
    print(f"Move/settle    : {args.settle:.1f} s")
    print(f"Trial length   : {args.seconds:.1f} s")
    print(f"Schedule       : {schedule_path}")
    print()
    print(
        "IMPORTANT: The frozen model may use ONLY EMPTY trial01 from this "
        "session. All labeled occupied trials are evaluation data."
    )
    print(
        "Do not change model weights, hyperparameters, hardware geometry, "
        "antenna orientation, or preprocessing after starting this session."
    )
    print("=" * 78)

    try:
        ser = open_serial_safely(
            args.port,
            args.baud,
        )
    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {args.port}: {exc}\n"
            "Close ESP-IDF Monitor and every other program using the receiver."
        ) from exc

    try:
        preflight(ser)

        for i, row in enumerate(schedule, 1):
            filename = (
                f"{row['zone']}_trial{int(row['trial']):02d}_"
                f"{row['orientation']}.csv"
            )

            path = session_dir / filename

            if path.exists() and not args.overwrite:
                print(
                    f"[{i:02d}/{len(schedule)}] "
                    f"SKIP existing {filename}"
                )
                continue

            label = (
                f"[{i:02d}/{len(schedule)}] "
                f"{row['phase'].upper()}: "
                f"{row['zone']} trial {row['trial']} "
                f"facing {row['orientation']}"
            )

            ok = run_one(
                ser,
                path=path,
                session=args.session,
                zone=row["zone"],
                trial=int(row["trial"]),
                orientation=row["orientation"],
                seconds=float(row["seconds"]),
                settle=args.settle,
                label=label,
            )

            if not ok:
                print(
                    "Stopped. Existing files are safe."
                )
                break

    finally:
        try:
            ser.close()
        except Exception:
            pass

    print("\nCollection finished.")
    print(
        f"Evaluate WITHOUT retraining:\n"
        f"  python evaluate_frozen_anchor_confirmatory.py "
        f"--sessions {args.session}"
    )


if __name__ == "__main__":
    main()
