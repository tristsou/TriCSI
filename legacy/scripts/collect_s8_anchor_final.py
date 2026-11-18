from __future__ import annotations

"""
TriCSI-Anchor FINAL S8 collector

This is the ONE new standardized session for the frozen final method.

Protocol
--------
1) EMPTY baseline:
   - 12 s, no person in the sensing area
   - saved as EMPTY_trial01_NA.csv
   - used ONLY as the target-session background calibration

2) Occupied one-shot calibration:
   - one 8 s trial at each Z1..Z9
   - facing N
   - saved as trial01 for each zone
   - used ONLY to form target-session prototypes

3) Final evaluation:
   - four 8 s trials per occupied zone: E, S, W, N
   - four EMPTY trials
   - randomized
   - saved as trials 02..05
   - NEVER used to fit or calibrate the model

Keyboard problem solved:
Press ENTER while at the computer, then you have --settle seconds (default 5)
to move to the requested location and become still. Recording starts only after
the countdown.

Hardware must remain in the standardized S4+ geometry.
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
from project_config import (
    RAW_DATA_DIR,
    SERIAL_BAUD,
    SERIAL_PORT,
    ZONE_CENTERS,
)


ZONES = [f"Z{i}" for i in range(1, 10)]
CALIBRATION_ORIENTATIONS = {1: "N", 2: "E"}
TEST_ORIENTATIONS = {3: "S", 4: "W", 5: "N", 6: "E"}


def estimate_rate(times: list[float]) -> float:
    if len(times) < 3:
        return 0.0
    x = np.asarray(times, dtype=np.float64)
    dt = np.diff(x)
    dt = dt[(dt > 0.002) & (dt < 1.0)]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def open_serial_safely(port: str, baud: int) -> serial.Serial:
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


def read_available_lines(
    ser: serial.Serial,
    buffer: bytearray,
) -> tuple[list[bytes], int]:
    waiting = ser.in_waiting
    chunk = ser.read(waiting if waiting > 0 else 4096)
    if not chunk:
        return [], 0

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

    return lines, len(chunk)


def preflight_serial(ser: serial.Serial, seconds: float = 2.5) -> None:
    print(f"Checking both links for {seconds:.1f} s...")
    ser.reset_input_buffer()

    buffer = bytearray()
    start = time.monotonic()
    counts = defaultdict(int)
    timestamps = {"left": [], "right": []}

    while time.monotonic() - start < seconds:
        lines, _ = read_available_lines(ser, buffer)
        for line in lines:
            record = parse_csi_line(line)
            if record is None:
                continue
            counts[record.link] += 1
            timestamps[record.link].append(record.device_time_s)

    lr = estimate_rate(timestamps["left"])
    rr = estimate_rate(timestamps["right"])

    print(
        f"Preflight: LEFT={counts['left']} (~{lr:.1f} Hz), "
        f"RIGHT={counts['right']} (~{rr:.1f} Hz)"
    )

    if counts["left"] == 0 or counts["right"] == 0:
        raise SystemExit(
            "Both links are not live. Check COM9/COM10 power, COM11 AP, "
            "and close ESP-IDF Monitor."
        )


def countdown(seconds: float, message: str) -> None:
    end = time.monotonic() + seconds
    last = None
    while True:
        rem = end - time.monotonic()
        if rem <= 0:
            break
        shown = int(np.ceil(rem))
        if shown != last:
            print(f"  {message} {shown}")
            last = shown
        time.sleep(0.05)


def record_trial(
    ser: serial.Serial,
    path: Path,
    *,
    session: str,
    zone: str,
    trial: int,
    orientation: str,
    seconds: float,
) -> dict:
    ser.reset_input_buffer()

    buffer = bytearray()
    counts = defaultdict(int)
    timestamps = {"left": [], "right": []}

    x_m, y_m = ("", "")
    if zone != "EMPTY":
        x_m, y_m = ZONE_CENTERS[zone]

    start_host = time.monotonic()
    end_host = start_host + seconds

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_csv_header())
        writer.writeheader()

        while time.monotonic() < end_host:
            lines, _ = read_available_lines(ser, buffer)

            for line in lines:
                record = parse_csi_line(line)
                if record is None:
                    continue

                counts[record.link] += 1
                timestamps[record.link].append(record.device_time_s)

                write_record(
                    writer,
                    record,
                    session=session,
                    zone=zone,
                    trial=trial,
                    orientation=orientation,
                    x_m=x_m,
                    y_m=y_m,
                )

    lr = estimate_rate(timestamps["left"])
    rr = estimate_rate(timestamps["right"])

    print(
        f"  saved {path.name}: "
        f"L={counts['left']} ({lr:.1f} Hz), "
        f"R={counts['right']} ({rr:.1f} Hz)"
    )

    if counts["left"] < 100 or counts["right"] < 100:
        print(
            "  WARNING: unusually few packets. Do not move hardware; "
            "repeat only this trial if necessary."
        )

    return {
        "left_packets": counts["left"],
        "right_packets": counts["right"],
        "left_rate": lr,
        "right_rate": rr,
    }


def prompt_and_record(
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
        print("After ENTER, leave the entire marked 3x3 sensing area.")
    else:
        x_m, y_m = ZONE_CENTERS[zone]
        print(
            f"After ENTER, move to {zone} center "
            f"({x_m:.2f} m, {y_m:.2f} m), facing {orientation}."
        )

    response = input(
        f"Press ENTER now; you will have {settle:.1f} s to move/settle "
        "(q + ENTER to stop): "
    ).strip().lower()

    if response == "q":
        return False

    countdown(settle, "move and settle...")

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
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--session", default="S8")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--empty-seconds", type=float, default=12.0)
    parser.add_argument("--settle", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    session_dir = RAW_DATA_DIR / args.session
    session_dir.mkdir(parents=True, exist_ok=True)

    schedule_rows = []

    # Phase A: one explicit EMPTY background calibration.
    schedule_rows.append(
        {
            "phase": "background_calibration",
            "zone": "EMPTY",
            "trial": 1,
            "orientation": "NA",
            "seconds": args.empty_seconds,
        }
    )

    # Phase B: small orientation-balanced target calibration.
    # Two short orthogonal views per zone: N and E.
    calibration = []
    for z in ZONES:
        for trial, orientation in CALIBRATION_ORIENTATIONS.items():
            calibration.append(
                {
                    "phase": "target_calibration",
                    "zone": z,
                    "trial": trial,
                    "orientation": orientation,
                    "seconds": 4.0,
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(calibration)
    schedule_rows.extend(calibration)

    # Phase C: untouched final evaluation.
    # Occupied: independent S/W/N/E trials 03..06.
    # EMPTY: trials02..05.
    evaluation = []

    for z in ZONES:
        for trial, orientation in TEST_ORIENTATIONS.items():
            evaluation.append(
                {
                    "phase": "final_evaluation",
                    "zone": z,
                    "trial": trial,
                    "orientation": orientation,
                    "seconds": args.seconds,
                }
            )

    for trial in range(2, 6):
        evaluation.append(
            {
                "phase": "final_evaluation",
                "zone": "EMPTY",
                "trial": trial,
                "orientation": "NA",
                "seconds": args.seconds,
            }
        )

    rng.shuffle(evaluation)
    schedule_rows.extend(evaluation)

    schedule_path = session_dir / f"schedule_{args.session}_anchor_final.csv"
    with schedule_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["order", "phase", "zone", "trial", "orientation", "seconds"],
        )
        writer.writeheader()
        for order, row in enumerate(schedule_rows, 1):
            writer.writerow({"order": order, **row})

    print("\n" + "=" * 78)
    print("TriCSI-Anchor FINAL S8 COLLECTION")
    print("=" * 78)
    print(f"Session        : {args.session}")
    print(f"Port           : {args.port} @ {args.baud}")
    print(f"Move/settle    : {args.settle:.1f} s")
    print(f"Occupied trial : {args.seconds:.1f} s")
    print(f"EMPTY baseline : {args.empty_seconds:.1f} s")
    print(f"Schedule       : {schedule_path}")
    print()
    print("S8 is the new untouched final session after S7 became development data.")
    print("Occupied calibration: trial01=N 4 s + trial02=E 4 s per zone.")
    print("Occupied final test: trials03-06 = S/W/N/E, independent recordings.")
    print("Do NOT move boxes, antennas, USB/coax routing, desk, monitor, or grid.")
    print("=" * 78)

    try:
        ser = open_serial_safely(args.port, args.baud)
    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {args.port}: {exc}\n"
            "Close ESP-IDF Monitor and every other program using COM11."
        ) from exc

    try:
        preflight_serial(ser)

        for order, row in enumerate(schedule_rows, 1):
            zone = row["zone"]
            trial = int(row["trial"])
            orientation = row["orientation"]
            phase = row["phase"]

            filename = f"{zone}_trial{trial:02d}_{orientation}.csv"
            path = session_dir / filename

            if path.exists() and not args.overwrite:
                print(
                    f"[{order:02d}/{len(schedule_rows)}] "
                    f"SKIP existing {filename}"
                )
                continue

            seconds = float(row["seconds"])

            label = (
                f"[{order:02d}/{len(schedule_rows)}] "
                f"{phase.upper()}: {zone} trial {trial} facing {orientation}"
            )

            ok = prompt_and_record(
                ser,
                path=path,
                session=args.session,
                zone=zone,
                trial=trial,
                orientation=orientation,
                seconds=seconds,
                settle=args.settle,
                label=label,
            )

            if not ok:
                print("Stopped. Existing files are safe.")
                break

    finally:
        try:
            ser.close()
        except Exception:
            pass

    print("\nCollection finished.")
    print(
        "Next command after S8 is complete:\n"
        "  python train_tricsi_anchor_final.py --test-session S8"
    )


if __name__ == "__main__":
    main()
