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


def open_serial_safely(port: str, baud: int) -> serial.Serial:
    """
    Open COM11 using the SAME sequence as the already-working real-time
    dual-link locator.

    Important: on this ESP32-S3/Windows setup, pre-setting DTR/RTS before
    opening the port can behave differently from opening first and then
    deasserting the control lines. The real-time locator is known to work,
    so the collector deliberately copies that exact order.
    """
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=0.02,
        write_timeout=1.0,
    )

    # This order matches serial_csi_dual_link_realtime_lltf_fixed.py.
    ser.dtr = False
    ser.rts = False

    try:
        ser.set_buffer_size(rx_size=262_144, tx_size=16_384)
    except (AttributeError, NotImplementedError, OSError):
        pass

    ser.reset_input_buffer()
    return ser


def read_available_lines(
    ser: serial.Serial,
    buffer: bytearray,
) -> tuple[list[bytes], int]:
    """Chunked serial reader, matching the low-latency detector approach."""
    waiting = ser.in_waiting
    chunk = ser.read(waiting if waiting > 0 else 4096)
    if not chunk:
        return [], 0

    nbytes = len(chunk)
    buffer.extend(chunk)
    lines = []

    while True:
        newline_index = buffer.find(b"\n")
        if newline_index < 0:
            break
        line = bytes(buffer[:newline_index]).rstrip(b"\r")
        del buffer[: newline_index + 1]
        lines.append(line)

    if len(buffer) > 1_000_000:
        buffer.clear()

    return lines, nbytes


def preflight_serial(
    ser: serial.Serial,
    max_wait_seconds: float = 30.0,
    min_packets_each: int = 8,
) -> None:
    """
    Opening COM11 resets this ESP32-S3/AP on the current Windows hardware.

    Therefore do NOT use a fixed 4-second preflight. Wait through boot, AP
    startup, and STA reconnection until CSI from BOTH configured transmitters
    is actually visible, or fail after a generous timeout.
    """
    print(
        f"COM11 may reboot when the serial port opens. "
        f"Waiting up to {max_wait_seconds:.0f} s for AP boot + both CSI links..."
    )

    buffer = bytearray()
    start = time.monotonic()
    last_status = start

    bytes_seen = 0
    lines_seen = 0
    csi_lines = 0
    accepted = 0
    counts = defaultdict(int)
    timestamps = {"left": [], "right": []}
    recent_text = []

    while time.monotonic() - start < max_wait_seconds:
        lines, nbytes = read_available_lines(ser, buffer)
        bytes_seen += nbytes

        for line in lines:
            lines_seen += 1
            decoded = line.decode("utf-8", errors="replace")
            recent_text.append(decoded[:240])
            if len(recent_text) > 20:
                recent_text.pop(0)

            if b"CSI_DATA" in line:
                csi_lines += 1

            record = parse_csi_line(line)
            if record is None:
                continue

            accepted += 1
            counts[record.link] += 1
            timestamps[record.link].append(record.device_time_s)

        now = time.monotonic()

        if (
            counts["left"] >= min_packets_each
            and counts["right"] >= min_packets_each
        ):
            left_rate = estimate_rate(timestamps["left"])
            right_rate = estimate_rate(timestamps["right"])

            print(
                f"Preflight PASSED after {now - start:.1f} s: "
                f"bytes={bytes_seen}, lines={lines_seen}, CSI={csi_lines}"
            )
            print(
                f"  left : {counts['left']} packets (~{left_rate:.1f} pkt/s)\n"
                f"  right: {counts['right']} packets (~{right_rate:.1f} pkt/s)"
            )
            print("Both CSI links are live. Starting the location schedule.\n")

            # Clear boot/preflight backlog so trial 1 begins with fresh packets.
            try:
                ser.reset_input_buffer()
            except serial.SerialException:
                pass
            return

        if now - last_status >= 2.0:
            print(
                f"  waiting... {now - start:4.1f}s | "
                f"bytes={bytes_seen} | CSI={csi_lines} | "
                f"left={counts['left']} right={counts['right']}"
            )
            last_status = now

    print("\nPRE-FLIGHT TIMED OUT.")
    print(
        f"bytes={bytes_seen}, lines={lines_seen}, CSI lines={csi_lines}, "
        f"accepted={accepted}, left={counts['left']}, right={counts['right']}"
    )

    if recent_text:
        print("\nLast serial lines seen:")
        for line in recent_text[-12:]:
            print("  " + line)

    if bytes_seen == 0:
        raise SystemExit(
            "\nNo serial bytes arrived at all. Check COM11/power/port ownership."
        )

    if csi_lines == 0:
        raise SystemExit(
            "\nThe AP booted and produced serial logs, but no CSI_DATA rows "
            "appeared within 30 seconds. Check AP CSI-to-serial and whether "
            "COM9/COM10 reconnected to the AP."
        )

    if counts["left"] == 0 or counts["right"] == 0:
        raise SystemExit(
            "\nCSI_DATA exists, but both configured transmitter MACs were not "
            "seen. Check COM9/COM10 power/Wi-Fi and the MAC values in "
            "project_config.py."
        )

    raise SystemExit("\nPreflight did not become healthy enough to collect data.")


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
        ser = open_serial_safely(args.port, args.baud)
    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {args.port}: {exc}\n"
            "Close ESP-IDF Monitor and every other program using the AP port."
        ) from exc

    try:
        preflight_serial(ser)

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
            serial_bytes = 0
            serial_buffer = bytearray()

            print(f"RECORDING {args.seconds:.1f} s — stay in position...")
            start = time.monotonic()
            end = start + args.seconds

            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=trial_csv_header())
                writer.writeheader()

                while time.monotonic() < end:
                    lines, nbytes = read_available_lines(ser, serial_buffer)
                    serial_bytes += nbytes

                    for line in lines:
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
                f"  accepted/raw serial lines: {accepted}/{raw_lines}\n"
                f"  serial bytes: {serial_bytes}"
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
