from __future__ import annotations

"""
Three-transmitter TriCSI trial collector.

Hardware mapping
----------------
Receiver/AP: COM5
Back-left TX:   02:00:00:00:00:01
Middle-back TX: 02:00:00:00:00:04
Back-right TX:  02:00:00:00:00:03

Default full-session protocol
-----------------------------
- Classes: EMPTY and Z1-Z9
- Five trials per class
- Occupied orientations: N, E, S, W, N
- Eight seconds per ordinary trial
- EMPTY trial 1 is collected first for 12 seconds as the background reference
- The remaining 49 trials are randomized
- Output: data_3tx/raw/<SESSION>/

The file format retains the source MAC and iq_000 ... iq_127 columns used by
the earlier TriCSI analysis, while adding a link column for the third path.
"""

import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import serial
except ImportError:
    serial = None


DEFAULT_PORT = "COM5"
DEFAULT_BAUD = 921600
DEFAULT_RAW_DIR = Path("data_3tx") / "raw"

MAC_TO_LINK = {
    "02:00:00:00:00:01": "back_left",
    "02:00:00:00:00:04": "middle_back",
    "02:00:00:00:00:03": "back_right",
}
LINK_ORDER = ("back_left", "middle_back", "back_right")

CLASS_ORDER = ("EMPTY", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9")
ORIENTATIONS = ("N", "E", "S", "W")

# Existing 8 ft x 6 ft grid centers, in meters.
ZONE_CENTERS = {
    "Z1": (0.4064, 1.5240),
    "Z2": (1.1176, 1.5240),
    "Z3": (1.8288, 1.5240),
    "Z4": (0.4064, 0.9144),
    "Z5": (1.1176, 0.9144),
    "Z6": (1.8288, 0.9144),
    "Z7": (0.4064, 0.3048),
    "Z8": (1.1176, 0.3048),
    "Z9": (1.8288, 0.3048),
}

ANSI_ESCAPE = re.compile(rb"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
RADIO_FIELDS = (
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "local_timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "real_time_set",
    "real_timestamp",
    "len",
)
IQ_COLUMNS = tuple(f"iq_{i:03d}" for i in range(128))
CSV_HEADER = (
    "host_time_s",
    "session",
    "zone",
    "trial",
    "orientation",
    "x_m",
    "y_m",
    "link",
    "mac",
    "role",
    *RADIO_FIELDS,
    *IQ_COLUMNS,
)


@dataclass(frozen=True)
class CSIRecord:
    link: str
    mac: str
    role: str
    host_time_s: float
    radio: dict[str, str]
    iq: tuple[int, ...]


def normalize_mac(value: str) -> str:
    return value.strip().upper()


def parse_csi_line(raw_line: bytes) -> CSIRecord | None:
    """Parse one ESP32-CSI-Tool `CSI_DATA,AP,...` record."""
    cleaned = ANSI_ESCAPE.sub(b"", raw_line).decode("utf-8", errors="ignore")
    start = cleaned.find("CSI_DATA,AP,")
    if start < 0:
        return None

    # The final field is the space-separated CSI vector inside brackets.
    fields = cleaned[start:].strip().split(",", 25)
    if len(fields) != 26:
        return None

    mac = normalize_mac(fields[2])
    link = MAC_TO_LINK.get(mac)
    if link is None:
        return None

    payload = fields[25]
    left_bracket = payload.find("[")
    right_bracket = payload.rfind("]")
    if left_bracket < 0 or right_bracket <= left_bracket:
        return None

    try:
        values = tuple(int(v) for v in payload[left_bracket + 1 : right_bracket].split())
    except ValueError:
        return None

    if len(values) < 128:
        return None

    # Preserve the same first 128 LLTF I/Q values used by the earlier pipeline.
    values = values[:128]
    radio_values = fields[3:25]
    radio = dict(zip(RADIO_FIELDS, radio_values, strict=True))

    return CSIRecord(
        link=link,
        mac=mac,
        role=fields[1].strip(),
        host_time_s=time.time(),
        radio=radio,
        iq=values,
    )


def open_serial_safely(port: str, baud: int):
    if serial is None:
        raise SystemExit(
            "PySerial is required. Install it with:\n"
            "  python -m pip install pyserial"
        )
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


def read_available_lines(ser, buffer: bytearray) -> list[bytes]:
    waiting = ser.in_waiting
    chunk = ser.read(waiting if waiting > 0 else 4096)
    if not chunk:
        return []

    buffer.extend(chunk)
    lines: list[bytes] = []

    while True:
        newline = buffer.find(b"\n")
        if newline < 0:
            break
        lines.append(bytes(buffer[:newline]).rstrip(b"\r"))
        del buffer[: newline + 1]

    if len(buffer) > 1_000_000:
        buffer.clear()

    return lines


def estimate_rate(host_times: list[float]) -> float:
    if len(host_times) < 2:
        return 0.0
    span = host_times[-1] - host_times[0]
    return (len(host_times) - 1) / span if span > 0 else 0.0


def preflight_serial(
    ser,
    *,
    max_wait_seconds: float,
    min_packets_each: int,
) -> dict[str, dict[str, float]]:
    """Wait for all three MAC addresses before allowing collection."""
    print(f"Waiting up to {max_wait_seconds:.0f} s for all three transmitters...")
    ser.reset_input_buffer()
    buffer = bytearray()
    times: dict[str, list[float]] = {link: [] for link in LINK_ORDER}
    deadline = time.monotonic() + max_wait_seconds

    while time.monotonic() < deadline:
        for raw_line in read_available_lines(ser, buffer):
            record = parse_csi_line(raw_line)
            if record is not None:
                times[record.link].append(time.monotonic())

        counts = {link: len(times[link]) for link in LINK_ORDER}
        if all(count >= min_packets_each for count in counts.values()):
            break

    result = {
        link: {
            "packets": len(times[link]),
            "rate_hz": estimate_rate(times[link]),
        }
        for link in LINK_ORDER
    }

    print("\nPreflight results:")
    for link in LINK_ORDER:
        print(
            f"  {link:12s} {result[link]['packets']:4.0f} packets  "
            f"~{result[link]['rate_hz']:5.1f} Hz  "
            f"{next(mac for mac, name in MAC_TO_LINK.items() if name == link)}"
        )

    missing = [link for link in LINK_ORDER if result[link]["packets"] < min_packets_each]
    if missing:
        raise SystemExit(
            "\nPreflight failed. Missing or weak links: "
            + ", ".join(missing)
            + ". Check transmitter power and MAC mapping before collecting trials."
        )

    return result


def build_schedule(
    *,
    trials_per_zone: int,
    trial_seconds: float,
    empty_baseline_seconds: float,
    seed: int,
) -> list[dict]:
    if trials_per_zone < 1:
        raise ValueError("trials_per_zone must be at least 1")

    baseline = {
        "phase": "background_calibration",
        "zone": "EMPTY",
        "trial": 1,
        "orientation": "NA",
        "seconds": empty_baseline_seconds,
    }

    remaining: list[dict] = []
    for zone in CLASS_ORDER:
        for trial in range(1, trials_per_zone + 1):
            if zone == "EMPTY" and trial == 1:
                continue

            orientation = (
                "NA"
                if zone == "EMPTY"
                else ORIENTATIONS[(trial - 1) % len(ORIENTATIONS)]
            )
            remaining.append(
                {
                    "phase": "standard_trial",
                    "zone": zone,
                    "trial": trial,
                    "orientation": orientation,
                    "seconds": trial_seconds,
                }
            )

    random.Random(seed).shuffle(remaining)
    return [baseline, *remaining]


def countdown(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    shown_last = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        shown = max(1, int(remaining + 0.999))
        if shown != shown_last:
            print(f"  Move and settle: {shown}")
            shown_last = shown
        time.sleep(0.05)


def record_to_row(
    record: CSIRecord,
    *,
    session: str,
    zone: str,
    trial: int,
    orientation: str,
) -> dict:
    x_m, y_m = ("", "") if zone == "EMPTY" else ZONE_CENTERS[zone]
    row = {
        "host_time_s": f"{record.host_time_s:.6f}",
        "session": session,
        "zone": zone,
        "trial": trial,
        "orientation": orientation,
        "x_m": x_m,
        "y_m": y_m,
        "link": record.link,
        "mac": record.mac,
        "role": record.role,
        **record.radio,
    }
    row.update({name: value for name, value in zip(IQ_COLUMNS, record.iq, strict=True)})
    return row


def record_trial(
    ser,
    path: Path,
    *,
    session: str,
    zone: str,
    trial: int,
    orientation: str,
    seconds: float,
) -> dict[str, dict[str, float]]:
    ser.reset_input_buffer()
    buffer = bytearray()
    counts = defaultdict(int)
    times: dict[str, list[float]] = {link: [] for link in LINK_ORDER}
    deadline = time.monotonic() + seconds

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()

        while time.monotonic() < deadline:
            for raw_line in read_available_lines(ser, buffer):
                record = parse_csi_line(raw_line)
                if record is None:
                    continue

                counts[record.link] += 1
                times[record.link].append(time.monotonic())
                writer.writerow(
                    record_to_row(
                        record,
                        session=session,
                        zone=zone,
                        trial=trial,
                        orientation=orientation,
                    )
                )

    quality = {
        link: {
            "packets": counts[link],
            "rate_hz": estimate_rate(times[link]),
        }
        for link in LINK_ORDER
    }

    summary = " | ".join(
        f"{link}={quality[link]['packets']:.0f} ({quality[link]['rate_hz']:.1f} Hz)"
        for link in LINK_ORDER
    )
    print(f"  Saved {path.name}: {summary}")
    return quality


def write_schedule(path: Path, schedule: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("order", "phase", "zone", "trial", "orientation", "seconds"),
        )
        writer.writeheader()
        for order, item in enumerate(schedule, 1):
            writer.writerow({"order": order, **item})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect three-transmitter TriCSI localization trials."
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--session", default="3TX_S1")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--trials-per-zone", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--empty-baseline-seconds", type=float, default=12.0)
    parser.add_argument("--settle", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=309)
    parser.add_argument("--preflight-seconds", type=float, default=30.0)
    parser.add_argument("--preflight-min-packets", type=int, default=8)
    parser.add_argument("--min-trial-packets", type=int, default=100)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if serial is None:
        raise SystemExit(
            "PySerial is required. Install it with:\n"
            "  python -m pip install pyserial"
        )

    if args.seconds <= 0 or args.empty_baseline_seconds <= 0 or args.settle < 0:
        raise SystemExit("Durations must be positive, and --settle cannot be negative.")

    session_dir = args.raw_dir / args.session
    session_dir.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(
        trials_per_zone=args.trials_per_zone,
        trial_seconds=args.seconds,
        empty_baseline_seconds=args.empty_baseline_seconds,
        seed=args.seed,
    )
    schedule_path = session_dir / f"schedule_{args.session}_3tx.csv"
    write_schedule(schedule_path, schedule)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session": args.session,
        "receiver_port": args.port,
        "baud": args.baud,
        "mac_to_link": MAC_TO_LINK,
        "trials_per_zone": args.trials_per_zone,
        "trial_seconds": args.seconds,
        "empty_baseline_seconds": args.empty_baseline_seconds,
        "settle_seconds": args.settle,
        "seed": args.seed,
        "zone_centers_m": ZONE_CENTERS,
    }
    with (session_dir / "session_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\n" + "=" * 78)
    print("TRICSI THREE-TRANSMITTER COLLECTION")
    print("=" * 78)
    print(f"Receiver/AP : {args.port} @ {args.baud}")
    print(f"Session     : {args.session}")
    print(f"Output      : {session_dir.resolve()}")
    print(f"Trials      : {len(schedule)} total")
    print(f"Schedule    : {schedule_path}")
    print("Transmitters: COM9 back-left, COM11 middle-back, COM10 back-right")
    print("Do not open the transmitter COM ports while collecting.")
    print("Do not move boards, antennas, furniture, cables, or grid marks.")
    print("=" * 78)

    try:
        ser = open_serial_safely(args.port, args.baud)
    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {args.port}: {exc}\n"
            "Close ESP-IDF Monitor, Arduino Serial Monitor, and any other "
            "program using the receiver port."
        ) from exc

    try:
        preflight_serial(
            ser,
            max_wait_seconds=args.preflight_seconds,
            min_packets_each=args.preflight_min_packets,
        )

        if args.preflight_only:
            print("\nPreflight passed. No trials were collected.")
            return

        for order, item in enumerate(schedule, 1):
            zone = str(item["zone"])
            trial = int(item["trial"])
            orientation = str(item["orientation"])
            seconds = float(item["seconds"])
            filename = f"{zone}_trial{trial:02d}_{orientation}.csv"
            path = session_dir / filename

            if path.exists() and not args.overwrite:
                print(f"[{order:02d}/{len(schedule)}] SKIP existing {filename}")
                continue

            print("\n" + "-" * 78)
            print(
                f"[{order:02d}/{len(schedule)}] NEXT: "
                f"{zone}, trial {trial}, facing {orientation}"
            )
            if zone == "EMPTY":
                print("After ENTER, leave the entire marked sensing area.")
            else:
                x_m, y_m = ZONE_CENTERS[zone]
                print(
                    f"After ENTER, stand at {zone} center "
                    f"({x_m:.2f} m, {y_m:.2f} m), facing {orientation}."
                )

            response = input(
                f"Press ENTER for a {args.settle:.1f} s move/settle countdown "
                "or type q then ENTER to stop: "
            ).strip().lower()
            if response == "q":
                print("Collection stopped. Existing files are safe.")
                break

            countdown(args.settle)
            print(f"  RECORDING for {seconds:.1f} s - remain still.")
            quality = record_trial(
                ser,
                path,
                session=args.session,
                zone=zone,
                trial=trial,
                orientation=orientation,
                seconds=seconds,
            )

            weak = [
                link
                for link in LINK_ORDER
                if quality[link]["packets"] < args.min_trial_packets
            ]
            if weak:
                print(
                    "  WARNING: low packet count for "
                    + ", ".join(weak)
                    + ". Note this trial and repeat it later if necessary."
                )

    except KeyboardInterrupt:
        print("\nCollection interrupted. Completed files are safe.")
    finally:
        ser.close()

    print("\nCollection finished.")


if __name__ == "__main__":
    main()
