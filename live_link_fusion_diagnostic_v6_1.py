from __future__ import annotations

"""
TriCSI V6.1 — LABELED LINK-FUSION DIAGNOSTIC WITH MOVE/SETTLE DELAY

Purpose
-------
This is NOT another tracker and it does NOT retrain anything.

It records a clean, ground-truth-labeled 5 second dwell at each requested zone
while saving the FULL probability vectors from:

    LEFT RF
    RIGHT RF
    current late-fusion model
    equal LEFT/RIGHT average
    class-weighted geometric fusion (PoE candidate)

Why this exists
---------------
The previous live sweep had no exact arrival/departure markers and only printed
the top class for LEFT and RIGHT. That is enough to see problems, but not enough
to know whether a failed zone (especially Z8 or Z3) still has useful hidden
probability mass in one or both links.

This script removes that ambiguity.

Default sequence:
    Z7 -> Z4 -> Z1 -> Z8 -> Z5 -> Z2 -> Z9 -> Z6 -> Z3

At each zone:
    1. Press ENTER while you are still at the keyboard.
    2. You then get a configurable move/settle countdown (default 4 s).
    3. Move to the requested tape mark and become still before countdown ends.
    4. Recording starts only AFTER the countdown, giving the 1 s CSI window
       time to flush movement/old-position packets.
    5. The script then records exactly ~5 seconds of NEW inference outputs.

It saves a CSV under RESULTS_DIR and prints per-zone + overall summaries.

Run:
    python live_link_fusion_diagnostic_v6.py

No plot is used. Keep ESP-IDF Monitor closed.
"""

import argparse
import collections
import csv
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import serial

from csi_io import parse_csi_line
from features import compute_window_features
from project_config import (
    MIN_PACKETS_PER_LINK,
    MODEL_DIR,
    RESULTS_DIR,
    SERIAL_BAUD,
    SERIAL_PORT,
    WINDOW_SECONDS,
)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

BUFFER_LOCK = threading.Lock()
PRED_LOCK = threading.Lock()
STOP_EVENT = threading.Event()

BUFFERS = {
    "left": collections.deque(),
    "right": collections.deque(),
}

STATS = {
    "left_packets": 0,
    "right_packets": 0,
    "bad_lines": 0,
    "bytes_read": 0,
    "serial_chunks": 0,
    "inferences": 0,
    "last_inference_ms": 0.0,
    "max_inference_ms": 0.0,
}

LATEST_PRED: Optional[dict] = None
SERIAL_ERROR: Optional[str] = None
INFERENCE_ERROR: Optional[str] = None


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def align_proba(model, X: np.ndarray, ordered_classes: list[str]) -> np.ndarray:
    raw = model.predict_proba(X)
    lookup = {str(c): i for i, c in enumerate(model.classes_)}
    out = np.zeros((len(X), len(ordered_classes)), dtype=np.float64)

    for j, cls in enumerate(ordered_classes):
        if cls in lookup:
            out[:, j] = raw[:, lookup[cls]]

    s = out.sum(axis=1, keepdims=True)
    return np.divide(
        out,
        s,
        out=np.full_like(out, 1.0 / len(ordered_classes)),
        where=s > 0,
    )


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, 0.0)
    s = float(np.sum(p))
    if s <= 0:
        return np.full_like(p, 1.0 / len(p))
    return p / s


def entropy01(p: np.ndarray) -> float:
    p = normalize(p)
    h = -float(np.sum(p * np.log(np.maximum(p, 1e-12))))
    return h / math.log(len(p))


def margin(p: np.ndarray) -> float:
    q = np.sort(normalize(p))
    if len(q) < 2:
        return 1.0
    return float(q[-1] - q[-2])


def top_zone(p: np.ndarray, classes: list[str]) -> str:
    return classes[int(np.argmax(p))]


def equal_fusion(p_left: np.ndarray, p_right: np.ndarray) -> np.ndarray:
    return normalize(0.5 * p_left + 0.5 * p_right)


def weighted_geometric_fusion(
    p_left: np.ndarray,
    p_right: np.ndarray,
    class_weights: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Class-weighted log opinion pool / product-of-experts candidate.

    For each zone z:
        score(z) =
            w_left(z)  * log P_left(z)
          + w_right(z) * log P_right(z)

    The existing class-specific OOF reliability weights are reused.
    No new parameters are fit here.
    """
    w = np.asarray(class_weights, dtype=np.float64)

    log_score = (
        w[0] * np.log(np.maximum(p_left, eps))
        + w[1] * np.log(np.maximum(p_right, eps))
    )

    log_score -= np.max(log_score)
    q = np.exp(log_score)
    return normalize(q)


def predict_one_window(bundle: dict, feature: dict[str, float]) -> dict:
    all_features = bundle["all_features"]
    left_features = bundle["left_features"]
    right_features = bundle["right_features"]
    occupied_classes = [str(x) for x in bundle["occupied_classes"]]

    try:
        X_all = np.asarray(
            [[feature[n] for n in all_features]],
            dtype=np.float64,
        )
        X_left = np.asarray(
            [[feature[n] for n in left_features]],
            dtype=np.float64,
        )
        X_right = np.asarray(
            [[feature[n] for n in right_features]],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Feature mismatch: live features are missing {exc}. "
            "Use the same features.py used for training."
        ) from exc

    occupancy_model = bundle["occupancy_model"]
    occ_raw = occupancy_model.predict_proba(X_all)
    occ_lookup = {str(c): i for i, c in enumerate(occupancy_model.classes_)}
    p_empty = float(occ_raw[0, occ_lookup["EMPTY"]])

    p_left = align_proba(
        bundle["left_model"],
        X_left,
        occupied_classes,
    )[0]

    p_right = align_proba(
        bundle["right_model"],
        X_right,
        occupied_classes,
    )[0]

    weights = np.asarray(bundle["weights"], dtype=np.float64)

    weighted_left = p_left * weights[0]
    weighted_right = p_right * weights[1]

    stack_X = np.hstack(
        [weighted_left[None, :], weighted_right[None, :]]
    )

    p_stack = align_proba(
        bundle["stacker"],
        stack_X,
        occupied_classes,
    )[0]

    # Current deployed late fusion.
    p_current = normalize(
        weighted_left + weighted_right + p_stack
    )

    # Diagnostics / one principled candidate.
    p_equal = equal_fusion(p_left, p_right)
    p_geo = weighted_geometric_fusion(
        p_left,
        p_right,
        weights,
    )

    return {
        "p_empty": p_empty,
        "p_left": p_left,
        "p_right": p_right,
        "p_stack": p_stack,
        "p_current": p_current,
        "p_equal": p_equal,
        "p_geo": p_geo,
    }


# ---------------------------------------------------------------------------
# Serial
# ---------------------------------------------------------------------------

def open_serial(port: str, baud: int) -> serial.Serial:
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=0.02,
            write_timeout=1.0,
            rtscts=False,
            dsrdtr=False,
        )
        ser.dtr = False
        ser.rts = False

        try:
            ser.set_buffer_size(
                rx_size=262_144,
                tx_size=16_384,
            )
        except (
            AttributeError,
            NotImplementedError,
            OSError,
        ):
            pass

        time.sleep(0.35)
        ser.reset_input_buffer()
        return ser

    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {port}: {exc}\n"
            "Close ESP-IDF Monitor and every other program using COM11."
        ) from exc


def serial_worker(ser: serial.Serial) -> None:
    global SERIAL_ERROR

    byte_buffer = bytearray()

    try:
        while not STOP_EVENT.is_set():
            waiting = ser.in_waiting
            chunk = ser.read(
                waiting if waiting > 0 else 4096
            )

            if not chunk:
                continue

            STATS["bytes_read"] += len(chunk)
            STATS["serial_chunks"] += 1
            byte_buffer.extend(chunk)

            while True:
                newline = byte_buffer.find(b"\n")
                if newline < 0:
                    break

                line = bytes(
                    byte_buffer[:newline]
                ).rstrip(b"\r")

                del byte_buffer[: newline + 1]

                try:
                    record = parse_csi_line(line)
                except Exception:
                    record = None
                    STATS["bad_lines"] += 1

                if record is None:
                    continue

                host_now = time.monotonic()

                packet = (
                    float(record.device_time_s),
                    float(record.rssi),
                    record.iq.copy(),
                    host_now,
                )

                with BUFFER_LOCK:
                    BUFFERS[record.link].append(
                        packet
                    )

                    STATS[
                        f"{record.link}_packets"
                    ] += 1

                    q = BUFFERS[record.link]
                    newest = float(q[-1][0])
                    cutoff = newest - 4.0

                    while (
                        q
                        and float(q[0][0]) < cutoff
                    ):
                        q.popleft()

            if len(byte_buffer) > 1_000_000:
                byte_buffer.clear()
                STATS["bad_lines"] += 1

    except serial.SerialException as exc:
        SERIAL_ERROR = str(exc)
        STOP_EVENT.set()


def newest_window(
    window_seconds: float,
    min_packets: int,
):
    with BUFFER_LOCK:
        if (
            not BUFFERS["left"]
            or not BUFFERS["right"]
        ):
            return None

        common_end = min(
            float(BUFFERS["left"][-1][0]),
            float(BUFFERS["right"][-1][0]),
        )

        start = common_end - window_seconds
        out = {}

        for link in (
            "left",
            "right",
        ):
            rows = [
                x
                for x in BUFFERS[link]
                if start
                <= float(x[0])
                <= common_end
            ]

            if len(rows) < min_packets:
                return None

            out[link] = {
                "rssi": np.asarray(
                    [x[1] for x in rows],
                    dtype=np.float64,
                ),
                "iq": np.asarray(
                    [x[2] for x in rows],
                    dtype=np.int8,
                ),
                "latest_host": float(
                    rows[-1][3]
                ),
            }

        out["common_end"] = common_end
        return out


def recent_packet_rate(
    link: str,
    span_s: float = 2.0,
) -> float:
    with BUFFER_LOCK:
        q = list(BUFFERS[link])

    if len(q) < 2:
        return 0.0

    end = float(q[-1][0])
    times = [
        float(x[0])
        for x in q
        if float(x[0]) >= end - span_s
    ]

    if len(times) < 2:
        return 0.0

    dt = times[-1] - times[0]

    return (
        float((len(times) - 1) / dt)
        if dt > 0
        else 0.0
    )


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------

def inference_worker(
    bundle: dict,
    infer_hz: float,
    window_seconds: float,
    min_packets: int,
) -> None:
    global LATEST_PRED, INFERENCE_ERROR

    occupied_classes = [
        str(x)
        for x in bundle["occupied_classes"]
    ]

    period = 1.0 / infer_hz
    next_run = time.monotonic()

    try:
        while not STOP_EVENT.is_set():
            now = time.monotonic()

            if now < next_run:
                time.sleep(
                    min(
                        0.01,
                        next_run - now,
                    )
                )
                continue

            next_run = now + period

            window = newest_window(
                window_seconds,
                min_packets,
            )

            if window is None:
                continue

            t0 = time.perf_counter()

            feature = compute_window_features(
                window["left"]["iq"],
                window["left"]["rssi"],
                window["right"]["iq"],
                window["right"]["rssi"],
            )

            pred = predict_one_window(
                bundle,
                feature,
            )

            inference_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            STATS["inferences"] += 1
            STATS[
                "last_inference_ms"
            ] = inference_ms

            STATS[
                "max_inference_ms"
            ] = max(
                STATS[
                    "max_inference_ms"
                ],
                inference_ms,
            )

            newest_host = max(
                float(
                    window["left"][
                        "latest_host"
                    ]
                ),
                float(
                    window["right"][
                        "latest_host"
                    ]
                ),
            )

            fresh_ms = max(
                0.0,
                (
                    time.monotonic()
                    - newest_host
                )
                * 1000.0,
            )

            result = {
                **pred,
                "left_top": top_zone(
                    pred["p_left"],
                    occupied_classes,
                ),
                "right_top": top_zone(
                    pred["p_right"],
                    occupied_classes,
                ),
                "current_top": top_zone(
                    pred["p_current"],
                    occupied_classes,
                ),
                "equal_top": top_zone(
                    pred["p_equal"],
                    occupied_classes,
                ),
                "geo_top": top_zone(
                    pred["p_geo"],
                    occupied_classes,
                ),
                "left_entropy": entropy01(
                    pred["p_left"]
                ),
                "right_entropy": entropy01(
                    pred["p_right"]
                ),
                "left_margin": margin(
                    pred["p_left"]
                ),
                "right_margin": margin(
                    pred["p_right"]
                ),
                "inference_ms": inference_ms,
                "fresh_ms": fresh_ms,
                "device_end": float(
                    window["common_end"]
                ),
                "timestamp": time.monotonic(),
            }

            with PRED_LOCK:
                LATEST_PRED = result

    except Exception as exc:
        INFERENCE_ERROR = repr(exc)
        STOP_EVENT.set()


# ---------------------------------------------------------------------------
# Diagnostic collection
# ---------------------------------------------------------------------------

def copy_latest():
    with PRED_LOCK:
        if LATEST_PRED is None:
            return None

        out = {}

        for k, v in LATEST_PRED.items():
            if isinstance(v, np.ndarray):
                out[k] = v.copy()
            else:
                out[k] = v

        return out


def wait_for_first_prediction(
    timeout_s: float,
):
    start = time.monotonic()

    while (
        time.monotonic() - start
        < timeout_s
    ):
        if STOP_EVENT.is_set():
            break

        pred = copy_latest()

        if pred is not None:
            return pred

        time.sleep(0.05)

    return None


def collect_dwell(
    true_zone: str,
    dwell_s: float,
    occupied_classes: list[str],
):
    """
    Record only NEW inference outputs for dwell_s seconds.
    """
    rows = []

    # Avoid accidentally including the prediction that existed before ENTER.
    pred0 = copy_latest()
    last_timestamp = (
        -1.0
        if pred0 is None
        else float(pred0["timestamp"])
    )

    start = time.monotonic()
    end = start + dwell_s

    while (
        time.monotonic() < end
        and not STOP_EVENT.is_set()
    ):
        pred = copy_latest()

        if (
            pred is None
            or float(pred["timestamp"])
            == last_timestamp
        ):
            time.sleep(0.01)
            continue

        last_timestamp = float(
            pred["timestamp"]
        )

        elapsed = (
            time.monotonic() - start
        )

        row = {
            "true_zone": true_zone,
            "sample_index": len(rows) + 1,
            "elapsed_s": elapsed,
            "device_end": pred["device_end"],
            "p_empty": pred["p_empty"],
            "left_top": pred["left_top"],
            "right_top": pred["right_top"],
            "current_top": pred["current_top"],
            "equal_top": pred["equal_top"],
            "geo_top": pred["geo_top"],
            "left_entropy": pred[
                "left_entropy"
            ],
            "right_entropy": pred[
                "right_entropy"
            ],
            "left_margin": pred[
                "left_margin"
            ],
            "right_margin": pred[
                "right_margin"
            ],
            "inference_ms": pred[
                "inference_ms"
            ],
            "fresh_ms": pred[
                "fresh_ms"
            ],
        }

        for i, zone in enumerate(
            occupied_classes
        ):
            row[f"left_{zone}"] = float(
                pred["p_left"][i]
            )
            row[f"right_{zone}"] = float(
                pred["p_right"][i]
            )
            row[f"current_{zone}"] = float(
                pred["p_current"][i]
            )
            row[f"equal_{zone}"] = float(
                pred["p_equal"][i]
            )
            row[f"geo_{zone}"] = float(
                pred["p_geo"][i]
            )

        rows.append(row)

        print(
            f"  {elapsed:4.1f}s | "
            f"L={pred['left_top']:2s} "
            f"R={pred['right_top']:2s} | "
            f"CURRENT={pred['current_top']:2s} "
            f"GEO={pred['geo_top']:2s} | "
            f"Ptrue "
            f"L={row[f'left_{true_zone}']:.2f} "
            f"R={row[f'right_{true_zone}']:.2f} "
            f"F={row[f'current_{true_zone}']:.2f} "
            f"G={row[f'geo_{true_zone}']:.2f}"
        )

    return rows


def accuracy(
    rows,
    key: str,
):
    if not rows:
        return float("nan")

    return float(
        np.mean(
            [
                r[key] == r["true_zone"]
                for r in rows
            ]
        )
    )


def mean_true_prob(
    rows,
    prefix: str,
):
    if not rows:
        return float("nan")

    return float(
        np.mean(
            [
                r[
                    f"{prefix}_"
                    f"{r['true_zone']}"
                ]
                for r in rows
            ]
        )
    )


def print_zone_summary(
    zone: str,
    rows,
):
    print(
        f"\n{zone} SUMMARY "
        f"({len(rows)} windows)"
    )

    methods = [
        ("LEFT", "left_top", "left"),
        ("RIGHT", "right_top", "right"),
        (
            "CURRENT",
            "current_top",
            "current",
        ),
        ("EQUAL", "equal_top", "equal"),
        ("GEO", "geo_top", "geo"),
    ]

    for label, key, prefix in methods:
        print(
            f"  {label:7s} "
            f"top1={accuracy(rows, key):.3f} "
            f"mean P(true)="
            f"{mean_true_prob(rows, prefix):.3f}"
        )


def write_csv(
    path: Path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ground-truth-labeled "
            "TriCSI link-fusion diagnostic."
        )
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
        "--model",
        type=Path,
        default=(
            MODEL_DIR
            / "link_ensemble_rf_s4_s5_s6.joblib"
        ),
    )

    parser.add_argument(
        "--window",
        type=float,
        default=WINDOW_SECONDS,
    )

    parser.add_argument(
        "--infer-hz",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--min-packets",
        type=int,
        default=MIN_PACKETS_PER_LINK,
    )

    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--sequence",
        default="741852963",
        help=(
            "Zone digits in test order. "
            "Default 741852963."
        ),
    )

    parser.add_argument(
        "--dwell",
        type=float,
        default=5.0,
        help=(
            "Seconds recorded after the move/settle delay "
            "at each stationary zone."
        ),
    )

    parser.add_argument(
        "--settle",
        type=float,
        default=4.0,
        help=(
            "Seconds after ENTER before recording starts. "
            "Use this time to move to the requested zone and become still. "
            "Default 4 s."
        ),
    )

    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Model not found: "
            f"{args.model}"
        )

    if args.infer_hz <= 0:
        raise SystemExit(
            "--infer-hz must be > 0."
        )

    if args.dwell <= 0:
        raise SystemExit(
            "--dwell must be > 0."
        )

    if args.settle < 0:
        raise SystemExit(
            "--settle must be >= 0."
        )

    sequence = []

    for ch in args.sequence:
        if ch not in "123456789":
            raise SystemExit(
                "--sequence must contain "
                "only digits 1..9."
            )
        sequence.append(
            f"Z{ch}"
        )

    bundle = joblib.load(
        args.model
    )

    occupied_classes = [
        str(x)
        for x in bundle[
            "occupied_classes"
        ]
    ]

    print(
        "\n"
        + "=" * 78
    )
    print(
        "TriCSI V6 — LABELED "
        "LINK-FUSION DIAGNOSTIC"
    )
    print(
        "=" * 78
    )
    print(
        f"Port      : "
        f"{args.port} @ {args.baud}"
    )
    print(
        f"Model     : "
        f"{args.model}"
    )
    print(
        f"Window    : "
        f"{args.window:.2f} s"
    )
    print(
        f"Infer     : "
        f"{args.infer_hz:.1f} Hz"
    )
    print(
        f"Sequence  : "
        + " -> ".join(sequence)
    )
    print(
        f"Move/settle: "
        f"{args.settle:.1f} s before recording"
    )
    print(
        f"Dwell      : "
        f"{args.dwell:.1f} s recorded each"
    )
    print(
        "=" * 78
    )
    print(
        "Close ESP-IDF Monitor "
        "and every other program "
        "using COM11."
    )

    ser = open_serial(
        args.port,
        args.baud,
    )

    serial_thread = threading.Thread(
        target=serial_worker,
        args=(ser,),
        daemon=True,
        name="serial-drain",
    )

    infer_thread = threading.Thread(
        target=inference_worker,
        args=(
            bundle,
            args.infer_hz,
            args.window,
            args.min_packets,
        ),
        daemon=True,
        name="fusion-diagnostic",
    )

    serial_thread.start()
    infer_thread.start()

    try:
        print(
            "\nWaiting for both links "
            "and first valid prediction..."
        )

        first = wait_for_first_prediction(
            args.startup_timeout
        )

        if first is None:
            raise SystemExit(
                "Timed out waiting for "
                "a valid two-link prediction."
            )

        print(
            "Ready. "
            f"Packet rates ≈ "
            f"{recent_packet_rate('left'):.1f}/"
            f"{recent_packet_rate('right'):.1f} Hz"
        )

        print(
            "\nIMPORTANT: "
            "You do NOT need to be on the zone when you press ENTER. "
            f"After ENTER you get {args.settle:.1f} seconds to move and settle. "
            "Recording begins only after that delay."
        )

        all_rows = []
        by_zone = {}

        for i, zone in enumerate(
            sequence,
            start=1,
        ):
            print(
                "\n"
                + "-" * 78
            )
            print(
                f"[{i}/{len(sequence)}] "
                f"Move to {zone}."
            )

            input(
                f"Press ENTER, then move to {zone}. "
                f"You will have {args.settle:.1f} s before recording: "
            )

            if args.settle > 0:
                settle_end = time.monotonic() + args.settle
                last_shown = None

                while not STOP_EVENT.is_set():
                    remaining = settle_end - time.monotonic()
                    if remaining <= 0:
                        break

                    shown = int(math.ceil(remaining))
                    if shown != last_shown:
                        print(
                            f"  Move to {zone} and settle... "
                            f"{shown}"
                        )
                        last_shown = shown

                    time.sleep(0.05)

            print(
                f"  RECORDING {zone} NOW — stay still for {args.dwell:.1f} s."
            )

            rows = collect_dwell(
                zone,
                args.dwell,
                occupied_classes,
            )

            by_zone[zone] = rows
            all_rows.extend(rows)

            print_zone_summary(
                zone,
                rows,
            )

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        out_dir = (
            RESULTS_DIR
            / "live_fusion_diagnostic"
        )

        csv_path = (
            out_dir
            / f"fusion_diag_{stamp}.csv"
        )

        write_csv(
            csv_path,
            all_rows,
        )

        print(
            "\n"
            + "=" * 78
        )
        print(
            "OVERALL LABELED-DWELL SUMMARY"
        )
        print(
            "=" * 78
        )

        methods = [
            (
                "LEFT",
                "left_top",
                "left",
            ),
            (
                "RIGHT",
                "right_top",
                "right",
            ),
            (
                "CURRENT",
                "current_top",
                "current",
            ),
            (
                "EQUAL",
                "equal_top",
                "equal",
            ),
            (
                "GEO",
                "geo_top",
                "geo",
            ),
        ]

        for label, key, prefix in methods:
            print(
                f"{label:7s} "
                f"top1={accuracy(all_rows, key):.3f} "
                f"mean P(true)="
                f"{mean_true_prob(all_rows, prefix):.3f}"
            )

        print(
            "\nPER-ZONE TOP1"
        )

        header = (
            "zone   LEFT  RIGHT CURRENT "
            "EQUAL   GEO"
        )
        print(header)

        for zone in sequence:
            rows = by_zone[zone]

            print(
                f"{zone:4s} "
                f"{accuracy(rows, 'left_top'):6.3f} "
                f"{accuracy(rows, 'right_top'):6.3f} "
                f"{accuracy(rows, 'current_top'):7.3f} "
                f"{accuracy(rows, 'equal_top'):6.3f} "
                f"{accuracy(rows, 'geo_top'):6.3f}"
            )

        print(
            f"\nSaved full probabilities to:\n"
            f"{csv_path}"
        )

        print(
            "\nDo NOT retrain from this file. "
            "This is a live diagnostic used "
            "to decide whether fusion can "
            "recover the remaining zones."
        )

    except KeyboardInterrupt:
        print(
            "\nStopped by user."
        )

    finally:
        STOP_EVENT.set()

        try:
            ser.close()
        except Exception:
            pass

        serial_thread.join(
            timeout=1.0
        )
        infer_thread.join(
            timeout=1.0
        )

        print(
            "\nPackets: "
            f"L={STATS['left_packets']} "
            f"R={STATS['right_packets']} | "
            f"inferences={STATS['inferences']}"
        )

        if SERIAL_ERROR:
            print(
                f"Serial error: "
                f"{SERIAL_ERROR}"
            )

        if INFERENCE_ERROR:
            print(
                f"Inference error: "
                f"{INFERENCE_ERROR}"
            )


if __name__ == "__main__":
    main()
