from __future__ import annotations

"""
TriCSI LIVE two-link late-fusion localizer — LOW-LATENCY V2.

Why V2 exists
-------------
V1 read one serial line, then sometimes did feature extraction + 3 RF predictions
+ plotting in the SAME loop. If inference/plotting took too long, COM11 could
accumulate old CSI. The display then looked both slow and wrong because it was
classifying stale packets.

V2 separates:
    Thread 1: drain COM11 continuously in chunks
    Thread 2: run localization from the newest complete 1.0 s CSI window
    Main    : display only

The trained model is unchanged:
    models/link_ensemble_rf_s4_s5_s6.joblib

The 1.0 s input window is intentionally unchanged because that is what the
offline model was trained on. V2 removes extra EMA lag and uses only a short
3-prediction recency-weighted consensus for the displayed "stable" zone.

Run:
    python live_link_ensemble_fast_v2.py

Fast terminal-only test:
    python live_link_ensemble_fast_v2.py --no-plot
"""

import argparse
import collections
import math
import threading
import time
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

# Each packet tuple:
#   (device_time_s, rssi, iq[128], host_arrival_monotonic)
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
# Model helpers
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

    # EMPTY / OCCUPIED gate
    occupancy_model = bundle["occupancy_model"]
    occ_raw = occupancy_model.predict_proba(X_all)
    occ_lookup = {str(c): i for i, c in enumerate(occupancy_model.classes_)}
    p_empty = float(occ_raw[0, occ_lookup["EMPTY"]])

    # Separate link classifiers
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

    # Same late fusion used in the offline experiment.
    p_zone = weighted_left + weighted_right + p_stack
    p_zone /= np.sum(p_zone)

    return {
        "p_empty": p_empty,
        "p_zone": p_zone,
        "p_left": p_left,
        "p_right": p_right,
    }


def weighted_xy(
    probs: np.ndarray,
    occupied_classes: list[str],
    zone_centers: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    centers = np.asarray(
        [zone_centers[z] for z in occupied_classes],
        dtype=np.float64,
    )
    xy = probs @ centers
    return float(xy[0]), float(xy[1])


# ---------------------------------------------------------------------------
# Serial input — ALWAYS drain COM11, independent of model/plot speed
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
            ser.set_buffer_size(rx_size=262_144, tx_size=16_384)
        except (AttributeError, NotImplementedError, OSError):
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
    """
    Chunked reader based on the collector/realtime approach that already worked
    well in this project. It never waits for plotting or RF inference.
    """
    global SERIAL_ERROR

    byte_buffer = bytearray()

    try:
        while not STOP_EVENT.is_set():
            waiting = ser.in_waiting

            # Drain everything currently available. If no byte is waiting,
            # request a modest chunk and let the short serial timeout sleep us.
            chunk = ser.read(waiting if waiting > 0 else 4096)

            if not chunk:
                continue

            STATS["bytes_read"] += len(chunk)
            STATS["serial_chunks"] += 1
            byte_buffer.extend(chunk)

            while True:
                newline = byte_buffer.find(b"\n")
                if newline < 0:
                    break

                line = bytes(byte_buffer[:newline]).rstrip(b"\r")
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
                    BUFFERS[record.link].append(packet)
                    STATS[f"{record.link}_packets"] += 1

                    # Keep only a few seconds; inference always uses the newest 1 s.
                    q = BUFFERS[record.link]
                    newest = float(q[-1][0])
                    cutoff = newest - 4.0

                    while q and float(q[0][0]) < cutoff:
                        q.popleft()

            # Safety against a malformed stream with no newlines.
            if len(byte_buffer) > 1_000_000:
                byte_buffer.clear()
                STATS["bad_lines"] += 1

    except serial.SerialException as exc:
        SERIAL_ERROR = str(exc)
        STOP_EVENT.set()


# ---------------------------------------------------------------------------
# Newest complete window snapshot
# ---------------------------------------------------------------------------

def newest_window(window_seconds: float, min_packets: int):
    with BUFFER_LOCK:
        if not BUFFERS["left"] or not BUFFERS["right"]:
            return None

        # Use the newest timestamp that both links have reached.
        common_end = min(
            float(BUFFERS["left"][-1][0]),
            float(BUFFERS["right"][-1][0]),
        )
        start = common_end - window_seconds

        out = {}

        for link in ("left", "right"):
            rows = [
                x for x in BUFFERS[link]
                if start <= float(x[0]) <= common_end
            ]

            if len(rows) < min_packets:
                return None

            out[link] = {
                "time": np.asarray([x[0] for x in rows], dtype=np.float64),
                "rssi": np.asarray([x[1] for x in rows], dtype=np.float64),
                "iq": np.asarray([x[2] for x in rows], dtype=np.int8),
                "latest_host": float(rows[-1][3]),
            }

        out["common_end"] = common_end
        return out


def recent_packet_rate(link: str, span_s: float = 2.0) -> float:
    with BUFFER_LOCK:
        q = list(BUFFERS[link])

    if len(q) < 2:
        return 0.0

    end = float(q[-1][0])
    times = [float(x[0]) for x in q if float(x[0]) >= end - span_s]

    if len(times) < 2:
        return 0.0

    dt = times[-1] - times[0]
    return float((len(times) - 1) / dt) if dt > 0 else 0.0


# ---------------------------------------------------------------------------
# Inference worker
# ---------------------------------------------------------------------------

def inference_worker(
    bundle: dict,
    infer_hz: float,
    window_seconds: float,
    min_packets: int,
    consensus_n: int,
    empty_threshold: float,
) -> None:
    global LATEST_PRED, INFERENCE_ERROR

    occupied_classes = [str(x) for x in bundle["occupied_classes"]]
    zone_centers = {
        str(k): tuple(map(float, v))
        for k, v in bundle["zone_centers"].items()
    }

    zone_history = collections.deque(maxlen=consensus_n)
    empty_history = collections.deque(maxlen=consensus_n)

    period = 1.0 / infer_hz
    next_run = time.monotonic()

    try:
        while not STOP_EVENT.is_set():
            now = time.monotonic()

            if now < next_run:
                time.sleep(min(0.01, next_run - now))
                continue

            next_run = now + period

            window = newest_window(window_seconds, min_packets)
            if window is None:
                continue

            t0 = time.perf_counter()

            feature = compute_window_features(
                window["left"]["iq"],
                window["left"]["rssi"],
                window["right"]["iq"],
                window["right"]["rssi"],
            )

            pred = predict_one_window(bundle, feature)

            inference_ms = (time.perf_counter() - t0) * 1000.0
            STATS["inferences"] += 1
            STATS["last_inference_ms"] = inference_ms
            STATS["max_inference_ms"] = max(
                STATS["max_inference_ms"],
                inference_ms,
            )

            p_zone_now = np.asarray(pred["p_zone"], dtype=np.float64)
            p_empty_now = float(pred["p_empty"])

            zone_history.append(p_zone_now)
            empty_history.append(p_empty_now)

            # Very short recency-weighted consensus.
            # For 3 predictions the weights are 1,2,3, so the newest estimate
            # has half the vote. This is far less laggy than a long EMA.
            hist = list(zone_history)
            w = np.arange(1, len(hist) + 1, dtype=np.float64)
            w /= np.sum(w)

            p_zone_stable = np.sum(
                np.stack(hist, axis=0) * w[:, None],
                axis=0,
            )
            p_zone_stable /= np.sum(p_zone_stable)

            empty_hist = np.asarray(list(empty_history), dtype=np.float64)
            p_empty_stable = float(np.sum(empty_hist * w))

            now_i = int(np.argmax(p_zone_now))
            stable_i = int(np.argmax(p_zone_stable))

            now_zone = occupied_classes[now_i]
            stable_zone = occupied_classes[stable_i]

            if p_empty_now >= empty_threshold:
                now_zone = "EMPTY"

            if p_empty_stable >= empty_threshold:
                stable_zone = "EMPTY"

            # Confidence / ambiguity
            sorted_p = np.sort(p_zone_stable)
            top1 = float(sorted_p[-1])
            top2 = float(sorted_p[-2]) if len(sorted_p) >= 2 else 0.0
            margin = top1 - top2

            x_m, y_m = weighted_xy(
                p_zone_stable,
                occupied_classes,
                zone_centers,
            )

            # Freshness: time since newest accepted packet reached Python.
            newest_host = max(
                float(window["left"]["latest_host"]),
                float(window["right"]["latest_host"]),
            )
            fresh_ms = max(
                0.0,
                (time.monotonic() - newest_host) * 1000.0,
            )

            left_zone = occupied_classes[int(np.argmax(pred["p_left"]))]
            right_zone = occupied_classes[int(np.argmax(pred["p_right"]))]

            result = {
                "stable_zone": stable_zone,
                "now_zone": now_zone,
                "p_zone": p_zone_stable,
                "p_empty": p_empty_stable,
                "zone_conf": top1,
                "margin": margin,
                "x_m": x_m,
                "y_m": y_m,
                "left_zone": left_zone,
                "right_zone": right_zone,
                "inference_ms": inference_ms,
                "fresh_ms": fresh_ms,
                "timestamp": time.monotonic(),
            }

            with PRED_LOCK:
                LATEST_PRED = result

    except Exception as exc:
        INFERENCE_ERROR = repr(exc)
        STOP_EVENT.set()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def create_plot(
    occupied_classes: list[str],
    zone_centers: dict[str, tuple[float, float]],
):
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots(figsize=(8.2, 6.8))

    xs = [zone_centers[z][0] for z in occupied_classes]
    ys = [zone_centers[z][1] for z in occupied_classes]

    mx = max(0.25, 0.3 * (max(xs) - min(xs)))
    my = max(0.25, 0.3 * (max(ys) - min(ys)))

    ax.set_xlim(min(xs) - mx, max(xs) + mx)
    ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("TriCSI live localization — low latency V2")

    for z in occupied_classes:
        x, y = zone_centers[z]
        ax.scatter([x], [y], s=80)
        ax.text(x, y, z, ha="center", va="center", fontsize=9)

    dot = ax.scatter([], [], s=260, marker="x", linewidths=3)

    status = ax.text(
        0.02,
        0.98,
        "Waiting for CSI...",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.9,
        },
    )

    fig.tight_layout()
    plt.show(block=False)

    return plt, fig, dot, status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Low-latency live TriCSI late-fusion localization."
    )

    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "link_ensemble_rf_s4_s5_s6.joblib",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=WINDOW_SECONDS,
        help="Keep at the trained 1.0 s unless deliberately testing otherwise.",
    )
    parser.add_argument(
        "--infer-hz",
        type=float,
        default=5.0,
        help="Live prediction refresh rate. Default 5 Hz.",
    )
    parser.add_argument(
        "--consensus",
        type=int,
        default=3,
        help="Short recency-weighted prediction history. Default 3.",
    )
    parser.add_argument(
        "--min-packets",
        type=int,
        default=MIN_PACKETS_PER_LINK,
    )
    parser.add_argument(
        "--empty-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "Run train_link_ensemble_rf.py first."
        )

    if args.infer_hz <= 0:
        raise SystemExit("--infer-hz must be > 0.")

    if args.consensus < 1:
        raise SystemExit("--consensus must be >= 1.")

    bundle = joblib.load(args.model)

    occupied_classes = [str(x) for x in bundle["occupied_classes"]]
    zone_centers = {
        str(k): tuple(map(float, v))
        for k, v in bundle["zone_centers"].items()
    }

    empty_threshold = (
        float(args.empty_threshold)
        if args.empty_threshold is not None
        else float(bundle.get("empty_threshold", 0.5))
    )

    print("\n" + "=" * 78)
    print("TriCSI LOW-LATENCY LIVE LOCALIZER V2")
    print("=" * 78)
    print(f"Port             : {args.port} @ {args.baud}")
    print(f"Model            : {args.model}")
    print(f"CSI window       : {args.window:.2f} s  (same as training)")
    print(f"Inference refresh: {args.infer_hz:.1f} Hz")
    print(f"Consensus        : {args.consensus} predictions")
    print(f"EMPTY threshold  : {empty_threshold:.2f}")
    print("=" * 78)
    print("Close ESP-IDF Monitor and every other program using COM11.")

    ser = open_serial(args.port, args.baud)

    serial_thread = threading.Thread(
        target=serial_worker,
        args=(ser,),
        name="serial-drain",
        daemon=True,
    )
    infer_thread = threading.Thread(
        target=inference_worker,
        args=(
            bundle,
            args.infer_hz,
            args.window,
            args.min_packets,
            args.consensus,
            empty_threshold,
        ),
        name="localizer",
        daemon=True,
    )

    serial_thread.start()
    infer_thread.start()

    if args.no_plot:
        plot = None
    else:
        plot = create_plot(occupied_classes, zone_centers)

    print("\nWaiting for both links...")

    start = time.monotonic()
    last_printed_pred_time = -1.0

    try:
        while not STOP_EVENT.is_set():
            if (
                time.monotonic() - start > args.startup_timeout
                and (
                    STATS["left_packets"] < args.min_packets
                    or STATS["right_packets"] < args.min_packets
                )
            ):
                raise SystemExit(
                    "\nTimed out waiting for both links. "
                    "Check both STAs and COM11."
                )

            with PRED_LOCK:
                pred = None if LATEST_PRED is None else dict(LATEST_PRED)

            if pred is not None and pred["timestamp"] != last_printed_pred_time:
                last_printed_pred_time = pred["timestamp"]

                lrate = recent_packet_rate("left")
                rrate = recent_packet_rate("right")

                top_i = np.argsort(pred["p_zone"])[::-1][:3]
                top_text = " ".join(
                    f"{occupied_classes[i]}={pred['p_zone'][i]:.2f}"
                    for i in top_i
                )

                # Freshness is the most important V2 diagnostic.
                # <100-150 ms is healthy; seconds means something is still stale.
                print(
                    f"{pred['stable_zone']:5s} | "
                    f"NOW={pred['now_zone']:5s} | "
                    f"P={pred['zone_conf']:.2f} "
                    f"margin={pred['margin']:.2f} "
                    f"empty={pred['p_empty']:.2f} | "
                    f"xy=({pred['x_m']:.2f},{pred['y_m']:.2f}) | "
                    f"L->{pred['left_zone']} R->{pred['right_zone']} | "
                    f"{lrate:.1f}/{rrate:.1f} Hz | "
                    f"infer={pred['inference_ms']:.0f}ms "
                    f"fresh={pred['fresh_ms']:.0f}ms | "
                    f"{top_text}"
                )

                if plot is not None:
                    plt, fig, dot, status = plot

                    if pred["stable_zone"] == "EMPTY":
                        dot.set_offsets(np.empty((0, 2)))
                    elif (
                        math.isfinite(pred["x_m"])
                        and math.isfinite(pred["y_m"])
                    ):
                        dot.set_offsets(
                            np.asarray([[pred["x_m"], pred["y_m"]]])
                        )

                    status.set_text(
                        f"STABLE: {pred['stable_zone']}\n"
                        f"NOW:    {pred['now_zone']}\n"
                        f"P={pred['zone_conf']:.2f}  margin={pred['margin']:.2f}\n"
                        f"EMPTY P={pred['p_empty']:.2f}\n"
                        f"xy=({pred['x_m']:.2f}, {pred['y_m']:.2f}) m\n"
                        f"LEFT {pred['left_zone']} | RIGHT {pred['right_zone']}\n"
                        f"fresh {pred['fresh_ms']:.0f} ms | "
                        f"infer {pred['inference_ms']:.0f} ms"
                    )

                    fig.canvas.draw_idle()

            if plot is not None:
                plt, fig, dot, status = plot
                if not plt.fignum_exists(fig.number):
                    break
                plt.pause(0.01)
            else:
                time.sleep(0.02)

    except KeyboardInterrupt:
        pass

    finally:
        STOP_EVENT.set()

        try:
            ser.close()
        except Exception:
            pass

        serial_thread.join(timeout=1.0)
        infer_thread.join(timeout=1.0)

        if plot is not None:
            try:
                plot[0].close(plot[1])
            except Exception:
                pass

        print("\n" + "=" * 78)
        print("V2 stopped")
        print("=" * 78)
        print(
            f"Packets: L={STATS['left_packets']} "
            f"R={STATS['right_packets']}"
        )
        print(
            f"Inferences={STATS['inferences']} | "
            f"last={STATS['last_inference_ms']:.1f} ms | "
            f"max={STATS['max_inference_ms']:.1f} ms"
        )

        if SERIAL_ERROR:
            print(f"Serial error: {SERIAL_ERROR}")

        if INFERENCE_ERROR:
            print(f"Inference error: {INFERENCE_ERROR}")


if __name__ == "__main__":
    main()
