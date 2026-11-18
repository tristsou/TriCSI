from __future__ import annotations

"""
TriCSI LIVE two-link late-fusion localizer.

Loads the frozen development model:
    models/link_ensemble_rf_s4_s5_s6.joblib

Live pipeline:
    COM11 AP CSI stream
        -> 1.0 s rolling window
        -> existing 542 engineered features
        -> EMPTY/OCCUPIED RF
        -> LEFT RF + RIGHT RF
        -> class-specific link weights
        -> logistic stacking fusion
        -> hard zone = argmax fused P(Z)
        -> continuous (x,y) = sum P(Zi) * center(Zi)

This script does NOT retrain or tune anything.

Typical run:
    python live_link_ensemble_localize.py

Terminal only:
    python live_link_ensemble_localize.py --no-plot
"""

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import serial

from csi_io import parse_csi_line
from features import compute_window_features
from project_config import (
    HOP_SECONDS,
    LIVE_PROBABILITY_ALPHA,
    MIN_PACKETS_PER_LINK,
    MODEL_DIR,
    SERIAL_BAUD,
    SERIAL_PORT,
    WINDOW_SECONDS,
)


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

    row_sum = out.sum(axis=1, keepdims=True)

    return np.divide(
        out,
        row_sum,
        out=np.full_like(out, 1.0 / len(ordered_classes)),
        where=row_sum > 0,
    )


def weighted_xy(
    zone_prob: np.ndarray,
    occupied_classes: list[str],
    zone_centers: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    centers = np.asarray(
        [zone_centers[z] for z in occupied_classes],
        dtype=np.float64,
    )
    xy = np.asarray(zone_prob, dtype=np.float64) @ centers
    return float(xy[0]), float(xy[1])


def predict_one_window(
    bundle: dict,
    feature: dict[str, float],
) -> dict:
    all_features = list(bundle["all_features"])
    left_features = list(bundle["left_features"])
    right_features = list(bundle["right_features"])
    occupied_classes = [str(x) for x in bundle["occupied_classes"]]

    try:
        X_all = np.asarray(
            [[feature[name] for name in all_features]],
            dtype=np.float64,
        )
        X_left = np.asarray(
            [[feature[name] for name in left_features]],
            dtype=np.float64,
        )
        X_right = np.asarray(
            [[feature[name] for name in right_features]],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Feature mismatch with saved model: missing {exc}. "
            "Use the same features.py version that was used to train the ensemble."
        ) from exc

    # Stage 1: EMPTY vs OCCUPIED.
    occupancy_model = bundle["occupancy_model"]
    occupancy_raw = occupancy_model.predict_proba(X_all)
    occupancy_lookup = {
        str(c): i
        for i, c in enumerate(occupancy_model.classes_)
    }

    if "EMPTY" not in occupancy_lookup:
        raise RuntimeError("Saved occupancy model does not contain EMPTY.")

    p_empty = float(
        occupancy_raw[0, occupancy_lookup["EMPTY"]]
    )

    # Stage 2: separate spatial models.
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

    weights = np.asarray(
        bundle["weights"],
        dtype=np.float64,
    )

    if weights.shape != (2, len(occupied_classes)):
        raise RuntimeError(
            f"Unexpected saved link-weight shape {weights.shape}; "
            f"expected (2, {len(occupied_classes)})."
        )

    weighted_left = p_left * weights[0]
    weighted_right = p_right * weights[1]

    # Same stacker input used during development.
    stack_X = np.hstack(
        [
            weighted_left[None, :],
            weighted_right[None, :],
        ]
    )

    p_stack = align_proba(
        bundle["stacker"],
        stack_X,
        occupied_classes,
    )[0]

    # Same final late fusion used during development.
    p_zone = weighted_left + weighted_right + p_stack
    p_zone = p_zone / np.sum(p_zone)

    return {
        "p_empty": p_empty,
        "p_zone": p_zone,
        "p_left": p_left,
        "p_right": p_right,
    }


# ---------------------------------------------------------------------------
# Serial/window helpers
# ---------------------------------------------------------------------------

def open_serial(port: str, baud: int) -> serial.Serial:
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=0.03,
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
        except (AttributeError, NotImplementedError, OSError):
            pass

        # Opening the AP port can cause a reboot on this setup.
        # Do not assume packets will be present immediately.
        time.sleep(0.35)
        ser.reset_input_buffer()
        return ser

    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {port}: {exc}\n"
            "Close ESP-IDF Monitor, Arduino Serial Monitor, and every other "
            "program using the AP COM port."
        ) from exc


def select_window(
    buffers: dict[str, collections.deque],
    window_seconds: float,
    min_packets: int,
):
    if not buffers["left"] or not buffers["right"]:
        return None

    # Last common AP-clock time observed by BOTH streams.
    newest_time = min(
        float(buffers["left"][-1][0]),
        float(buffers["right"][-1][0]),
    )

    start_time = newest_time - window_seconds

    out = {}

    for link in ("left", "right"):
        selected = [
            row
            for row in buffers[link]
            if start_time <= row[0] <= newest_time
        ]

        if len(selected) < min_packets:
            return None

        out[link] = {
            "time": np.asarray(
                [row[0] for row in selected],
                dtype=np.float64,
            ),
            "rssi": np.asarray(
                [row[1] for row in selected],
                dtype=np.float64,
            ),
            "iq": np.asarray(
                [row[2] for row in selected],
                dtype=np.int8,
            ),
        }

    return out


def prune_buffers(
    buffers: dict[str, collections.deque],
    newest_time: float,
    keep_seconds: float,
):
    oldest_allowed = newest_time - keep_seconds

    for link in ("left", "right"):
        while (
            buffers[link]
            and buffers[link][0][0] < oldest_allowed
        ):
            buffers[link].popleft()


def packet_rate(buffer: collections.deque, span: float = 3.0) -> float:
    if len(buffer) < 2:
        return 0.0

    latest = float(buffer[-1][0])
    times = [
        float(row[0])
        for row in buffer
        if float(row[0]) >= latest - span
    ]

    if len(times) < 2:
        return 0.0

    dt = times[-1] - times[0]
    return (len(times) - 1) / dt if dt > 0 else 0.0


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def create_plot(
    occupied_classes: list[str],
    zone_centers: dict[str, tuple[float, float]],
):
    import matplotlib.pyplot as plt

    plt.ion()

    xs = [zone_centers[z][0] for z in occupied_classes]
    ys = [zone_centers[z][1] for z in occupied_classes]

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)

    margin_x = max(0.25, 0.30 * dx)
    margin_y = max(0.25, 0.30 * dy)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    ax.set_xlim(
        min(xs) - margin_x,
        max(xs) + margin_x,
    )
    ax.set_ylim(
        min(ys) - margin_y,
        max(ys) + margin_y,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("TriCSI live two-link localization")

    for zone in occupied_classes:
        x, y = zone_centers[zone]
        ax.scatter([x], [y], s=85, marker="o")
        ax.text(
            x,
            y,
            zone,
            ha="center",
            va="center",
            fontsize=9,
        )

    predicted_dot = ax.scatter(
        [],
        [],
        s=260,
        marker="x",
        linewidths=3,
    )

    status = ax.text(
        0.02,
        0.98,
        "Waiting for both links...",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.88,
        },
    )

    fig.tight_layout()
    plt.show(block=False)

    return plt, fig, ax, predicted_dot, status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live TriCSI two-link late-fusion localization."
    )

    parser.add_argument(
        "--port",
        default=SERIAL_PORT,
        help=f"AP serial port (default: {SERIAL_PORT})",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=SERIAL_BAUD,
        help=f"AP baud rate (default: {SERIAL_BAUD})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "link_ensemble_rf_s4_s5_s6.joblib",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=WINDOW_SECONDS,
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=HOP_SECONDS,
    )
    parser.add_argument(
        "--min-packets",
        type=int,
        default=MIN_PACKETS_PER_LINK,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=LIVE_PROBABILITY_ALPHA,
        help=(
            "Causal probability smoothing; larger = faster response. "
            f"Default {LIVE_PROBABILITY_ALPHA}."
        ),
    )
    parser.add_argument(
        "--empty-threshold",
        type=float,
        default=None,
        help="Override the frozen model's EMPTY threshold.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for both links after COM port opens.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "Run train_link_ensemble_rf.py first."
        )

    if not (0.0 < args.alpha <= 1.0):
        raise SystemExit("--alpha must be in (0, 1].")

    bundle = joblib.load(args.model)

    required_keys = [
        "occupancy_model",
        "left_model",
        "right_model",
        "stacker",
        "weights",
        "all_features",
        "left_features",
        "right_features",
        "occupied_classes",
        "zone_centers",
    ]

    missing = [
        k
        for k in required_keys
        if k not in bundle
    ]

    if missing:
        raise SystemExit(
            f"Saved model is missing keys: {missing}\n"
            "Re-run train_link_ensemble_rf.py."
        )

    occupied_classes = [
        str(x)
        for x in bundle["occupied_classes"]
    ]

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
    print("TriCSI LIVE TWO-LINK LATE-FUSION LOCALIZER")
    print("=" * 78)
    print(f"Model : {args.model}")
    print(f"Port  : {args.port} @ {args.baud}")
    print(f"Window: {args.window:.2f} s")
    print(f"Hop   : {args.hop:.2f} s")
    print(f"Min packets/link: {args.min_packets}")
    print(f"Smoothing alpha : {args.alpha:.2f}")
    print(f"EMPTY threshold : {empty_threshold:.2f}")
    print(f"LEFT features   : {len(bundle['left_features'])}")
    print(f"RIGHT features  : {len(bundle['right_features'])}")
    print("=" * 78)
    print(
        "Close ESP-IDF Monitor and all other programs using the AP serial port."
    )

    ser = open_serial(args.port, args.baud)

    buffers = {
        "left": collections.deque(),
        "right": collections.deque(),
    }

    packet_counts = {
        "left": 0,
        "right": 0,
    }

    smoothed_empty = None
    smoothed_zone = None

    if args.no_plot:
        plot = None
    else:
        plot = create_plot(
            occupied_classes,
            zone_centers,
        )

    print(
        "\nWaiting for both CSI links. "
        "The AP may reboot when the serial port opens..."
    )

    startup_start = time.monotonic()
    first_ready_printed = False
    last_prediction_host = 0.0
    last_wait_print = 0.0

    try:
        while True:
            raw_line = ser.readline()

            if raw_line:
                record = parse_csi_line(raw_line)

                if record is not None:
                    packet_counts[record.link] += 1

                    buffers[record.link].append(
                        (
                            float(record.device_time_s),
                            float(record.rssi),
                            record.iq.copy(),
                        )
                    )

                    prune_buffers(
                        buffers,
                        float(record.device_time_s),
                        keep_seconds=max(
                            4.0,
                            args.window + 2.0,
                        ),
                    )

            now_host = time.monotonic()

            # Startup diagnostics.
            if (
                now_host - last_wait_print >= 1.0
                and (
                    packet_counts["left"] < args.min_packets
                    or packet_counts["right"] < args.min_packets
                )
            ):
                print(
                    f"waiting... LEFT={packet_counts['left']:4d} "
                    f"RIGHT={packet_counts['right']:4d}"
                )
                last_wait_print = now_host

            if (
                now_host - startup_start > args.startup_timeout
                and (
                    packet_counts["left"] < args.min_packets
                    or packet_counts["right"] < args.min_packets
                )
            ):
                raise SystemExit(
                    "\nTimed out waiting for both transmitters.\n"
                    "Check that both STAs are connected to testopen, COM11 is the AP, "
                    "and the current MAC addresses in project_config.py are correct."
                )

            if now_host - last_prediction_host < args.hop:
                if plot is not None:
                    plot[0].pause(0.001)
                continue

            last_prediction_host = now_host

            window = select_window(
                buffers,
                args.window,
                args.min_packets,
            )

            if window is None:
                if plot is not None:
                    plot[0].pause(0.001)
                continue

            if not first_ready_printed:
                print("\nBoth links ready. Live localization started.\n")
                first_ready_printed = True

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

            current_empty = float(pred["p_empty"])
            current_zone = np.asarray(
                pred["p_zone"],
                dtype=np.float64,
            )

            if smoothed_empty is None:
                smoothed_empty = current_empty
                smoothed_zone = current_zone.copy()
            else:
                a = args.alpha
                smoothed_empty = (
                    a * current_empty
                    + (1.0 - a) * smoothed_empty
                )
                smoothed_zone = (
                    a * current_zone
                    + (1.0 - a) * smoothed_zone
                )
                smoothed_zone /= np.sum(smoothed_zone)

            best_i = int(np.argmax(smoothed_zone))
            best_zone = occupied_classes[best_i]
            zone_conf = float(smoothed_zone[best_i])

            is_empty = smoothed_empty >= empty_threshold
            shown_zone = "EMPTY" if is_empty else best_zone

            x_m, y_m = weighted_xy(
                smoothed_zone,
                occupied_classes,
                zone_centers,
            )

            # For diagnostics, show each base link's current best zone too.
            left_i = int(np.argmax(pred["p_left"]))
            right_i = int(np.argmax(pred["p_right"]))

            left_zone = occupied_classes[left_i]
            right_zone = occupied_classes[right_i]

            top_i = np.argsort(smoothed_zone)[::-1][:3]
            top_text = " ".join(
                f"{occupied_classes[i]}={smoothed_zone[i]:.2f}"
                for i in top_i
            )

            left_rate = packet_rate(buffers["left"])
            right_rate = packet_rate(buffers["right"])

            if is_empty:
                confidence_text = f"emptyP={smoothed_empty:.2f}"
            else:
                confidence_text = (
                    f"zoneP={zone_conf:.2f} emptyP={smoothed_empty:.2f}"
                )

            print(
                f"{shown_zone:5s} | {confidence_text:23s} | "
                f"xy=({x_m:4.2f},{y_m:4.2f}) m | "
                f"L->{left_zone:2s} R->{right_zone:2s} | "
                f"{left_rate:4.1f}/{right_rate:4.1f} Hz | "
                f"{top_text}"
            )

            if plot is not None:
                plt, fig, ax, predicted_dot, status = plot

                if is_empty:
                    predicted_dot.set_offsets(
                        np.empty((0, 2))
                    )
                elif math.isfinite(x_m) and math.isfinite(y_m):
                    predicted_dot.set_offsets(
                        np.asarray([[x_m, y_m]])
                    )

                status.set_text(
                    f"ZONE: {shown_zone}\n"
                    f"zone P: {zone_conf:.2f}\n"
                    f"EMPTY P: {smoothed_empty:.2f}\n"
                    f"xy: ({x_m:.2f}, {y_m:.2f}) m\n"
                    f"LEFT base:  {left_zone}\n"
                    f"RIGHT base: {right_zone}\n"
                    f"rates: L {left_rate:.1f} / R {right_rate:.1f} Hz\n"
                    f"{top_text}"
                )

                fig.canvas.draw_idle()
                plt.pause(0.001)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            ser.close()
        except Exception:
            pass

        if plot is not None:
            try:
                plot[0].close(plot[1])
            except Exception:
                pass

        print("\nLive localizer stopped.")


if __name__ == "__main__":
    main()
