from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import serial

from csi_io import parse_csi_line
from features import compute_window_features
from project_config import (
    ANCHORS,
    LIVE_MIN_CONFIDENCE,
    LIVE_PROBABILITY_ALPHA,
    MODEL_DIR,
    ROOM_HEIGHT_M,
    ROOM_WIDTH_M,
    SERIAL_BAUD,
    SERIAL_PORT,
    ZONE_CENTERS,
)


def weighted_xy(probabilities: np.ndarray, classes: np.ndarray) -> tuple[float, float]:
    indices = []
    centers = []
    for i, cls in enumerate(classes):
        if cls in ZONE_CENTERS:
            indices.append(i)
            centers.append(ZONE_CENTERS[cls])

    p = probabilities[indices].astype(np.float64)
    p_sum = float(np.sum(p))
    if p_sum <= 1e-12:
        return float("nan"), float("nan")
    p /= p_sum
    xy = p @ np.asarray(centers, dtype=np.float64)
    return float(xy[0]), float(xy[1])


def make_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, ROOM_WIDTH_M)
    ax.set_ylim(0, ROOM_HEIGHT_M)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("TriCSI live localization")

    for zone, (x, y) in ZONE_CENTERS.items():
        ax.text(x, y, zone, ha="center", va="center", alpha=0.5)

    for name, (x, y) in ANCHORS.items():
        ax.scatter([x], [y], marker="^", s=90)
        ax.text(x, y + 0.12, name, ha="center", fontsize=8)

    predicted_dot = ax.scatter([], [], s=150)
    status = ax.text(
        0.02,
        0.98,
        "Waiting for data...",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    plt.show(block=False)
    return fig, ax, predicted_dot, status


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 3x3 TriCSI fingerprint localization.")
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "localizer_rf.joblib",
    )
    parser.add_argument("--confidence", type=float, default=LIVE_MIN_CONFIDENCE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}\nRun train_localizer.py first.")

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    window_seconds = float(bundle["window_seconds"])
    min_packets = int(bundle["min_packets_per_link"])
    classes = np.asarray(model.classes_, dtype=object)

    buffers = {
        "left": collections.deque(),
        "right": collections.deque(),
    }

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.03, write_timeout=1.0)
        ser.dtr = False
        ser.rts = False
        time.sleep(0.3)
        ser.reset_input_buffer()
    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {args.port}: {exc}\n"
            "Close ESP-IDF Monitor and other serial programs."
        ) from exc

    if args.no_plot:
        plot = None
    else:
        plot = make_plot()

    smoothed_probability = None
    last_prediction_host = 0.0
    prediction_period = float(bundle.get("hop_seconds", 0.25))

    print("Live TriCSI localizer started.")
    print(f"Model : {args.model}")
    print(f"Window: {window_seconds:.2f} s")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            line = ser.readline()
            if line:
                record = parse_csi_line(line)
                if record is not None:
                    buffers[record.link].append(
                        (record.device_time_s, float(record.rssi), record.iq.copy())
                    )

                    newest = record.device_time_s
                    oldest_allowed = newest - window_seconds - 1.0
                    for link in ("left", "right"):
                        while buffers[link] and buffers[link][0][0] < oldest_allowed:
                            buffers[link].popleft()

            now_host = time.monotonic()
            if now_host - last_prediction_host < prediction_period:
                if plot is not None:
                    plt.pause(0.001)
                continue
            last_prediction_host = now_host

            if not buffers["left"] or not buffers["right"]:
                continue

            newest_time = min(buffers["left"][-1][0], buffers["right"][-1][0])
            start_time = newest_time - window_seconds

            window = {}
            enough = True

            for link in ("left", "right"):
                selected = [row for row in buffers[link] if start_time <= row[0] <= newest_time]
                if len(selected) < min_packets:
                    enough = False
                    break
                window[link] = {
                    "rssi": np.asarray([row[1] for row in selected], dtype=np.float64),
                    "iq": np.asarray([row[2] for row in selected], dtype=np.int8),
                }

            if not enough:
                continue

            feature = compute_window_features(
                window["left"]["iq"],
                window["left"]["rssi"],
                window["right"]["iq"],
                window["right"]["rssi"],
            )

            try:
                X = np.asarray(
                    [[feature[name] for name in feature_names]],
                    dtype=np.float64,
                )
            except KeyError as exc:
                raise RuntimeError(
                    f"Feature mismatch with saved model: missing {exc}. "
                    "Retrain the model using the same code version."
                ) from exc

            current_probability = model.predict_proba(X)[0]

            if smoothed_probability is None:
                smoothed_probability = current_probability
            else:
                smoothed_probability = (
                    LIVE_PROBABILITY_ALPHA * current_probability
                    + (1.0 - LIVE_PROBABILITY_ALPHA) * smoothed_probability
                )

            best_index = int(np.argmax(smoothed_probability))
            best_zone = str(classes[best_index])
            confidence = float(smoothed_probability[best_index])
            x_m, y_m = weighted_xy(smoothed_probability, classes)

            shown_zone = best_zone if confidence >= args.confidence else f"UNCERTAIN ({best_zone})"

            top_indices = np.argsort(smoothed_probability)[::-1][:3]
            top_text = "  ".join(
                f"{classes[i]}={smoothed_probability[i]:.2f}" for i in top_indices
            )

            print(
                f"{shown_zone:16s} conf={confidence:.2f} "
                f"xy=({x_m:4.2f},{y_m:4.2f})  "
                f"L={len(window['left']['iq']):2d} R={len(window['right']['iq']):2d}  "
                f"{top_text}"
            )

            if plot is not None:
                fig, ax, predicted_dot, status = plot

                if best_zone == "EMPTY" and confidence >= args.confidence:
                    predicted_dot.set_offsets(np.empty((0, 2)))
                elif np.isfinite(x_m) and np.isfinite(y_m):
                    predicted_dot.set_offsets(np.asarray([[x_m, y_m]]))

                status.set_text(
                    f"{shown_zone}\n"
                    f"confidence {confidence:.2f}\n"
                    f"estimated xy ({x_m:.2f}, {y_m:.2f})\n"
                    f"{top_text}\n"
                    f"packets L={len(window['left']['iq'])} "
                    f"R={len(window['right']['iq'])}"
                )
                fig.canvas.draw_idle()
                plt.pause(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("Live localizer stopped.")


if __name__ == "__main__":
    main()
