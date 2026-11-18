"""
Dual-link ESP32 Wi-Fi CSI motion/localization display.

Hardware mapping:
    COM9  / LEFT STA  / MAC 02:00:00:00:00:01
    COM10 / RIGHT STA / MAC 02:00:00:00:00:02
    COM11 / AP receiver / serial output

Run from the active_ap directory:
    idf.py -p COM11 monitor -b 921600 2>&1 | py -3.11 C:\\ESP32-CSI-Tool\\python_utils\\serial_plot_csi_dual_link_live.py

This script estimates the region of a MOVING person. It does not yet provide
true x-y coordinates or reliable detection of a completely motionless person.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# USER SETTINGS
# =============================================================================

LEFT_MAC = "02:00:00:00:00:01"
RIGHT_MAC = "02:00:00:00:00:02"

MAC_TO_LINK = {
    LEFT_MAC: "left",
    RIGHT_MAC: "right",
}

# Each STA is currently expected to send about 25 packets per second.
HISTORY_PACKETS = 800
CALIBRATION_PACKETS = 300

# At 25 packets/s, 15 packets is about 0.6 seconds.
# If you later raise each STA to 50 packets/s, try MOTION_WINDOW = 25 or 30.
MOTION_WINDOW = 15
SCORE_SMOOTHING_WINDOW = 5

# Hampel-style removal of isolated CSI glitches.
HAMPEL_HALF_WINDOW = 3
HAMPEL_SIGMA = 4.5

# Motion-state hysteresis.
ENTER_THRESHOLD = 6.0
EXIT_THRESHOLD = 3.0
ENTER_UPDATES = 2
EXIT_UPDATES = 4

# Slowly update the quiet model only when the link looks confidently quiet.
ADAPTIVE_QUIET_MODEL = True
QUIET_LEARNING_RATE = 0.003
ADAPT_ONLY_BELOW_SCORE = 2.0

MIN_MEDIAN_AMPLITUDE = 1.0
PLOT_Y_MAX = 15.0
PLOT_REFRESH_SECONDS = 0.20
STATUS_PRINT_SECONDS = 1.0

# Ignore unknown transmitters, including 00:00:00:00:00:00.
VALID_MACS = set(MAC_TO_LINK)


# =============================================================================
# SMALL MATH HELPERS
# =============================================================================


def robust_mad(values: np.ndarray, axis=None) -> np.ndarray:
    """Median absolute deviation, scaled like a standard deviation."""
    median = np.median(values, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(values - median), axis=axis)


def packet_shape_matrix(
    amplitudes: np.ndarray,
    carrier_indices: np.ndarray,
) -> np.ndarray:
    """
    Turn raw amplitude into a frequency-shape representation.

    Log amplitude reduces multiplicative scale effects. Subtracting each
    packet's median removes packet-wide power/AGC changes while preserving the
    relative pattern across subcarriers.
    """
    chosen = np.maximum(amplitudes[:, carrier_indices], 1e-6)
    log_amplitude = np.log(chosen)
    common_mode = np.median(log_amplitude, axis=1, keepdims=True)
    return log_amplitude - common_mode


def choose_subcarriers(calibration_amplitudes: np.ndarray) -> np.ndarray:
    """Keep usable subcarriers and remove the noisiest 20 percent."""
    median_amplitude = np.median(calibration_amplitudes, axis=0)
    amplitude_mad = robust_mad(calibration_amplitudes, axis=0)
    robust_cv = amplitude_mad / np.maximum(median_amplitude, 1e-6)

    candidates = np.where(median_amplitude > MIN_MEDIAN_AMPLITUDE)[0]

    if len(candidates) < 12:
        candidates = np.where(median_amplitude > 0.25)[0]

    if len(candidates) == 0:
        raise RuntimeError("No usable CSI subcarriers were found.")

    noise_cutoff = np.percentile(robust_cv[candidates], 80)
    stable = candidates[robust_cv[candidates] <= noise_cutoff]

    if len(stable) < min(16, len(candidates)):
        ordered = candidates[np.argsort(robust_cv[candidates])]
        stable = ordered[: min(32, len(ordered))]

    return np.sort(stable)


def correlated_hampel_filter(matrix: np.ndarray) -> np.ndarray:
    """
    Remove isolated subcarrier glitches while preserving broad changes.

    A true body movement often changes many subcarriers together. A random
    radio glitch often affects only a small fraction of them.
    """
    filtered = matrix.copy()
    half = HAMPEL_HALF_WINDOW

    if len(matrix) < 2 * half + 1:
        return filtered

    for row_index in range(half, len(matrix) - half):
        window = matrix[row_index - half : row_index + half + 1]
        local_median = np.median(window, axis=0)
        local_mad = 1.4826 * np.median(
            np.abs(window - local_median),
            axis=0,
        )

        threshold = HAMPEL_SIGMA * np.maximum(local_mad, 0.20)
        outliers = np.abs(matrix[row_index] - local_median) > threshold
        outlier_fraction = float(np.mean(outliers))

        # Replace only isolated/few-carrier spikes.
        if 0.0 < outlier_fraction < 0.25:
            filtered[row_index, outliers] = local_median[outliers]

    return filtered


def rolling_variance(matrix: np.ndarray, window: int) -> np.ndarray:
    """Efficient rolling sample variance for every selected subcarrier."""
    if len(matrix) < window:
        return np.empty((0, matrix.shape[1]), dtype=np.float64)

    cumulative = np.vstack(
        [np.zeros((1, matrix.shape[1])), np.cumsum(matrix, axis=0)]
    )
    cumulative_squared = np.vstack(
        [np.zeros((1, matrix.shape[1])), np.cumsum(matrix * matrix, axis=0)]
    )

    sums = cumulative[window:] - cumulative[:-window]
    sums_squared = cumulative_squared[window:] - cumulative_squared[:-window]

    variance = (
        sums_squared - (sums * sums) / window
    ) / max(window - 1, 1)

    return np.maximum(variance, 0.0)


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing moving average with the same output length."""
    if len(values) == 0:
        return values.copy()

    if len(values) < window:
        cumulative = np.cumsum(values, dtype=np.float64)
        return cumulative / np.arange(1, len(values) + 1)

    prefix = (
        np.cumsum(values[: window - 1], dtype=np.float64)
        / np.arange(1, window)
    )
    kernel = np.ones(window, dtype=np.float64) / window
    trailing = np.convolve(values, kernel, mode="valid")
    return np.concatenate([prefix, trailing])


# =============================================================================
# ONE INDEPENDENT DETECTOR FOR EACH RADIO LINK
# =============================================================================


@dataclass
class LinkDetector:
    name: str
    mac: str
    amplitudes: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=HISTORY_PACKETS)
    )
    timestamps: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=HISTORY_PACKETS)
    )
    rssi_values: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=HISTORY_PACKETS)
    )

    expected_subcarrier_count: Optional[int] = None
    calibrated: bool = False
    selected_subcarriers: Optional[np.ndarray] = None
    shape_center: Optional[np.ndarray] = None
    shape_scale: Optional[np.ndarray] = None
    quiet_center: Optional[float] = None
    quiet_scale: Optional[float] = None
    quiet_scale_floor: Optional[float] = None

    movement_detected: bool = False
    above_threshold_updates: int = 0
    below_threshold_updates: int = 0
    last_detection_timestamp: Optional[int] = None
    latest_score: float = 0.0

    received_since_last_print: int = 0
    total_received: int = 0

    def add_packet(
        self,
        amplitude: np.ndarray,
        timestamp: int,
        rssi: int,
    ) -> bool:
        """Add one CSI packet to this link's history."""
        if self.expected_subcarrier_count is None:
            self.expected_subcarrier_count = len(amplitude)

        if len(amplitude) != self.expected_subcarrier_count:
            return False

        self.amplitudes.append(amplitude)
        self.timestamps.append(timestamp)
        self.rssi_values.append(rssi)
        self.received_since_last_print += 1
        self.total_received += 1
        return True

    def calibration_progress(self) -> int:
        return min(len(self.amplitudes), CALIBRATION_PACKETS)

    def calibrate(self) -> None:
        if self.calibrated or len(self.amplitudes) < CALIBRATION_PACKETS:
            return

        calibration = np.asarray(
            list(self.amplitudes)[:CALIBRATION_PACKETS],
            dtype=np.float64,
        )

        self.selected_subcarriers = choose_subcarriers(calibration)

        shapes = packet_shape_matrix(
            calibration,
            self.selected_subcarriers,
        )
        self.shape_center = np.median(shapes, axis=0)
        self.shape_scale = np.maximum(
            robust_mad(shapes, axis=0),
            0.03,
        )

        quiet_metric, _ = self.calculate_motion_metric(calibration)

        if len(quiet_metric) < 10:
            raise RuntimeError(
                f"{self.name}: insufficient calibration data."
            )

        edge = min(10, len(quiet_metric) // 4)
        if len(quiet_metric) > 2 * edge:
            quiet_values = quiet_metric[edge:-edge]
        else:
            quiet_values = quiet_metric

        self.quiet_center = float(np.median(quiet_values))
        initial_mad = float(robust_mad(quiet_values))
        self.quiet_scale_floor = max(
            0.10 * max(self.quiet_center, 1e-3),
            1e-3,
        )
        self.quiet_scale = max(initial_mad, self.quiet_scale_floor)
        self.calibrated = True

        print(
            f"{self.name.upper()} calibrated | "
            f"MAC {self.mac} | "
            f"subcarriers {len(self.selected_subcarriers)} | "
            f"quiet center {self.quiet_center:.4f} | "
            f"quiet scale {self.quiet_scale:.4f}"
        )

    def calculate_motion_metric(
        self,
        amplitudes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return motion metric and matching packet indices."""
        if (
            self.selected_subcarriers is None
            or self.shape_center is None
            or self.shape_scale is None
        ):
            return np.array([]), np.array([], dtype=int)

        shapes = packet_shape_matrix(
            amplitudes,
            self.selected_subcarriers,
        )
        normalized = (
            shapes - self.shape_center
        ) / self.shape_scale
        cleaned = correlated_hampel_filter(normalized)

        variance_by_carrier = rolling_variance(
            cleaned,
            MOTION_WINDOW,
        )

        if len(variance_by_carrier) == 0:
            return np.array([]), np.array([], dtype=int)

        motion_metric = np.sqrt(
            np.median(variance_by_carrier, axis=1)
        )
        motion_metric = smooth_1d(
            motion_metric,
            SCORE_SMOOTHING_WINDOW,
        )

        # Each metric point corresponds to the last packet in its window.
        packet_indices = np.arange(
            MOTION_WINDOW - 1,
            MOTION_WINDOW - 1 + len(motion_metric),
        )
        return motion_metric, packet_indices

    def update_detection(self, latest_metric: float, timestamp: int) -> float:
        """Update movement state only once for each newest packet."""
        if (
            self.quiet_center is None
            or self.quiet_scale is None
            or self.quiet_scale_floor is None
        ):
            return 0.0

        score = max(
            0.0,
            (latest_metric - self.quiet_center)
            / max(self.quiet_scale, self.quiet_scale_floor),
        )

        # Avoid repeatedly counting the same newest sample on every redraw.
        if timestamp == self.last_detection_timestamp:
            self.latest_score = score
            return score

        self.last_detection_timestamp = timestamp

        if self.movement_detected:
            if score < EXIT_THRESHOLD:
                self.below_threshold_updates += 1
            else:
                self.below_threshold_updates = 0

            if self.below_threshold_updates >= EXIT_UPDATES:
                self.movement_detected = False
                self.below_threshold_updates = 0
                self.above_threshold_updates = 0
        else:
            if score > ENTER_THRESHOLD:
                self.above_threshold_updates += 1
            else:
                self.above_threshold_updates = 0

            if self.above_threshold_updates >= ENTER_UPDATES:
                self.movement_detected = True
                self.above_threshold_updates = 0
                self.below_threshold_updates = 0

        if (
            ADAPTIVE_QUIET_MODEL
            and not self.movement_detected
            and score < ADAPT_ONLY_BELOW_SCORE
        ):
            alpha = QUIET_LEARNING_RATE
            old_center = self.quiet_center

            self.quiet_center = (
                (1.0 - alpha) * self.quiet_center
                + alpha * latest_metric
            )

            current_deviation = 1.4826 * abs(
                latest_metric - old_center
            )
            self.quiet_scale = max(
                self.quiet_scale_floor,
                (1.0 - alpha) * self.quiet_scale
                + alpha * current_deviation,
            )

        self.latest_score = score
        return score

    def analyze(self) -> tuple[np.ndarray, np.ndarray]:
        """Return recent relative times and normalized motion scores."""
        if not self.calibrated:
            self.calibrate()

        if not self.calibrated:
            return np.array([]), np.array([])

        amplitude_array = np.asarray(
            self.amplitudes,
            dtype=np.float64,
        )
        timestamp_array = np.asarray(
            self.timestamps,
            dtype=np.int64,
        )

        metric, packet_indices = self.calculate_motion_metric(
            amplitude_array
        )

        if len(metric) == 0:
            return np.array([]), np.array([])

        metric_timestamps = timestamp_array[packet_indices]
        self.update_detection(
            float(metric[-1]),
            int(metric_timestamps[-1]),
        )

        scores = np.maximum(
            0.0,
            (metric - self.quiet_center)
            / max(self.quiet_scale, self.quiet_scale_floor),
        )

        relative_seconds = (
            metric_timestamps - metric_timestamps[-1]
        ) / 1_000_000.0

        # Use packet-window index if timestamps become malformed.
        if (
            not np.all(np.isfinite(relative_seconds))
            or np.any(np.diff(relative_seconds) < 0)
            or abs(relative_seconds[0]) > 120
        ):
            relative_seconds = np.arange(
                -len(scores) + 1,
                1,
                dtype=np.float64,
            )

        return relative_seconds, scores

    def packet_rate(self) -> float:
        times = np.asarray(self.timestamps, dtype=np.float64)
        if len(times) < 20:
            return 0.0

        count = min(100, len(times) - 1)
        elapsed = (times[-1] - times[-count - 1]) / 1_000_000.0
        return count / elapsed if elapsed > 0 else 0.0

    def latest_rssi(self) -> Optional[int]:
        return int(self.rssi_values[-1]) if self.rssi_values else None


LINKS = {
    "left": LinkDetector("left", LEFT_MAC),
    "right": LinkDetector("right", RIGHT_MAC),
}


# =============================================================================
# CSI TEXT PARSER
# =============================================================================

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def parse_csi_line(line: str) -> bool:
    """Parse and route one AP CSI line to the proper link."""
    try:
        line = ANSI_ESCAPE.sub("", line)
        start = line.find("CSI_DATA,AP,")

        if start < 0:
            return False

        line = line[start:]
        fields = line.split(",", 25)

        if len(fields) < 26:
            return False

        mac = fields[2].strip().upper()

        if mac not in VALID_MACS:
            return False

        rssi = int(fields[3].strip())
        local_timestamp = int(fields[18].strip())

        payload = fields[25]
        left_bracket = payload.find("[")
        right_bracket = payload.rfind("]")

        if left_bracket < 0 or right_bracket <= left_bracket:
            return False

        raw_values = [
            int(value)
            for value in payload[left_bracket + 1 : right_bracket].split()
            if value
        ]

        if len(raw_values) < 4:
            return False

        if len(raw_values) % 2 != 0:
            raw_values = raw_values[:-1]

        # ESP32 raw CSI order is imaginary, real, imaginary, real, ...
        imaginary = np.asarray(raw_values[0::2], dtype=np.float64)
        real = np.asarray(raw_values[1::2], dtype=np.float64)
        amplitude = np.sqrt(imaginary * imaginary + real * real)

        link_name = MAC_TO_LINK[mac]
        return LINKS[link_name].add_packet(
            amplitude=amplitude,
            timestamp=local_timestamp,
            rssi=rssi,
        )

    except (ValueError, IndexError, OverflowError):
        return False


# =============================================================================
# SIMPLE MOVING-PERSON REGION ESTIMATE
# =============================================================================


def predicted_region() -> str:
    left_active = LINKS["left"].movement_detected
    right_active = LINKS["right"].movement_detected

    if not left_active and not right_active:
        return "QUIET / NO MOVEMENT"
    if left_active and not right_active:
        return "MOVEMENT NEAR LEFT LINK"
    if right_active and not left_active:
        return "MOVEMENT NEAR RIGHT LINK"
    return "MOVEMENT AFFECTING BOTH LINKS / CENTER"


# =============================================================================
# LIVE GRAPH
# =============================================================================

plt.ion()
fig, axis = plt.subplots(figsize=(12, 7))
fig.canvas.manager.set_window_title("Dual-Link ESP32 CSI Motion Locator")
plt.show(block=False)


def draw_calibration_screen() -> None:
    axis.clear()

    left_progress = LINKS["left"].calibration_progress()
    right_progress = LINKS["right"].calibration_progress()

    axis.text(
        0.5,
        0.68,
        "CALIBRATING BOTH CSI LINKS",
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=19,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.49,
        f"LEFT  {left_progress}/{CALIBRATION_PACKETS} packets\n"
        f"RIGHT {right_progress}/{CALIBRATION_PACKETS} packets",
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=15,
        family="monospace",
    )
    axis.text(
        0.5,
        0.30,
        "Keep the room, person, boards, antennas, and USB cables still.",
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=12,
    )

    axis.set_xlim(0, CALIBRATION_PACKETS)
    axis.set_ylim(0, PLOT_Y_MAX)
    axis.set_xlabel("Calibration packets")
    axis.set_ylabel("Motion score")
    axis.grid(True, alpha=0.20)
    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def draw_live_graph() -> None:
    left_times, left_scores = LINKS["left"].analyze()
    right_times, right_scores = LINKS["right"].analyze()

    if not LINKS["left"].calibrated or not LINKS["right"].calibrated:
        draw_calibration_screen()
        return

    axis.clear()

    if len(left_scores):
        axis.plot(
            left_times,
            np.clip(left_scores, 0, PLOT_Y_MAX),
            linewidth=2.3,
            label=(
                f"LEFT link — {LINKS['left'].latest_score:.1f}"
            ),
        )

    if len(right_scores):
        axis.plot(
            right_times,
            np.clip(right_scores, 0, PLOT_Y_MAX),
            linewidth=2.3,
            label=(
                f"RIGHT link — {LINKS['right'].latest_score:.1f}"
            ),
        )

    axis.axhspan(
        0,
        EXIT_THRESHOLD,
        alpha=0.08,
        label="Quiet region",
    )
    axis.axhline(
        ENTER_THRESHOLD,
        linestyle="--",
        linewidth=1.8,
        label=f"Movement threshold ({ENTER_THRESHOLD:.0f})",
    )
    axis.axhline(
        EXIT_THRESHOLD,
        linestyle=":",
        linewidth=1.3,
        label=f"Release threshold ({EXIT_THRESHOLD:.0f})",
    )

    all_times = []
    if len(left_times):
        all_times.extend([left_times[0], left_times[-1]])
    if len(right_times):
        all_times.extend([right_times[0], right_times[-1]])

    if all_times:
        axis.set_xlim(min(all_times), max(all_times))

    axis.set_ylim(0, PLOT_Y_MAX)
    axis.set_yticks([0, 1, 2, 3, 6, 9, 12, 15])
    axis.set_yticklabels(["0", "1", "2", "3", "6", "9", "12", "15+"])
    axis.set_xlabel("Time relative to now (seconds)")
    axis.set_ylabel("Motion score above each link's quiet noise")

    region = predicted_region()
    axis.set_title(
        f"{region}\n"
        f"LEFT {LINKS['left'].latest_score:.1f} | "
        f"RIGHT {LINKS['right'].latest_score:.1f}",
        fontsize=15,
        fontweight="bold",
    )

    left_rssi = LINKS["left"].latest_rssi()
    right_rssi = LINKS["right"].latest_rssi()

    info = (
        f"LEFT  MAC {LEFT_MAC}\n"
        f"      {LINKS['left'].packet_rate():.1f} pkt/s, "
        f"RSSI {left_rssi if left_rssi is not None else '—'} dBm\n"
        f"RIGHT MAC {RIGHT_MAC}\n"
        f"      {LINKS['right'].packet_rate():.1f} pkt/s, "
        f"RSSI {right_rssi if right_rssi is not None else '—'} dBm"
    )

    axis.text(
        0.01,
        0.98,
        info,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.78,
        },
    )

    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# =============================================================================
# MAIN LOOP
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read dual-link ESP32 CSI directly from a serial port."
    )
    parser.add_argument(
        "--port",
        default="COM11",
        help="AP serial port. Default: COM11",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
        help="AP runtime serial baud rate. Default: 921600",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    try:
        import serial
    except ImportError:
        print(
            "Missing pyserial. Install it with:\n"
            "  py -3.11 -m pip install pyserial",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("Dual-link CSI detector started.")
    print(f"Serial port: {args.port}")
    print(f"Baud rate:   {args.baud}")
    print(f"LEFT  MAC: {LEFT_MAC}")
    print(f"RIGHT MAC: {RIGHT_MAC}")
    print("Opening the AP serial port directly...")

    last_plot_time = 0.0
    last_status_time = 0.0

    try:
        with serial.Serial(
            port=args.port,
            baudrate=args.baud,
            timeout=0.10,
            write_timeout=1.0,
        ) as serial_port:
            # Avoid intentionally forcing bootloader mode.
            serial_port.dtr = False
            serial_port.rts = False
            serial_port.reset_input_buffer()

            print("Serial port opened. Waiting for CSI_DATA,AP lines...")
            print("Keep the room still while both links calibrate.")

            while plt.fignum_exists(fig.number):
                raw_line = serial_port.readline()
                incoming_line = raw_line.decode("utf-8", errors="replace")

                if incoming_line:
                    parse_csi_line(incoming_line)

                now = time.monotonic()

                # Keep the Matplotlib window responsive even when no packet
                # arrives during this short serial timeout.
                if now - last_plot_time >= PLOT_REFRESH_SECONDS:
                    draw_live_graph()
                    plt.pause(0.001)
                    last_plot_time = now
                else:
                    plt.pause(0.001)

                if now - last_status_time >= STATUS_PRINT_SECONDS:
                    left = LINKS["left"]
                    right = LINKS["right"]
                    print(
                        f"LEFT {left.received_since_last_print:3d} pkt/s "
                        f"score={left.latest_score:5.2f} | "
                        f"RIGHT {right.received_since_last_print:3d} pkt/s "
                        f"score={right.latest_score:5.2f} | "
                        f"{predicted_region()}"
                    )
                    left.received_since_last_print = 0
                    right.received_since_last_print = 0
                    last_status_time = now

    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        print(
            "Close ESP-IDF Monitor, Arduino Serial Monitor, the old graph, "
            "and any other program using the AP COM port.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("Dual-link CSI detector stopped.")

if __name__ == "__main__":
    main()
