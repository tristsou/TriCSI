"""
Smooth, low-lag dual-link ESP32 Wi-Fi CSI movement locator.

This version keeps the successful signal-processing idea from the original
single-link movement detector, but runs it independently for the left and
right transmitter links. It is optimized for live use:

* serial reading and CSI processing run in a background thread;
* only the newest packet/window is processed, rather than recomputing all
  800 packets for both links every graph refresh;
* the graph uses display-only resampling and smoothing without changing
  the movement detector or location decision;
* Matplotlib blitting redraws only the changing artists when supported;
* location uses relative left-vs-right score dominance, not only two hard
  yes/no movement flags.

Hardware mapping:
    LEFT STA  / COM9  / MAC 02:00:00:00:00:01
    RIGHT STA / COM10 / MAC 02:00:00:00:00:02
    AP receiver / COM11 / 921600 baud

Install:
    py -3.14 -m pip install --upgrade numpy matplotlib pyserial

Run:
    py -3.14 serial_csi_dual_link_smooth.py --port COM11 --baud 921600

The program locates MOVEMENT, not a completely motionless person.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, Optional

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is missing. Install it with:\n"
        "  py -3.14 -m pip install --upgrade numpy"
    ) from exc

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Matplotlib/Tk is unavailable. Install Matplotlib with:\n"
        "  py -3.14 -m pip install --upgrade matplotlib"
    ) from exc

try:
    import serial
except ImportError as exc:
    raise SystemExit(
        "pySerial is missing. Install it with:\n"
        "  py -3.14 -m pip install --upgrade pyserial"
    ) from exc


# =============================================================================
# HARDWARE AND DETECTOR SETTINGS
# =============================================================================

LEFT_MAC = "02:00:00:00:00:01"
RIGHT_MAC = "02:00:00:00:00:02"

MAC_TO_LINK = {
    LEFT_MAC: "left",
    RIGHT_MAC: "right",
}

# These match the successful original movement detector.
CALIBRATION_PACKETS = 300
MOTION_WINDOW = 30
SCORE_SMOOTHING_WINDOW = 5
HAMPEL_HALF_WINDOW = 3
HAMPEL_SIGMA = 4.5

ENTER_THRESHOLD = 6.0
EXIT_THRESHOLD = 3.0
ENTER_UPDATES = 2
EXIT_UPDATES = 4

ADAPTIVE_QUIET_MODEL = True
QUIET_LEARNING_RATE = 0.003
ADAPT_ONLY_BELOW_SCORE = 2.0

MIN_MEDIAN_AMPLITUDE = 1.0

# Location is allowed to use a slightly weaker but clearly dominant response.
# This helps when one physical radio path is naturally less sensitive.
LOCATION_MIN_SCORE = 3.5
DOMINANCE_RATIO = 1.35
LOCATION_HOLD_SECONDS = 0.45

# Start both at 1.0. Only change RIGHT_SCORE_GAIN after checking that the right
# link receives packets normally but its score is consistently too small.
LEFT_SCORE_GAIN = 1.0
RIGHT_SCORE_GAIN = 1.0

SCORE_HISTORY_POINTS = 500
DISPLAY_SECONDS = 25.0
PLOT_Y_MAX = 15.0

# Display-only settings. These do not affect movement detection.
DISPLAY_RESAMPLE_POINTS = 220
DISPLAY_SMOOTHING_WINDOW = 9  # odd number; 7-13 is a useful range
PLOT_REFRESH_SECONDS = 0.07   # about 14 frames per second
STATUS_PRINT_SECONDS = 2.0

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


# =============================================================================
# SIGNAL-PROCESSING HELPERS
# =============================================================================


def robust_mad(values: np.ndarray, axis=None) -> np.ndarray:
    """Median absolute deviation, scaled like a standard deviation."""
    median = np.median(values, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(values - median), axis=axis)


def packet_shape_matrix(
    amplitudes: np.ndarray,
    carrier_indices: np.ndarray,
) -> np.ndarray:
    """Log amplitude with packet-wide common gain removed."""
    chosen = np.maximum(amplitudes[:, carrier_indices], 1e-6)
    log_amplitude = np.log(chosen)
    return log_amplitude - np.median(log_amplitude, axis=1, keepdims=True)


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

    cutoff = np.percentile(robust_cv[candidates], 80)
    stable = candidates[robust_cv[candidates] <= cutoff]

    if len(stable) < min(16, len(candidates)):
        ordered = candidates[np.argsort(robust_cv[candidates])]
        stable = ordered[: min(32, len(ordered))]

    return np.sort(stable)


def correlated_hampel_filter(matrix: np.ndarray) -> np.ndarray:
    """Original detector's offline Hampel-style glitch suppression."""
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
        fraction = float(np.mean(outliers))

        # Suppress isolated radio glitches, but keep coordinated movement.
        if 0.0 < fraction < 0.25:
            filtered[row_index, outliers] = local_median[outliers]

    return filtered


def rolling_variance(matrix: np.ndarray, window: int) -> np.ndarray:
    """Rolling sample variance for each selected subcarrier."""
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
        sums_squared - sums * sums / window
    ) / max(window - 1, 1)
    return np.maximum(variance, 0.0)


def trailing_average(values: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average with one result per input value."""
    if len(values) == 0:
        return values.copy()

    output = np.empty(len(values), dtype=np.float64)
    cumulative = np.cumsum(values, dtype=np.float64)

    for index in range(len(values)):
        start = max(0, index - window + 1)
        total = cumulative[index] - (cumulative[start - 1] if start else 0.0)
        output[index] = total / (index - start + 1)

    return output


# =============================================================================
# ONE INCREMENTAL COPY OF THE ORIGINAL DETECTOR PER LINK
# =============================================================================


@dataclass
class LinkDetector:
    name: str
    mac: str
    score_gain: float = 1.0

    expected_subcarrier_count: Optional[int] = None
    calibration_amplitudes: list[np.ndarray] = field(default_factory=list)

    calibrated: bool = False
    selected_subcarriers: Optional[np.ndarray] = None
    shape_center: Optional[np.ndarray] = None
    shape_scale: Optional[np.ndarray] = None
    quiet_center: float = 0.0
    quiet_scale: float = 1.0
    quiet_scale_floor: float = 1e-3

    # Only the newest normalized packets are needed for live scoring.
    normalized_window: Deque[np.ndarray] = field(
        default_factory=lambda: collections.deque(maxlen=MOTION_WINDOW)
    )
    hampel_history: Deque[np.ndarray] = field(
        default_factory=lambda: collections.deque(
            maxlen=2 * HAMPEL_HALF_WINDOW + 1
        )
    )
    metric_history: Deque[float] = field(
        default_factory=lambda: collections.deque(
            maxlen=SCORE_SMOOTHING_WINDOW
        )
    )

    score_times: Deque[float] = field(
        default_factory=lambda: collections.deque(maxlen=SCORE_HISTORY_POINTS)
    )
    score_history: Deque[float] = field(
        default_factory=lambda: collections.deque(maxlen=SCORE_HISTORY_POINTS)
    )
    packet_arrival_times: Deque[float] = field(
        default_factory=lambda: collections.deque(maxlen=250)
    )

    latest_raw_score: float = 0.0
    latest_score: float = 0.0
    movement_detected: bool = False
    above_threshold_updates: int = 0
    below_threshold_updates: int = 0
    total_received: int = 0
    rejected_length: int = 0
    latest_rssi: Optional[int] = None

    def add_packet(
        self,
        amplitude: np.ndarray,
        rssi: int,
        arrival_time: float,
    ) -> None:
        """Process exactly one new CSI packet."""
        if self.expected_subcarrier_count is None:
            self.expected_subcarrier_count = len(amplitude)

        if len(amplitude) != self.expected_subcarrier_count:
            self.rejected_length += 1
            return

        self.total_received += 1
        self.latest_rssi = rssi
        self.packet_arrival_times.append(arrival_time)

        if not self.calibrated:
            self.calibration_amplitudes.append(amplitude)
            if len(self.calibration_amplitudes) >= CALIBRATION_PACKETS:
                self.calibrate()
            return

        self.process_live_packet(amplitude, arrival_time)

    def calibration_progress(self) -> int:
        return min(len(self.calibration_amplitudes), CALIBRATION_PACKETS)

    def calibrate(self) -> None:
        """Build the independent quiet model using the original pipeline."""
        calibration = np.asarray(
            self.calibration_amplitudes[:CALIBRATION_PACKETS],
            dtype=np.float64,
        )

        self.selected_subcarriers = choose_subcarriers(calibration)
        shapes = packet_shape_matrix(calibration, self.selected_subcarriers)

        self.shape_center = np.median(shapes, axis=0)
        self.shape_scale = np.maximum(robust_mad(shapes, axis=0), 0.03)

        normalized = (shapes - self.shape_center) / self.shape_scale
        cleaned = correlated_hampel_filter(normalized)
        variance_by_carrier = rolling_variance(cleaned, MOTION_WINDOW)

        if len(variance_by_carrier) < 10:
            raise RuntimeError(f"{self.name}: not enough quiet calibration data")

        raw_metric = np.sqrt(np.median(variance_by_carrier, axis=1))
        quiet_metric = trailing_average(raw_metric, SCORE_SMOOTHING_WINDOW)

        edge = min(10, len(quiet_metric) // 4)
        quiet_values = (
            quiet_metric[edge:-edge]
            if len(quiet_metric) > 2 * edge
            else quiet_metric
        )

        self.quiet_center = float(np.median(quiet_values))
        initial_mad = float(robust_mad(quiet_values))
        self.quiet_scale_floor = max(
            0.10 * max(self.quiet_center, 1e-3),
            1e-3,
        )
        self.quiet_scale = max(initial_mad, self.quiet_scale_floor)

        # Seed the live rolling windows with the end of quiet calibration.
        for row in cleaned[-MOTION_WINDOW:]:
            self.normalized_window.append(row.copy())
            self.hampel_history.append(row.copy())
        for value in raw_metric[-SCORE_SMOOTHING_WINDOW:]:
            self.metric_history.append(float(value))

        self.calibrated = True
        self.calibration_amplitudes.clear()

        print(
            f"{self.name.upper()} calibrated | MAC {self.mac} | "
            f"subcarriers={len(self.selected_subcarriers)} | "
            f"quiet center={self.quiet_center:.4f} | "
            f"quiet scale={self.quiet_scale:.4f}"
        )

    def causal_hampel(self, normalized_row: np.ndarray) -> np.ndarray:
        """Low-cost live version of the original glitch filter."""
        self.hampel_history.append(normalized_row.copy())

        if len(self.hampel_history) < self.hampel_history.maxlen:
            return normalized_row

        history = np.asarray(self.hampel_history, dtype=np.float64)
        local_median = np.median(history, axis=0)
        local_mad = 1.4826 * np.median(
            np.abs(history - local_median),
            axis=0,
        )
        threshold = HAMPEL_SIGMA * np.maximum(local_mad, 0.20)
        outliers = np.abs(normalized_row - local_median) > threshold
        fraction = float(np.mean(outliers))

        cleaned = normalized_row.copy()
        if 0.0 < fraction < 0.25:
            cleaned[outliers] = local_median[outliers]
        return cleaned

    def process_live_packet(
        self,
        amplitude: np.ndarray,
        arrival_time: float,
    ) -> None:
        assert self.selected_subcarriers is not None
        assert self.shape_center is not None
        assert self.shape_scale is not None

        shape = packet_shape_matrix(
            amplitude[np.newaxis, :],
            self.selected_subcarriers,
        )[0]
        normalized = (shape - self.shape_center) / self.shape_scale
        cleaned = self.causal_hampel(normalized)
        self.normalized_window.append(cleaned)

        if len(self.normalized_window) < MOTION_WINDOW:
            return

        window = np.asarray(self.normalized_window, dtype=np.float64)
        variance_by_carrier = np.var(window, axis=0, ddof=1)
        raw_metric = float(np.sqrt(np.median(variance_by_carrier)))

        self.metric_history.append(raw_metric)
        smoothed_metric = float(np.mean(self.metric_history))

        raw_score = max(
            0.0,
            (smoothed_metric - self.quiet_center)
            / max(self.quiet_scale, self.quiet_scale_floor),
        )
        adjusted_score = raw_score * self.score_gain

        self.latest_raw_score = raw_score
        self.latest_score = adjusted_score
        self.update_detection(adjusted_score, smoothed_metric)

        self.score_times.append(arrival_time)
        self.score_history.append(adjusted_score)

    def update_detection(
        self,
        score: float,
        latest_metric: float,
    ) -> None:
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
            deviation = 1.4826 * abs(latest_metric - old_center)
            self.quiet_scale = max(
                self.quiet_scale_floor,
                (1.0 - alpha) * self.quiet_scale + alpha * deviation,
            )

    def packet_rate(self) -> float:
        if len(self.packet_arrival_times) < 2:
            return 0.0
        elapsed = self.packet_arrival_times[-1] - self.packet_arrival_times[0]
        return (
            (len(self.packet_arrival_times) - 1) / elapsed
            if elapsed > 0
            else 0.0
        )

    def snapshot(self, now: float) -> dict:
        times = np.asarray(self.score_times, dtype=np.float64)
        scores = np.asarray(self.score_history, dtype=np.float64)
        relative_times = times - now if len(times) else times
        return {
            "name": self.name,
            "calibrated": self.calibrated,
            "calibration_progress": self.calibration_progress(),
            "times": relative_times,
            "scores": scores,
            "latest_score": self.latest_score,
            "latest_raw_score": self.latest_raw_score,
            "movement": self.movement_detected,
            "packet_rate": self.packet_rate(),
            "rssi": self.latest_rssi,
            "total": self.total_received,
            "rejected_length": self.rejected_length,
        }


LINKS = {
    "left": LinkDetector("left", LEFT_MAC, LEFT_SCORE_GAIN),
    "right": LinkDetector("right", RIGHT_MAC, RIGHT_SCORE_GAIN),
}

STATE_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
SERIAL_ERROR: Optional[str] = None


# =============================================================================
# CSI PARSER AND SERIAL READER
# =============================================================================


def parse_csi_line(line: str, arrival_time: float) -> bool:
    """Parse one AP CSI line and route it by transmitter MAC."""
    try:
        line = ANSI_ESCAPE.sub("", line)
        start = line.find("CSI_DATA,AP,")
        if start < 0:
            return False

        fields = line[start:].split(",", 25)
        if len(fields) < 26:
            return False

        mac = fields[2].strip().upper()
        link_name = MAC_TO_LINK.get(mac)
        if link_name is None:
            return False

        rssi = int(fields[3].strip())
        payload = fields[25]
        left_bracket = payload.find("[")
        right_bracket = payload.rfind("]")
        if left_bracket < 0 or right_bracket <= left_bracket:
            return False

        raw_values = np.fromstring(
            payload[left_bracket + 1 : right_bracket],
            dtype=np.float64,
            sep=" ",
        )
        if len(raw_values) < 4:
            return False
        if len(raw_values) % 2:
            raw_values = raw_values[:-1]

        imaginary = raw_values[0::2]
        real = raw_values[1::2]
        amplitude = np.hypot(imaginary, real)

        with STATE_LOCK:
            LINKS[link_name].add_packet(amplitude, rssi, arrival_time)
        return True

    except (ValueError, IndexError, OverflowError, RuntimeError) as exc:
        # Keep the live reader running; the periodic status line will reveal
        # missing packets or calibration problems.
        print(f"Skipped malformed CSI packet: {exc}", file=sys.stderr)
        return False


def serial_reader(port: str, baud: int) -> None:
    global SERIAL_ERROR

    try:
        with serial.Serial(
            port=port,
            baudrate=baud,
            timeout=0.05,
            write_timeout=1.0,
        ) as serial_port:
            serial_port.dtr = False
            serial_port.rts = False
            serial_port.reset_input_buffer()

            print(f"Serial port {port} opened at {baud} baud.")
            print("Keep the room still until both links finish calibration.")

            while not STOP_EVENT.is_set():
                raw_line = serial_port.readline()
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                parse_csi_line(line, time.monotonic())

    except serial.SerialException as exc:
        SERIAL_ERROR = str(exc)
        STOP_EVENT.set()


# =============================================================================
# LOCATION LOGIC
# =============================================================================


@dataclass
class LocationState:
    label: str = "CALIBRATING"
    candidate: str = "CALIBRATING"
    candidate_since: float = field(default_factory=time.monotonic)

    def update(self, left: dict, right: dict, now: float) -> str:
        if not left["calibrated"] or not right["calibrated"]:
            proposed = "CALIBRATING"
        else:
            left_score = float(left["latest_score"])
            right_score = float(right["latest_score"])
            strongest = max(left_score, right_score)

            # Preserve the original detector's high-confidence movement state,
            # but also permit a clearly dominant weaker link to identify side.
            active = (
                bool(left["movement"])
                or bool(right["movement"])
                or strongest >= LOCATION_MIN_SCORE
            )

            if not active:
                proposed = "QUIET / NO MOVEMENT"
            elif left_score >= max(
                LOCATION_MIN_SCORE,
                right_score * DOMINANCE_RATIO,
            ):
                proposed = "MOVEMENT NEAR LEFT LINK"
            elif right_score >= max(
                LOCATION_MIN_SCORE,
                left_score * DOMINANCE_RATIO,
            ):
                proposed = "MOVEMENT NEAR RIGHT LINK"
            else:
                proposed = "MOVEMENT AFFECTING BOTH LINKS / CENTER"

        if proposed != self.candidate:
            self.candidate = proposed
            self.candidate_since = now

        if (
            self.label == "CALIBRATING"
            or now - self.candidate_since >= LOCATION_HOLD_SECONDS
        ):
            self.label = self.candidate

        return self.label


LOCATION_STATE = LocationState()


# =============================================================================
# DISPLAY-ONLY SMOOTHING
# =============================================================================


def triangular_kernel(window: int) -> np.ndarray:
    """Return a normalized triangular smoothing kernel."""
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return np.ones(1, dtype=np.float64)

    half = window // 2
    rising = np.arange(1, half + 2, dtype=np.float64)
    kernel = np.concatenate([rising, rising[-2::-1]])
    return kernel / np.sum(kernel)


DISPLAY_KERNEL = triangular_kernel(DISPLAY_SMOOTHING_WINDOW)


def smooth_series_for_display(
    times: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample and smooth one plotted line.

    This changes only what the graph looks like. The detector, thresholds,
    title scores, and location decision still use the original live scores.
    """
    if len(times) == 0 or len(scores) == 0:
        return times, scores

    usable = (
        np.isfinite(times)
        & np.isfinite(scores)
        & (times >= -DISPLAY_SECONDS)
        & (times <= 0.5)
    )
    x = np.asarray(times[usable], dtype=np.float64)
    y = np.asarray(scores[usable], dtype=np.float64)

    if len(x) < 2:
        return x, np.clip(y, 0.0, PLOT_Y_MAX)

    # Remove duplicate/non-increasing timestamps before interpolation.
    keep = np.concatenate(([True], np.diff(x) > 1e-6))
    x = x[keep]
    y = y[keep]
    if len(x) < 2:
        return x, np.clip(y, 0.0, PLOT_Y_MAX)

    point_count = min(
        DISPLAY_RESAMPLE_POINTS,
        max(len(x), 60),
    )
    display_x = np.linspace(x[0], x[-1], point_count)
    display_y = np.interp(display_x, x, y)

    if len(display_y) >= len(DISPLAY_KERNEL):
        pad = len(DISPLAY_KERNEL) // 2
        padded = np.pad(display_y, (pad, pad), mode="edge")
        display_y = np.convolve(
            padded,
            DISPLAY_KERNEL,
            mode="valid",
        )

    return display_x, np.clip(display_y, 0.0, PLOT_Y_MAX)


DISPLAY_LOCATION_LABELS = {
    "CALIBRATING": "CALIBRATING — KEEP THE ROOM STILL",
    "QUIET / NO MOVEMENT": "QUIET / NO MOVEMENT",
    "MOVEMENT NEAR LEFT LINK": "MOVEMENT NEAR LEFT LINK",
    "MOVEMENT NEAR RIGHT LINK": "MOVEMENT NEAR RIGHT LINK",
    "MOVEMENT AFFECTING BOTH LINKS / CENTER": "MOVEMENT IN CENTER / BOTH LINKS",
}


# =============================================================================
# LOW-OVERHEAD LIVE GRAPH
# =============================================================================


def create_plot():
    plt.ion()
    figure, axis = plt.subplots(
        figsize=(12.5, 7.5),
        constrained_layout=True,
    )
    figure.canvas.manager.set_window_title(
        "Smooth Dual-Link ESP32 CSI Locator"
    )

    (left_line,) = axis.plot(
        [],
        [],
        linewidth=2.4,
        label="LEFT link",
        antialiased=True,
        solid_capstyle="round",
        solid_joinstyle="round",
        animated=True,
    )
    (right_line,) = axis.plot(
        [],
        [],
        linewidth=2.4,
        label="RIGHT link",
        antialiased=True,
        solid_capstyle="round",
        solid_joinstyle="round",
        animated=True,
    )

    axis.axhspan(0, EXIT_THRESHOLD, alpha=0.08, label="Quiet region")
    axis.axhline(
        ENTER_THRESHOLD,
        linestyle="--",
        linewidth=1.5,
        label=f"Movement threshold ({ENTER_THRESHOLD:.0f})",
    )
    axis.axhline(
        EXIT_THRESHOLD,
        linestyle=":",
        linewidth=1.2,
        label=f"Release threshold ({EXIT_THRESHOLD:.0f})",
    )

    axis.set_xlim(-DISPLAY_SECONDS, 0)
    axis.set_ylim(0, PLOT_Y_MAX)
    axis.set_xlabel("Time relative to now (seconds)")
    axis.set_ylabel("Motion score above each link's quiet noise")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)

    # Use figure-level text instead of a long axes title. Constrained layout
    # reserves room for it, so it stays visible when the window is resized.
    location_text = figure.suptitle(
        "CALIBRATING — KEEP THE ROOM STILL",
        fontsize=15,
        fontweight="bold",
        y=0.985,
        wrap=True,
    )
    location_text.set_animated(True)

    score_text = figure.text(
        0.5,
        0.935,
        "LEFT 0.0   |   RIGHT 0.0",
        ha="center",
        va="top",
        fontsize=11,
        animated=True,
    )

    status_text = axis.text(
        0.012,
        0.985,
        "Starting...",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        animated=True,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.82,
        },
    )

    blit_state = {
        "background": None,
        "needs_background": True,
        "supports_blit": bool(figure.canvas.supports_blit),
    }

    def mark_background_dirty(_event=None):
        blit_state["needs_background"] = True

    figure.canvas.mpl_connect("resize_event", mark_background_dirty)
    figure.canvas.draw()

    if blit_state["supports_blit"]:
        blit_state["background"] = figure.canvas.copy_from_bbox(figure.bbox)
        blit_state["needs_background"] = False

    plt.show(block=False)

    return (
        figure,
        axis,
        left_line,
        right_line,
        location_text,
        score_text,
        status_text,
        blit_state,
    )


def snapshot_links(now: float) -> tuple[dict, dict]:
    with STATE_LOCK:
        left = LINKS["left"].snapshot(now)
        right = LINKS["right"].snapshot(now)
    return left, right


def render_dynamic_artists(
    figure,
    axis,
    left_line,
    right_line,
    location_text,
    score_text,
    status_text,
    blit_state,
) -> None:
    """Use blitting when available, with a safe full-draw fallback."""
    canvas = figure.canvas

    if not blit_state["supports_blit"]:
        canvas.draw_idle()
        canvas.flush_events()
        return

    if (
        blit_state["needs_background"]
        or blit_state["background"] is None
    ):
        canvas.draw()
        blit_state["background"] = canvas.copy_from_bbox(figure.bbox)
        blit_state["needs_background"] = False

    canvas.restore_region(blit_state["background"])
    axis.draw_artist(left_line)
    axis.draw_artist(right_line)
    axis.draw_artist(status_text)
    figure.draw_artist(location_text)
    figure.draw_artist(score_text)
    canvas.blit(figure.bbox)
    canvas.flush_events()


def update_plot(
    figure,
    axis,
    left_line,
    right_line,
    location_text,
    score_text,
    status_text,
    blit_state,
    left: dict,
    right: dict,
    now: float,
) -> None:
    if not left["calibrated"] or not right["calibrated"]:
        left_line.set_data([], [])
        right_line.set_data([], [])
        location = "CALIBRATING"
        location_text.set_text(DISPLAY_LOCATION_LABELS[location])
        score_text.set_text(
            f"LEFT {left['calibration_progress']}/{CALIBRATION_PACKETS}"
            f"   |   "
            f"RIGHT {right['calibration_progress']}/{CALIBRATION_PACKETS}"
        )
    else:
        left_x, left_y = smooth_series_for_display(
            left["times"],
            left["scores"],
        )
        right_x, right_y = smooth_series_for_display(
            right["times"],
            right["scores"],
        )
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)

        location = LOCATION_STATE.update(left, right, now)
        location_text.set_text(
            DISPLAY_LOCATION_LABELS.get(location, location)
        )
        score_text.set_text(
            f"LEFT {left['latest_score']:.1f}"
            f"   |   "
            f"RIGHT {right['latest_score']:.1f}"
        )

    status_text.set_text(
        f"LEFT  {left['packet_rate']:5.1f} pkt/s  "
        f"RSSI {str(left['rssi']):>4}  "
        f"score {left['latest_score']:5.2f}\n"
        f"RIGHT {right['packet_rate']:5.1f} pkt/s  "
        f"RSSI {str(right['rssi']):>4}  "
        f"score {right['latest_score']:5.2f}\n"
        f"Rejected length: L={left['rejected_length']} "
        f"R={right['rejected_length']}"
    )

    render_dynamic_artists(
        figure,
        axis,
        left_line,
        right_line,
        location_text,
        score_text,
        status_text,
        blit_state,
    )

# =============================================================================
# MAIN
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smooth dual-link ESP32 CSI movement locator."
    )
    parser.add_argument("--port", default="COM11")
    parser.add_argument("--baud", type=int, default=921600)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    print("Smooth dual-link CSI locator started.")
    print(f"Python: {sys.version.split()[0]}")
    print(f"LEFT  MAC: {LEFT_MAC}")
    print(f"RIGHT MAC: {RIGHT_MAC}")

    (
        figure,
        axis,
        left_line,
        right_line,
        location_text,
        score_text,
        status_text,
        blit_state,
    ) = create_plot()

    reader = threading.Thread(
        target=serial_reader,
        args=(args.port, args.baud),
        daemon=True,
    )
    reader.start()

    last_plot = 0.0
    last_status = 0.0

    try:
        while plt.fignum_exists(figure.number) and not STOP_EVENT.is_set():
            now = time.monotonic()

            if now - last_plot >= PLOT_REFRESH_SECONDS:
                left, right = snapshot_links(now)
                update_plot(
                    figure,
                    axis,
                    left_line,
                    right_line,
                    location_text,
                    score_text,
                    status_text,
                    blit_state,
                    left,
                    right,
                    now,
                )
                last_plot = now

            if now - last_status >= STATUS_PRINT_SECONDS:
                left, right = snapshot_links(now)
                location = LOCATION_STATE.update(left, right, now)
                print(
                    f"LEFT {left['packet_rate']:5.1f} pkt/s "
                    f"score={left['latest_score']:5.2f} | "
                    f"RIGHT {right['packet_rate']:5.1f} pkt/s "
                    f"score={right['latest_score']:5.2f} | "
                    f"{location}"
                )
                last_status = now

            plt.pause(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        STOP_EVENT.set()
        reader.join(timeout=1.0)
        plt.close(figure)

    if SERIAL_ERROR:
        raise SystemExit(
            f"Serial error: {SERIAL_ERROR}\n"
            "Close ESP-IDF Monitor, Arduino Serial Monitor, and any other "
            "program using the AP COM port."
        )

    print("Smooth dual-link CSI locator stopped.")


if __name__ == "__main__":
    main()
