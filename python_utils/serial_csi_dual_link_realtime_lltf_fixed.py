"""
Low-latency dual-link ESP32 Wi-Fi CSI movement localization with fixed LLTF parsing.

Designed for this hardware mapping:
    LEFT STA  / COM9  / MAC 02:00:00:00:00:01
    RIGHT STA / COM10 / MAC 02:00:00:00:00:02
    AP receiver / COM11 / 921600 baud

The detector preserves the successful ideas from the original single-link
movement script:
    * log-amplitude CSI shape
    * packet-wide median/common-mode removal
    * quiet calibration with stable subcarrier selection
    * robust per-subcarrier normalization
    * Hampel-style isolated-glitch suppression
    * rolling temporal variance
    * robust fusion across subcarriers
    * quiet-normalized score and hysteresis

It adds low-latency multi-link engineering:
    * direct serial access (no idf.py monitor pipe)
    * chunked serial reads
    * a bounded drop-oldest queue so stale CSI never accumulates
    * independent calibration and scoring for each link
    * a time-normalized motion window derived from each link's packet rate
    * incremental rolling sums instead of recomputing the full history
    * a small packet-to-packet CSI-change feature for faster response
    * no display-only smoothing or resampling delay
    * normalizes 128/256/384-byte CSI payloads to the first 128 LLTF bytes
    * separates unknown-MAC rows from genuinely malformed serial rows

This estimates the location of MOVEMENT as LEFT, CENTER/BOTH, RIGHT, or QUIET.
It is not coordinate-level localization of a motionless person. Coordinate or
multi-zone localization requires labeled fingerprint data and a trained model.

Python 3.14+ dependencies:
    py -3.14 -m pip install --upgrade numpy matplotlib pyserial

Run:
    py -3.14 serial_csi_dual_link_realtime_lltf_fixed.py --port COM11 --baud 921600
"""

from __future__ import annotations

import argparse
import collections
import math
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
    import serial
except ImportError as exc:
    raise SystemExit(
        "pySerial is missing. Install it with:\n"
        "  py -3.14 -m pip install --upgrade pyserial"
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


# =============================================================================
# HARDWARE
# =============================================================================

LEFT_MAC = "02:00:00:00:00:01"
RIGHT_MAC = "02:00:00:00:00:02"

MAC_TO_LINK = {
    LEFT_MAC: "left",
    RIGHT_MAC: "right",
}


# Steven Hernandez's firmware prints CSI as signed int8 values in this order:
# LLTF, then HT-LTF, then STBC-HT-LTF. Each complex sample is imaginary first,
# then real. We use only the first LLTF block so every accepted packet has the
# same 64-complex-subcarrier shape, regardless of whether the metadata says
# 128, 256, 376, 380, 384, or another larger total length.
LLTF_RAW_VALUE_COUNT = 128
LLTF_COMPLEX_SUBCARRIERS = LLTF_RAW_VALUE_COUNT // 2


# =============================================================================
# LOW-LATENCY DETECTOR DEFAULTS
# =============================================================================

# 200 packets is about 5 s at 40 pkt/s or 8 s at 25 pkt/s.
CALIBRATION_PACKETS = 200

# The old script used 30 packets. At 25 pkt/s that alone spans 1.2 seconds.
# Here the packet count is chosen after calibration so the window spans about
# the same amount of real time on both links.
MOTION_WINDOW_SECONDS = 0.50
MIN_MOTION_WINDOW_PACKETS = 4
MAX_MOTION_WINDOW_PACKETS = 28

# Limit computation and reduce redundant/noisy carriers. This is selected from
# the quietest usable carriers for each link, not hard-coded carrier numbers.
MAX_SELECTED_SUBCARRIERS = 24
MIN_MEDIAN_AMPLITUDE = 1.0

# Causal Hampel-style suppression: it uses only current/past packets, so it does
# not add a future-data delay.
HAMPEL_HISTORY_PACKETS = 5
HAMPEL_SIGMA = 4.5
HAMPEL_MIN_SCALE = 0.20

# Exponential smoothing is causal and has much less delay than a 5-packet box
# average. Larger alpha responds faster; 0.45 is a practical compromise.
METRIC_EMA_ALPHA = 0.45

# Packet-to-packet CSI vector change responds before a full variance window.
# Most of the score still comes from the original variance metric.
DELTA_FEATURE_WEIGHT = 0.25

ENTER_THRESHOLD = 5.5
EXIT_THRESHOLD = 2.7
ENTER_UPDATES = 2
EXIT_UPDATES = 3

ADAPTIVE_QUIET_MODEL = True
QUIET_LEARNING_RATE = 0.002
ADAPT_ONLY_BELOW_SCORE = 1.5

# Coarse location from relative link disturbance.
LOCATION_MIN_SCORE = 2.8
LOCATION_DOMINANCE = 0.18
LOCATION_CONFIRM_FRAMES = 2
QUIET_CONFIRM_FRAMES = 3

LEFT_SCORE_GAIN = 1.0
RIGHT_SCORE_GAIN = 1.0

# Serial pipeline. A queue that grows beyond this point is stale; old lines are
# dropped rather than displayed seconds later.
MAX_QUEUED_LINES = 160
STALE_BACKLOG_TRIGGER = 16
STALE_BACKLOG_KEEP = 8
SERIAL_READ_TIMEOUT_SECONDS = 0.02

# Graph. There is no extra graph smoothing; the detector's causal EMA is what
# you see. Keeping only recent points makes redraws inexpensive.
DISPLAY_SECONDS = 12.0
SCORE_HISTORY_POINTS = 480
PLOT_Y_MAX = 15.0
PLOT_REFRESH_HZ = 15.0
STATUS_PRINT_SECONDS = 2.0


# =============================================================================
# NUMERICAL HELPERS
# =============================================================================


def robust_mad(values: np.ndarray, axis=None) -> np.ndarray:
    """Median absolute deviation, scaled like a standard deviation."""
    median = np.median(values, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(values - median), axis=axis)


def packet_shape_matrix(
    amplitudes: np.ndarray,
    carrier_indices: np.ndarray,
) -> np.ndarray:
    """Log amplitude with packet-wide common gain/AGC change removed."""
    chosen = np.maximum(amplitudes[:, carrier_indices], 1e-6)
    log_amplitude = np.log(chosen)
    return log_amplitude - np.median(log_amplitude, axis=1, keepdims=True)


def packet_shape(
    amplitude: np.ndarray,
    carrier_indices: np.ndarray,
) -> np.ndarray:
    chosen = np.maximum(amplitude[carrier_indices], 1e-6)
    log_amplitude = np.log(chosen)
    return log_amplitude - np.median(log_amplitude)


def choose_subcarriers(calibration_amplitudes: np.ndarray) -> np.ndarray:
    """Select the most stable usable carriers during quiet calibration."""
    median_amplitude = np.median(calibration_amplitudes, axis=0)
    amplitude_mad = robust_mad(calibration_amplitudes, axis=0)
    robust_cv = amplitude_mad / np.maximum(median_amplitude, 1e-6)

    candidates = np.where(median_amplitude > MIN_MEDIAN_AMPLITUDE)[0]
    if len(candidates) < 12:
        candidates = np.where(median_amplitude > 0.25)[0]
    if len(candidates) == 0:
        raise RuntimeError("No usable CSI subcarriers were found.")

    ordered = candidates[np.argsort(robust_cv[candidates])]
    count = min(MAX_SELECTED_SUBCARRIERS, len(ordered))
    count = max(min(12, len(ordered)), count)
    return np.sort(ordered[:count])


def correlated_hampel_filter_offline(matrix: np.ndarray) -> np.ndarray:
    """Offline calibration version of the original glitch filter."""
    filtered = matrix.copy()
    half = HAMPEL_HISTORY_PACKETS // 2
    if len(matrix) < 2 * half + 1:
        return filtered

    for index in range(half, len(matrix) - half):
        window = matrix[index - half : index + half + 1]
        local_median = np.median(window, axis=0)
        local_mad = 1.4826 * np.median(
            np.abs(window - local_median),
            axis=0,
        )
        threshold = HAMPEL_SIGMA * np.maximum(
            local_mad,
            HAMPEL_MIN_SCALE,
        )
        outliers = np.abs(matrix[index] - local_median) > threshold
        fraction = float(np.mean(outliers))
        if 0.0 < fraction < 0.25:
            filtered[index, outliers] = local_median[outliers]

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


def causal_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    result = np.empty(len(values), dtype=np.float64)
    result[0] = float(values[0])
    for index in range(1, len(values)):
        result[index] = (
            alpha * float(values[index])
            + (1.0 - alpha) * result[index - 1]
        )
    return result


def robust_center_scale(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 1.0, 1e-3

    edge = min(8, len(finite) // 5)
    useful = finite[edge:-edge] if len(finite) > 2 * edge else finite
    center = float(np.median(useful))
    scale_floor = max(0.10 * max(center, 1e-3), 1e-3)
    scale = max(float(robust_mad(useful)), scale_floor)
    return center, scale, scale_floor


def estimate_packet_rate(timestamp_seconds: np.ndarray) -> float:
    if len(timestamp_seconds) < 4:
        return 25.0
    differences = np.diff(timestamp_seconds)
    useful = differences[
        np.isfinite(differences)
        & (differences > 0.002)
        & (differences < 0.25)
    ]
    if len(useful) == 0:
        return 25.0
    return float(1.0 / np.median(useful))


# =============================================================================
# DROP-OLDEST LINE QUEUE
# =============================================================================


class LatestLineQueue:
    """Thread-safe queue that prioritizes fresh CSI over stale CSI."""

    def __init__(self, max_lines: int) -> None:
        self._items: Deque[tuple[float, bytes]] = collections.deque()
        self._condition = threading.Condition()
        self._max_lines = max_lines
        self.dropped_overflow = 0
        self.dropped_stale = 0

    def put(self, item: tuple[float, bytes]) -> None:
        with self._condition:
            while len(self._items) >= self._max_lines:
                self._items.popleft()
                self.dropped_overflow += 1
            self._items.append(item)
            self._condition.notify()

    def get(
        self,
        stop_event: threading.Event,
        timeout: float = 0.10,
    ) -> Optional[tuple[float, bytes]]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._items and not stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

            if not self._items:
                return None

            if len(self._items) > STALE_BACKLOG_TRIGGER:
                drop_count = len(self._items) - STALE_BACKLOG_KEEP
                for _ in range(drop_count):
                    self._items.popleft()
                self.dropped_stale += drop_count

            return self._items.popleft()

    def depth(self) -> int:
        with self._condition:
            return len(self._items)

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()


# =============================================================================
# ONE INDEPENDENT LOW-LATENCY DETECTOR PER LINK
# =============================================================================


@dataclass
class LinkDetector:
    name: str
    mac: str
    score_gain: float = 1.0

    expected_subcarrier_count: Optional[int] = None
    calibration_amplitudes: list[np.ndarray] = field(default_factory=list)
    calibration_timestamps: list[float] = field(default_factory=list)

    calibrated: bool = False
    selected_subcarriers: Optional[np.ndarray] = None
    shape_center: Optional[np.ndarray] = None
    shape_scale: Optional[np.ndarray] = None
    motion_window_packets: int = MIN_MOTION_WINDOW_PACKETS

    variance_center: float = 0.0
    variance_scale: float = 1.0
    variance_scale_floor: float = 1e-3
    delta_center: float = 0.0
    delta_scale: float = 1.0
    delta_scale_floor: float = 1e-3

    normalized_window: Deque[np.ndarray] = field(default_factory=collections.deque)
    running_sum: Optional[np.ndarray] = None
    running_sum_squared: Optional[np.ndarray] = None
    hampel_history: Deque[np.ndarray] = field(
        default_factory=lambda: collections.deque(maxlen=HAMPEL_HISTORY_PACKETS)
    )
    previous_cleaned: Optional[np.ndarray] = None

    variance_ema: Optional[float] = None
    delta_ema: Optional[float] = None

    score_times: Deque[float] = field(
        default_factory=lambda: collections.deque(maxlen=SCORE_HISTORY_POINTS)
    )
    score_history: Deque[float] = field(
        default_factory=lambda: collections.deque(maxlen=SCORE_HISTORY_POINTS)
    )
    rate_timestamps: Deque[float] = field(
        default_factory=lambda: collections.deque(maxlen=300)
    )

    latest_score: float = 0.0
    latest_variance_score: float = 0.0
    latest_delta_score: float = 0.0
    latest_rssi: Optional[int] = None
    movement_detected: bool = False
    above_threshold_updates: int = 0
    below_threshold_updates: int = 0

    total_received: int = 0
    rejected_length: int = 0
    timestamp_resets: int = 0

    def calibration_progress(self) -> int:
        return min(len(self.calibration_amplitudes), CALIBRATION_PACKETS)

    def add_packet(
        self,
        amplitude: np.ndarray,
        rssi: int,
        timestamp_seconds: float,
    ) -> None:
        if self.expected_subcarrier_count is None:
            self.expected_subcarrier_count = LLTF_COMPLEX_SUBCARRIERS

        if len(amplitude) != self.expected_subcarrier_count:
            self.rejected_length += 1
            return

        if self.rate_timestamps and timestamp_seconds < self.rate_timestamps[-1]:
            self.timestamp_resets += 1
            self.rate_timestamps.clear()
            self.score_times.clear()
            self.score_history.clear()

        self.total_received += 1
        self.latest_rssi = rssi
        self.rate_timestamps.append(timestamp_seconds)
        while (
            len(self.rate_timestamps) > 2
            and timestamp_seconds - self.rate_timestamps[0] > 2.0
        ):
            self.rate_timestamps.popleft()

        if not self.calibrated:
            self.calibration_amplitudes.append(amplitude)
            self.calibration_timestamps.append(timestamp_seconds)
            if len(self.calibration_amplitudes) >= CALIBRATION_PACKETS:
                self.calibrate()
            return

        self.process_live_packet(amplitude, timestamp_seconds)

    def calibrate(self) -> None:
        amplitudes = np.asarray(
            self.calibration_amplitudes[:CALIBRATION_PACKETS],
            dtype=np.float64,
        )
        timestamps = np.asarray(
            self.calibration_timestamps[:CALIBRATION_PACKETS],
            dtype=np.float64,
        )

        self.selected_subcarriers = choose_subcarriers(amplitudes)
        shapes = packet_shape_matrix(amplitudes, self.selected_subcarriers)
        self.shape_center = np.median(shapes, axis=0)
        self.shape_scale = np.maximum(robust_mad(shapes, axis=0), 0.03)

        normalized = (shapes - self.shape_center) / self.shape_scale
        cleaned = correlated_hampel_filter_offline(normalized)

        estimated_rate = estimate_packet_rate(timestamps)
        self.motion_window_packets = int(
            np.clip(
                round(estimated_rate * MOTION_WINDOW_SECONDS),
                MIN_MOTION_WINDOW_PACKETS,
                MAX_MOTION_WINDOW_PACKETS,
            )
        )

        variance_by_carrier = rolling_variance(
            cleaned,
            self.motion_window_packets,
        )
        if len(variance_by_carrier) < 8:
            raise RuntimeError(
                f"{self.name}: insufficient calibration data for "
                f"window={self.motion_window_packets}"
            )

        variance_metric = np.sqrt(np.median(variance_by_carrier, axis=1))
        variance_metric = causal_ema(variance_metric, METRIC_EMA_ALPHA)

        differences = np.diff(cleaned, axis=0)
        delta_metric = np.sqrt(np.mean(differences * differences, axis=1))
        delta_metric = causal_ema(delta_metric, METRIC_EMA_ALPHA)

        (
            self.variance_center,
            self.variance_scale,
            self.variance_scale_floor,
        ) = robust_center_scale(variance_metric)
        (
            self.delta_center,
            self.delta_scale,
            self.delta_scale_floor,
        ) = robust_center_scale(delta_metric)

        selected_count = len(self.selected_subcarriers)
        self.running_sum = np.zeros(selected_count, dtype=np.float64)
        self.running_sum_squared = np.zeros(selected_count, dtype=np.float64)
        self.normalized_window.clear()
        self.hampel_history.clear()

        for row in cleaned[-self.motion_window_packets :]:
            self._append_window(row.copy())
            self.hampel_history.append(row.copy())

        self.previous_cleaned = cleaned[-1].copy()
        self.variance_ema = float(variance_metric[-1])
        self.delta_ema = float(delta_metric[-1])
        self.calibrated = True

        self.calibration_amplitudes.clear()
        self.calibration_timestamps.clear()

        print(
            f"{self.name.upper()} calibrated | "
            f"rate={estimated_rate:.1f} pkt/s | "
            f"window={self.motion_window_packets} packets "
            f"({self.motion_window_packets / max(estimated_rate, 1e-6):.2f}s) | "
            f"subcarriers={selected_count}"
        )
        if estimated_rate < 8.0:
            print(
                f"WARNING: {self.name.upper()} is only {estimated_rate:.1f} pkt/s. "
                "At this rate, Python cannot provide sub-second variance-based "
                "localization. Correct the active_sta FreeRTOS delay so "
                "PACKET_RATE 25 produces about 25 packets/s."
            )

    def _causal_hampel(self, normalized: np.ndarray) -> np.ndarray:
        if len(self.hampel_history) < 3:
            cleaned = normalized.copy()
            self.hampel_history.append(cleaned.copy())
            return cleaned

        window = np.vstack([*self.hampel_history, normalized])
        local_median = np.median(window, axis=0)
        local_mad = 1.4826 * np.median(
            np.abs(window - local_median),
            axis=0,
        )
        threshold = HAMPEL_SIGMA * np.maximum(
            local_mad,
            HAMPEL_MIN_SCALE,
        )
        outliers = np.abs(normalized - local_median) > threshold
        fraction = float(np.mean(outliers))

        cleaned = normalized.copy()
        if 0.0 < fraction < 0.25:
            cleaned[outliers] = local_median[outliers]

        self.hampel_history.append(cleaned.copy())
        return cleaned

    def _append_window(self, row: np.ndarray) -> None:
        if self.running_sum is None or self.running_sum_squared is None:
            raise RuntimeError(f"{self.name}: detector window is uninitialized")

        while len(self.normalized_window) >= self.motion_window_packets:
            oldest = self.normalized_window.popleft()
            self.running_sum -= oldest
            self.running_sum_squared -= oldest * oldest

        self.normalized_window.append(row)
        self.running_sum += row
        self.running_sum_squared += row * row

    def _variance_metric(self) -> Optional[float]:
        count = len(self.normalized_window)
        if count < self.motion_window_packets:
            return None
        if self.running_sum is None or self.running_sum_squared is None:
            return None

        variance = (
            self.running_sum_squared
            - self.running_sum * self.running_sum / count
        ) / max(count - 1, 1)
        variance = np.maximum(variance, 0.0)
        return float(np.sqrt(np.median(variance)))

    def process_live_packet(
        self,
        amplitude: np.ndarray,
        timestamp_seconds: float,
    ) -> None:
        if (
            self.selected_subcarriers is None
            or self.shape_center is None
            or self.shape_scale is None
        ):
            return

        shape = packet_shape(amplitude, self.selected_subcarriers)
        normalized = (shape - self.shape_center) / self.shape_scale
        cleaned = self._causal_hampel(normalized)

        if self.previous_cleaned is None:
            delta_raw = self.delta_center
        else:
            difference = cleaned - self.previous_cleaned
            delta_raw = float(np.sqrt(np.mean(difference * difference)))
        self.previous_cleaned = cleaned.copy()

        self._append_window(cleaned)
        variance_raw = self._variance_metric()
        if variance_raw is None:
            return

        if self.variance_ema is None:
            self.variance_ema = variance_raw
        else:
            self.variance_ema = (
                METRIC_EMA_ALPHA * variance_raw
                + (1.0 - METRIC_EMA_ALPHA) * self.variance_ema
            )

        if self.delta_ema is None:
            self.delta_ema = delta_raw
        else:
            self.delta_ema = (
                METRIC_EMA_ALPHA * delta_raw
                + (1.0 - METRIC_EMA_ALPHA) * self.delta_ema
            )

        variance_score = max(
            0.0,
            (self.variance_ema - self.variance_center)
            / max(self.variance_scale, self.variance_scale_floor),
        )
        delta_score = max(
            0.0,
            (self.delta_ema - self.delta_center)
            / max(self.delta_scale, self.delta_scale_floor),
        )

        combined = (
            (1.0 - DELTA_FEATURE_WEIGHT) * variance_score
            + DELTA_FEATURE_WEIGHT * delta_score
        )
        score = max(0.0, combined * self.score_gain)

        self.latest_variance_score = variance_score
        self.latest_delta_score = delta_score
        self.latest_score = score

        self._update_hysteresis(score)
        self._adapt_quiet_model(score, variance_raw, delta_raw)

        self.score_times.append(timestamp_seconds)
        self.score_history.append(score)

    def _update_hysteresis(self, score: float) -> None:
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

    def _adapt_quiet_model(
        self,
        score: float,
        variance_raw: float,
        delta_raw: float,
    ) -> None:
        if (
            not ADAPTIVE_QUIET_MODEL
            or self.movement_detected
            or score >= ADAPT_ONLY_BELOW_SCORE
        ):
            return

        alpha = QUIET_LEARNING_RATE

        old_variance_center = self.variance_center
        self.variance_center = (
            (1.0 - alpha) * self.variance_center
            + alpha * variance_raw
        )
        variance_deviation = 1.4826 * abs(
            variance_raw - old_variance_center
        )
        self.variance_scale = max(
            self.variance_scale_floor,
            (1.0 - alpha) * self.variance_scale
            + alpha * variance_deviation,
        )

        old_delta_center = self.delta_center
        self.delta_center = (
            (1.0 - alpha) * self.delta_center
            + alpha * delta_raw
        )
        delta_deviation = 1.4826 * abs(delta_raw - old_delta_center)
        self.delta_scale = max(
            self.delta_scale_floor,
            (1.0 - alpha) * self.delta_scale
            + alpha * delta_deviation,
        )

    def packet_rate(self) -> float:
        if len(self.rate_timestamps) < 2:
            return 0.0
        elapsed = self.rate_timestamps[-1] - self.rate_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.rate_timestamps) - 1) / elapsed

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "calibrated": self.calibrated,
            "calibration_progress": self.calibration_progress(),
            "window_packets": self.motion_window_packets,
            "selected_count": (
                0
                if self.selected_subcarriers is None
                else len(self.selected_subcarriers)
            ),
            "times": np.asarray(self.score_times, dtype=np.float64),
            "scores": np.asarray(self.score_history, dtype=np.float64),
            "latest_score": self.latest_score,
            "variance_score": self.latest_variance_score,
            "delta_score": self.latest_delta_score,
            "movement": self.movement_detected,
            "packet_rate": self.packet_rate(),
            "rssi": self.latest_rssi,
            "total_received": self.total_received,
            "rejected_length": self.rejected_length,
            "timestamp_resets": self.timestamp_resets,
        }


LINKS = {
    "left": LinkDetector(
        name="left",
        mac=LEFT_MAC,
        score_gain=LEFT_SCORE_GAIN,
    ),
    "right": LinkDetector(
        name="right",
        mac=RIGHT_MAC,
        score_gain=RIGHT_SCORE_GAIN,
    ),
}

STATE_LOCK = threading.Lock()


# =============================================================================
# CSI PARSING AND THREADS
# =============================================================================


@dataclass
class PipelineStats:
    serial_bytes: int = 0
    csi_lines_read: int = 0
    parsed_packets: int = 0
    ignored_unknown_mac: int = 0
    malformed_lines: int = 0
    short_payload_lines: int = 0
    odd_payload_lines: int = 0
    lltf_truncated_packets: int = 0
    reported_length_mismatch: int = 0
    payload_length_counts: collections.Counter[int] = field(
        default_factory=collections.Counter
    )
    reported_length_counts: collections.Counter[int] = field(
        default_factory=collections.Counter
    )
    last_error: str = ""


PIPELINE_STATS = PipelineStats()


def parse_csi_line(line: bytes) -> Optional[tuple[str, int, float, np.ndarray]]:
    """
    Parse one ESP32-CSI-Tool row and return a fixed 64-complex LLTF vector.

    The CSV metadata field ``len`` reports the CSI buffer length received by
    ESP-IDF. Steven Hernandez's firmware can still print only the first 128
    signed values when "collect only LLTF" is enabled. Conversely, when a
    longer payload is printed, ESP-IDF orders it LLTF, HT-LTF, STBC-HT-LTF.
    Therefore the first 128 signed values are the common LLTF block in both
    cases.

    Unknown MAC rows are ignored but are not counted as malformed.
    """
    marker = b"CSI_DATA,AP,"
    start = line.find(marker)
    if start < 0:
        return None

    fields = line[start:].split(b",", 25)
    if len(fields) < 26:
        PIPELINE_STATS.malformed_lines += 1
        return None

    try:
        mac = fields[2].strip().decode("ascii", errors="ignore").upper()
        link_name = MAC_TO_LINK.get(mac)
        if link_name is None:
            PIPELINE_STATS.ignored_unknown_mac += 1
            return None

        rssi = int(fields[3].strip())
        reported_length = int(fields[24].strip())
        PIPELINE_STATS.reported_length_counts[reported_length] += 1

        # Prefer the repository's steady-clock timestamp, which is already in
        # seconds and does not wrap as quickly as the Wi-Fi rx_ctrl timestamp.
        try:
            timestamp_seconds = float(fields[23].strip())
        except ValueError:
            timestamp_seconds = int(fields[18].strip()) / 1_000_000.0

        payload = fields[25]
        left_bracket = payload.find(b"[")
        right_bracket = payload.rfind(b"]")
        if left_bracket < 0 or right_bracket <= left_bracket:
            PIPELINE_STATS.malformed_lines += 1
            return None

        payload_text = payload[left_bracket + 1 : right_bracket].decode(
            "ascii",
            errors="ignore",
        )
        raw = np.fromstring(payload_text, sep=" ", dtype=np.float64)
        raw_count = int(len(raw))
        PIPELINE_STATS.payload_length_counts[raw_count] += 1

        if raw_count < LLTF_RAW_VALUE_COUNT:
            PIPELINE_STATS.short_payload_lines += 1
            return None

        if raw_count % 2:
            PIPELINE_STATS.odd_payload_lines += 1
            raw = raw[:-1]
            raw_count -= 1

        # A metadata length larger than the printed payload is normal when the
        # firmware setting "collect only LLTF" is enabled. Count only the
        # opposite/inconsistent case as a diagnostic mismatch.
        if raw_count > reported_length and reported_length > 0:
            PIPELINE_STATS.reported_length_mismatch += 1

        if raw_count > LLTF_RAW_VALUE_COUNT:
            PIPELINE_STATS.lltf_truncated_packets += 1

        # LLTF is always first in the ESP-IDF CSI buffer.
        lltf_raw = raw[:LLTF_RAW_VALUE_COUNT]

        imaginary = lltf_raw[0::2]
        real = lltf_raw[1::2]
        amplitude = np.hypot(imaginary, real)

        if len(amplitude) != LLTF_COMPLEX_SUBCARRIERS:
            PIPELINE_STATS.malformed_lines += 1
            return None

        return link_name, rssi, timestamp_seconds, amplitude

    except (ValueError, IndexError, OverflowError):
        PIPELINE_STATS.malformed_lines += 1
        return None


def serial_reader(
    serial_port: serial.Serial,
    line_queue: LatestLineQueue,
    stop_event: threading.Event,
) -> None:
    buffer = bytearray()

    try:
        while not stop_event.is_set():
            waiting = serial_port.in_waiting
            chunk = serial_port.read(waiting if waiting > 0 else 4096)
            if not chunk:
                continue

            PIPELINE_STATS.serial_bytes += len(chunk)
            buffer.extend(chunk)

            while True:
                newline_index = buffer.find(b"\n")
                if newline_index < 0:
                    break

                line = bytes(buffer[:newline_index]).rstrip(b"\r")
                del buffer[: newline_index + 1]

                if b"CSI_DATA,AP," not in line:
                    continue

                PIPELINE_STATS.csi_lines_read += 1
                line_queue.put((time.monotonic(), line))

            # Guard against a corrupt stream with no newline.
            if len(buffer) > 1_000_000:
                buffer.clear()
                PIPELINE_STATS.malformed_lines += 1

    except Exception as exc:  # serial disconnects must stop the whole app
        PIPELINE_STATS.last_error = f"serial reader: {exc}"
        stop_event.set()
        line_queue.wake_all()


def csi_processor(
    line_queue: LatestLineQueue,
    stop_event: threading.Event,
) -> None:
    try:
        while not stop_event.is_set():
            queued = line_queue.get(stop_event, timeout=0.10)
            if queued is None:
                continue

            _host_arrival, line = queued
            parsed = parse_csi_line(line)
            if parsed is None:
                continue

            link_name, rssi, timestamp_seconds, amplitude = parsed
            with STATE_LOCK:
                LINKS[link_name].add_packet(
                    amplitude=amplitude,
                    rssi=rssi,
                    timestamp_seconds=timestamp_seconds,
                )
            PIPELINE_STATS.parsed_packets += 1

    except Exception as exc:
        PIPELINE_STATS.last_error = f"CSI processor: {exc}"
        stop_event.set()
        line_queue.wake_all()


# =============================================================================
# LOCATION FUSION
# =============================================================================


@dataclass
class LocationDecision:
    current: str = "CALIBRATING"
    candidate: str = "CALIBRATING"
    candidate_frames: int = 0
    confidence: float = 0.0

    def classify_instant(self, left: dict, right: dict) -> tuple[str, float]:
        if not left["calibrated"] or not right["calibrated"]:
            return "CALIBRATING", 0.0

        left_score = float(left["latest_score"])
        right_score = float(right["latest_score"])
        maximum = max(left_score, right_score)

        movement_active = (
            bool(left["movement"])
            or bool(right["movement"])
            or maximum >= LOCATION_MIN_SCORE
        )
        if not movement_active:
            return "QUIET", 1.0

        total = left_score + right_score + 1e-9
        dominance = (left_score - right_score) / total
        confidence = min(1.0, abs(dominance) / max(LOCATION_DOMINANCE, 1e-6))

        if dominance >= LOCATION_DOMINANCE and left_score >= LOCATION_MIN_SCORE:
            return "LEFT", confidence
        if dominance <= -LOCATION_DOMINANCE and right_score >= LOCATION_MIN_SCORE:
            return "RIGHT", confidence
        return "CENTER / BOTH LINKS", 1.0 - min(1.0, abs(dominance))

    def update(self, left: dict, right: dict) -> tuple[str, float]:
        instant, confidence = self.classify_instant(left, right)

        if instant == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = instant
            self.candidate_frames = 1

        required = (
            QUIET_CONFIRM_FRAMES
            if instant == "QUIET"
            else LOCATION_CONFIRM_FRAMES
        )
        if instant == "CALIBRATING" or self.candidate_frames >= required:
            self.current = instant
            self.confidence = confidence

        return self.current, self.confidence


LOCATION_DECISION = LocationDecision()


# =============================================================================
# GRAPH
# =============================================================================


def create_plot():
    plt.ion()
    matplotlib.rcParams["path.simplify"] = True
    matplotlib.rcParams["path.simplify_threshold"] = 0.5

    figure, axis = plt.subplots(figsize=(12.4, 7.2))
    figure.canvas.manager.set_window_title(
        "Real-Time Dual-Link ESP32 CSI Movement Localization"
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.12,
        top=0.78,
    )

    (left_line,) = axis.plot(
        [],
        [],
        linewidth=2.2,
        label="LEFT link",
        solid_capstyle="round",
    )
    (right_line,) = axis.plot(
        [],
        [],
        linewidth=2.2,
        label="RIGHT link",
        solid_capstyle="round",
    )

    axis.axhspan(0, EXIT_THRESHOLD, alpha=0.08, label="Quiet region")
    axis.axhline(
        ENTER_THRESHOLD,
        linestyle="--",
        linewidth=1.4,
        label=f"Enter threshold ({ENTER_THRESHOLD:.1f})",
    )
    axis.axhline(
        EXIT_THRESHOLD,
        linestyle=":",
        linewidth=1.2,
        label=f"Exit threshold ({EXIT_THRESHOLD:.1f})",
    )

    axis.set_xlim(-DISPLAY_SECONDS, 0.0)
    axis.set_ylim(0.0, PLOT_Y_MAX)
    axis.set_xlabel("Time relative to newest CSI packet (seconds)")
    axis.set_ylabel("Quiet-normalized movement score")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)

    title_text = figure.suptitle(
        "CALIBRATING — KEEP THE ROOM STILL",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    score_text = figure.text(
        0.5,
        0.915,
        "LEFT 0.0   |   RIGHT 0.0",
        ha="center",
        va="top",
        fontsize=11,
    )
    status_text = figure.text(
        0.5,
        0.865,
        "Starting serial pipeline...",
        ha="center",
        va="top",
        fontsize=9,
        family="monospace",
    )

    figure.canvas.draw()
    plt.show(block=False)
    return figure, axis, left_line, right_line, title_text, score_text, status_text


def history_for_plot(
    times: np.ndarray,
    scores: np.ndarray,
    newest_timestamp: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(times) == 0 or len(scores) == 0:
        return np.empty(0), np.empty(0)

    x = times - newest_timestamp
    usable = (
        np.isfinite(x)
        & np.isfinite(scores)
        & (x >= -DISPLAY_SECONDS)
        & (x <= 0.25)
    )
    x = x[usable]
    y = np.clip(scores[usable], 0.0, PLOT_Y_MAX)

    # Hard cap the points drawn. This decimates only the old visual history;
    # the detector always processes every non-stale packet.
    max_plot_points = 260
    if len(x) > max_plot_points:
        indices = np.linspace(0, len(x) - 1, max_plot_points).astype(int)
        x = x[indices]
        y = y[indices]

    return x, y


def snapshot_links() -> tuple[dict, dict]:
    with STATE_LOCK:
        return LINKS["left"].snapshot(), LINKS["right"].snapshot()


def update_plot(
    figure,
    _axis,
    left_line,
    right_line,
    title_text,
    score_text,
    status_text,
    left: dict,
    right: dict,
    line_queue: LatestLineQueue,
) -> str:
    location, confidence = LOCATION_DECISION.update(left, right)

    latest_times: list[float] = []
    if len(left["times"]):
        latest_times.append(float(left["times"][-1]))
    if len(right["times"]):
        latest_times.append(float(right["times"][-1]))
    newest_timestamp = max(latest_times) if latest_times else 0.0

    left_x, left_y = history_for_plot(
        left["times"],
        left["scores"],
        newest_timestamp,
    )
    right_x, right_y = history_for_plot(
        right["times"],
        right["scores"],
        newest_timestamp,
    )
    left_line.set_data(left_x, left_y)
    right_line.set_data(right_x, right_y)

    if location == "CALIBRATING":
        title_text.set_text("CALIBRATING — KEEP THE ROOM STILL")
        score_text.set_text(
            f"LEFT {left['calibration_progress']}/{CALIBRATION_PACKETS}"
            f"   |   RIGHT {right['calibration_progress']}/{CALIBRATION_PACKETS}"
        )
    else:
        title_text.set_text(location)
        score_text.set_text(
            f"LEFT {left['latest_score']:.1f}"
            f"   |   RIGHT {right['latest_score']:.1f}"
            f"   |   confidence {confidence:.2f}"
        )

    status_text.set_text(
        f"L {left['packet_rate']:5.1f} pkt/s  RSSI {str(left['rssi']):>4}  "
        f"window {left['window_packets']:2d}  "
        f"var {left['variance_score']:4.1f}  Δ {left['delta_score']:4.1f}   |   "
        f"R {right['packet_rate']:5.1f} pkt/s  RSSI {str(right['rssi']):>4}  "
        f"window {right['window_packets']:2d}  "
        f"var {right['variance_score']:4.1f}  Δ {right['delta_score']:4.1f}\n"
        f"queue {line_queue.depth():2d}  "
        f"stale {line_queue.dropped_stale}  "
        f"overflow {line_queue.dropped_overflow}  "
        f"bad {PIPELINE_STATS.malformed_lines}  "
        f"short {PIPELINE_STATS.short_payload_lines}  "
        f"unknown {PIPELINE_STATS.ignored_unknown_mac}\n"
        f"LLTF fixed at {LLTF_COMPLEX_SUBCARRIERS} complex carriers  "
        f"long payloads normalized {PIPELINE_STATS.lltf_truncated_packets}  "
        f"length rejects L={left['rejected_length']} "
        f"R={right['rejected_length']}"
    )

    figure.canvas.draw_idle()
    figure.canvas.flush_events()
    return location


# =============================================================================
# COMMAND LINE AND MAIN
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Low-latency dual-link ESP32 CSI localization with fixed LLTF parsing."
    )
    parser.add_argument("--port", default="COM11", help="AP serial port")
    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
        help="AP runtime serial baud rate",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run terminal-only for latency/packet-rate testing",
    )
    return parser


def open_serial_port(port: str, baud: int) -> serial.Serial:
    serial_port = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=SERIAL_READ_TIMEOUT_SECONDS,
        write_timeout=1.0,
    )
    serial_port.dtr = False
    serial_port.rts = False

    try:
        serial_port.set_buffer_size(rx_size=262_144, tx_size=16_384)
    except (AttributeError, NotImplementedError, OSError):
        pass

    serial_port.reset_input_buffer()
    return serial_port


def main() -> None:
    args = build_argument_parser().parse_args()

    print("Real-time dual-link CSI locator with fixed LLTF parsing started.")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Port: {args.port} @ {args.baud}")
    print(f"LEFT : {LEFT_MAC}")
    print(f"RIGHT: {RIGHT_MAC}")
    print("Close ESP-IDF Monitor and every other program using the AP port.")

    try:
        serial_port = open_serial_port(args.port, args.baud)
    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {args.port}: {exc}\n"
            "Close ESP-IDF Monitor, Arduino Serial Monitor, old graph scripts, "
            "and any backend using this COM port."
        ) from exc

    print("Serial port opened. Keep the room still during calibration.")

    stop_event = threading.Event()
    line_queue = LatestLineQueue(MAX_QUEUED_LINES)

    reader_thread = threading.Thread(
        target=serial_reader,
        args=(serial_port, line_queue, stop_event),
        name="serial-reader",
        daemon=True,
    )
    processor_thread = threading.Thread(
        target=csi_processor,
        args=(line_queue, stop_event),
        name="csi-processor",
        daemon=True,
    )
    reader_thread.start()
    processor_thread.start()

    plot_objects = None if args.no_plot else create_plot()
    last_status_print = 0.0
    refresh_period = 1.0 / PLOT_REFRESH_HZ

    try:
        while not stop_event.is_set():
            loop_start = time.monotonic()
            left, right = snapshot_links()

            if plot_objects is not None:
                figure = plot_objects[0]
                if not plt.fignum_exists(figure.number):
                    break
                location = update_plot(
                    *plot_objects,
                    left,
                    right,
                    line_queue,
                )
                plt.pause(0.001)
            else:
                location, _confidence = LOCATION_DECISION.update(left, right)

            now = time.monotonic()
            if now - last_status_print >= STATUS_PRINT_SECONDS:
                print(
                    f"L {left['packet_rate']:5.1f} pkt/s "
                    f"score={left['latest_score']:5.2f} | "
                    f"R {right['packet_rate']:5.1f} pkt/s "
                    f"score={right['latest_score']:5.2f} | "
                    f"{location} | queue={line_queue.depth()} "
                    f"stale_dropped={line_queue.dropped_stale}"
                )
                last_status_print = now

            remaining = refresh_period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        line_queue.wake_all()
        try:
            serial_port.close()
        except Exception:
            pass
        reader_thread.join(timeout=1.0)
        processor_thread.join(timeout=1.0)

        if PIPELINE_STATS.last_error:
            print(f"Pipeline stopped because: {PIPELINE_STATS.last_error}")
        print(
            "Stopped | "
            f"CSI lines={PIPELINE_STATS.csi_lines_read} | "
            f"parsed={PIPELINE_STATS.parsed_packets} | "
            f"stale dropped={line_queue.dropped_stale} | "
            f"overflow dropped={line_queue.dropped_overflow} | "
            f"bad={PIPELINE_STATS.malformed_lines} | "
            f"short={PIPELINE_STATS.short_payload_lines} | "
            f"unknown={PIPELINE_STATS.ignored_unknown_mac} | "
            f"long payloads normalized={PIPELINE_STATS.lltf_truncated_packets} | "
            f"payload lengths={dict(PIPELINE_STATS.payload_length_counts)} | "
            f"reported lengths={dict(PIPELINE_STATS.reported_length_counts)}"
        )


if __name__ == "__main__":
    main()
