import math
import collections
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from wait_timer import WaitTimer
from read_stdin import readline, print_until_first_csi_line

# ---------------- Settings ----------------
HISTORY_PACKETS = 800
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

PLOT_Y_MAX = 15.0
MIN_MEDIAN_AMPLITUDE = 1.0

# ---------------- State ----------------
perm_amp = collections.deque(maxlen=HISTORY_PACKETS)
perm_time = collections.deque(maxlen=HISTORY_PACKETS)

packet_count = 0
total_packet_count = 0
expected_subcarrier_count = None

calibrated = False
selected_subcarriers = None
shape_center = None
shape_scale = None
quiet_center = None
quiet_scale = None
quiet_scale_floor = None

movement_detected = False
above_threshold_updates = 0
below_threshold_updates = 0

print_stats_wait_timer = WaitTimer(1.0)
render_plot_wait_timer = WaitTimer(0.2)

plt.ion()
fig = plt.figure(figsize=(11, 6))
fig.canvas.draw()
plt.show(block=False)


def robust_mad(values, axis=None):
    med = np.median(values, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(values - med), axis=axis)


def packet_shape_matrix(amplitudes, carrier_indices):
    chosen = np.maximum(amplitudes[:, carrier_indices], 1e-6)
    log_amp = np.log(chosen)
    # Remove packet-wide gain/AGC changes.
    return log_amp - np.median(log_amp, axis=1, keepdims=True)


def choose_subcarriers(calibration_amplitudes):
    median_amp = np.median(calibration_amplitudes, axis=0)
    amp_mad = robust_mad(calibration_amplitudes, axis=0)
    robust_cv = amp_mad / np.maximum(median_amp, 1e-6)

    candidates = np.where(median_amp > MIN_MEDIAN_AMPLITUDE)[0]
    if len(candidates) < 12:
        candidates = np.where(median_amp > 0.25)[0]
    if len(candidates) == 0:
        raise RuntimeError("No usable CSI subcarriers found.")

    # Remove the noisiest 20% observed during quiet calibration.
    cutoff = np.percentile(robust_cv[candidates], 80)
    stable = candidates[robust_cv[candidates] <= cutoff]

    if len(stable) < min(16, len(candidates)):
        ordered = candidates[np.argsort(robust_cv[candidates])]
        stable = ordered[:min(32, len(ordered))]

    return np.sort(stable)


def correlated_hampel_filter(matrix):
    filtered = matrix.copy()
    half = HAMPEL_HALF_WINDOW

    if len(matrix) < 2 * half + 1:
        return filtered

    for i in range(half, len(matrix) - half):
        window = matrix[i-half:i+half+1]
        local_med = np.median(window, axis=0)
        local_mad = 1.4826 * np.median(np.abs(window - local_med), axis=0)
        threshold = HAMPEL_SIGMA * np.maximum(local_mad, 0.20)
        outliers = np.abs(matrix[i] - local_med) > threshold

        # Suppress isolated radio glitches, but preserve coordinated motion.
        fraction = np.mean(outliers)
        if 0 < fraction < 0.25:
            filtered[i, outliers] = local_med[outliers]

    return filtered


def rolling_variance(matrix, window):
    if len(matrix) < window:
        return np.empty((0, matrix.shape[1]))

    c1 = np.vstack([np.zeros((1, matrix.shape[1])), np.cumsum(matrix, axis=0)])
    c2 = np.vstack([np.zeros((1, matrix.shape[1])), np.cumsum(matrix * matrix, axis=0)])

    sums = c1[window:] - c1[:-window]
    sums_sq = c2[window:] - c2[:-window]
    var = (sums_sq - sums * sums / window) / max(window - 1, 1)
    return np.maximum(var, 0.0)


def smooth_1d(values, window):
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def calculate_motion_metric(amplitudes):
    shapes = packet_shape_matrix(amplitudes, selected_subcarriers)
    normalized = (shapes - shape_center) / shape_scale
    cleaned = correlated_hampel_filter(normalized)

    var_by_carrier = rolling_variance(cleaned, MOTION_WINDOW)
    if len(var_by_carrier) == 0:
        return np.array([])

    # Robust fusion across subcarriers.
    metric = np.sqrt(np.median(var_by_carrier, axis=1))
    return smooth_1d(metric, SCORE_SMOOTHING_WINDOW)


def calibrate(calibration_amplitudes):
    global calibrated, selected_subcarriers, shape_center, shape_scale
    global quiet_center, quiet_scale, quiet_scale_floor

    selected_subcarriers = choose_subcarriers(calibration_amplitudes)
    shapes = packet_shape_matrix(calibration_amplitudes, selected_subcarriers)

    shape_center = np.median(shapes, axis=0)
    shape_scale = np.maximum(robust_mad(shapes, axis=0), 0.03)

    quiet_metric = calculate_motion_metric(calibration_amplitudes)
    edge = min(10, len(quiet_metric) // 4)
    quiet_values = quiet_metric[edge:-edge] if len(quiet_metric) > 2 * edge else quiet_metric

    quiet_center = float(np.median(quiet_values))
    initial_mad = float(robust_mad(quiet_values))
    quiet_scale_floor = max(0.10 * max(quiet_center, 1e-3), 1e-3)
    quiet_scale = max(initial_mad, quiet_scale_floor)
    calibrated = True

    print(
        f"Calibration complete | selected subcarriers: {len(selected_subcarriers)} | "
        f"quiet center: {quiet_center:.4f} | quiet scale: {quiet_scale:.4f}"
    )


def update_detection(latest_metric):
    global quiet_center, quiet_scale, movement_detected
    global above_threshold_updates, below_threshold_updates

    score = max(0.0, (latest_metric - quiet_center) / max(quiet_scale, quiet_scale_floor))

    if movement_detected:
        below_threshold_updates = below_threshold_updates + 1 if score < EXIT_THRESHOLD else 0
        if below_threshold_updates >= EXIT_UPDATES:
            movement_detected = False
            below_threshold_updates = 0
            above_threshold_updates = 0
    else:
        above_threshold_updates = above_threshold_updates + 1 if score > ENTER_THRESHOLD else 0
        if above_threshold_updates >= ENTER_UPDATES:
            movement_detected = True
            above_threshold_updates = 0
            below_threshold_updates = 0

    # Follow slow quiet drift, never an active movement.
    if (
        ADAPTIVE_QUIET_MODEL
        and not movement_detected
        and score < ADAPT_ONLY_BELOW_SCORE
    ):
        a = QUIET_LEARNING_RATE
        old_center = quiet_center
        quiet_center = (1-a) * quiet_center + a * latest_metric
        dev = 1.4826 * abs(latest_metric - old_center)
        quiet_scale = max(quiet_scale_floor, (1-a) * quiet_scale + a * dev)

    return score


def carrier_plot(amplitudes, timestamps):
    plt.clf()
    ax = plt.gca()

    data = np.asarray(amplitudes, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)

    if data.ndim != 2 or len(data) < 3:
        return

    if not calibrated:
        if len(data) < CALIBRATION_PACKETS:
            ax.text(
                0.5, 0.62, "CALIBRATING QUIET ROOM",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=18, fontweight="bold"
            )
            ax.text(
                0.5, 0.43,
                f"{len(data)}/{CALIBRATION_PACKETS} packets\n"
                "Keep people, objects, boards, antennas, and cables still",
                ha="center", va="center", transform=ax.transAxes, fontsize=13
            )
            ax.set_xlim(0, CALIBRATION_PACKETS)
            ax.set_ylim(0, PLOT_Y_MAX)
            ax.set_xlabel("Calibration packets")
            ax.set_ylabel("Motion score")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.canvas.flush_events()
            plt.show()
            return

        calibrate(data[:CALIBRATION_PACKETS])

    metric = calculate_motion_metric(data)
    if len(metric) == 0:
        return

    latest_score = update_detection(float(metric[-1]))
    scores = np.maximum(
        0.0,
        (metric - quiet_center) / max(quiet_scale, quiet_scale_floor)
    )

    aligned_times = times[MOTION_WINDOW - 1:][-len(scores):]
    if len(aligned_times) == len(scores) and len(aligned_times) > 1:
        x = (aligned_times - aligned_times[-1]) / 1_000_000.0
        if (
            not np.all(np.isfinite(x))
            or np.any(np.diff(x) < 0)
            or abs(x[0]) > 60
        ):
            x = np.arange(-len(scores) + 1, 1)
            xlabel = "Recent motion windows"
        else:
            xlabel = "Time relative to now (seconds)"
    else:
        x = np.arange(-len(scores) + 1, 1)
        xlabel = "Recent motion windows"

    status = "MOVEMENT DETECTED" if movement_detected else "QUIET"
    line_color = "tab:red" if movement_detected else "tab:blue"

    clipped = np.clip(scores, 0, PLOT_Y_MAX)
    ax.plot(x, clipped, linewidth=2.3, color=line_color, label="Motion score")
    ax.axhspan(0, EXIT_THRESHOLD, alpha=0.08, color="tab:green", label="Quiet region")
    ax.axhline(
        ENTER_THRESHOLD, linestyle="--", linewidth=1.8,
        color="tab:red", label=f"Detection threshold ({ENTER_THRESHOLD:.0f})"
    )
    ax.axhline(
        EXIT_THRESHOLD, linestyle=":", linewidth=1.3,
        color="tab:orange", label=f"Release threshold ({EXIT_THRESHOLD:.0f})"
    )
    ax.fill_between(
        x, ENTER_THRESHOLD, clipped,
        where=scores >= ENTER_THRESHOLD, alpha=0.20, color="tab:red"
    )

    if len(x) > 1:
        ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, PLOT_Y_MAX)
    ax.set_yticks([0, 1, 2, 3, 6, 9, 12, 15])
    ax.set_yticklabels(["0", "1", "2", "3", "6", "9", "12", "15+"])

    rate_text = "rate unknown"
    if len(times) > 20:
        back = min(101, len(times))
        dt = (times[-1] - times[-back]) / 1_000_000.0
        if dt > 0:
            rate_text = f"{((back - 1) / dt):.0f} pkt/s"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Motion score above calibrated quiet noise")
    ax.set_title(f"{status} | score {latest_score:.1f} | {rate_text}")
    ax.text(
        0.01, 0.97,
        f"Selected subcarriers: {len(selected_subcarriers)}\n"
        f"Enter > {ENTER_THRESHOLD:.0f}, release < {EXIT_THRESHOLD:.0f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.flush_events()
    plt.show()


def process(line):
    global expected_subcarrier_count

    try:
        fields = line.split(",", 25)
        if len(fields) < 26:
            return False

        local_timestamp = int(fields[18].strip())
        payload = fields[25]
        left, right = payload.find("["), payload.rfind("]")
        if left < 0 or right <= left:
            return False

        raw = [int(v) for v in payload[left+1:right].split() if v]
        if len(raw) < 4:
            return False
        if len(raw) % 2:
            raw = raw[:-1]

        imaginary = np.asarray(raw[0::2], dtype=np.float64)
        real = np.asarray(raw[1::2], dtype=np.float64)
        amplitudes = np.sqrt(imaginary**2 + real**2)

        if expected_subcarrier_count is None:
            expected_subcarrier_count = len(amplitudes)
        if len(amplitudes) != expected_subcarrier_count:
            return False

        perm_amp.append(amplitudes)
        perm_time.append(local_timestamp)
        return True

    except (ValueError, IndexError, OverflowError):
        return False


print_until_first_csi_line()

while True:
    line = readline()
    if "CSI_DATA" not in line:
        continue
    if not process(line):
        continue

    packet_count += 1
    total_packet_count += 1

    if print_stats_wait_timer.check():
        print_stats_wait_timer.update()
        print(
            "Packet Count:", packet_count, "per second.",
            "Total Count:", total_packet_count
        )
        packet_count = 0

    if render_plot_wait_timer.check() and len(perm_amp) > 2:
        render_plot_wait_timer.update()
        carrier_plot(perm_amp, perm_time)
