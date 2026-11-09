from __future__ import annotations

"""
TriCSI LIVE two-link late-fusion localizer — BAYESIAN STATE FILTER V5.

Why V5 exists
-------------
V1 read one serial line, then sometimes did feature extraction + 3 RF predictions
+ plotting in the SAME loop. If inference/plotting took too long, COM11 could
accumulate old CSI. The display then looked both slow and wrong because it was
classifying stale packets.

V5 keeps the threaded low-latency reader but replaces hand-written switch rules with a recursive Bayesian state filter over EMPTY + Z1..Z9:
    Thread 1: drain COM11 continuously in chunks
    Thread 2: run localization from the newest complete 1.0 s CSI window
    Main    : display only

The trained model is unchanged:
    models/link_ensemble_rf_s4_s5_s6.joblib

The 1.0 s input window and frozen S4+S5+S6 link-ensemble model are unchanged.
The RF probabilities become observation evidence. A transition matrix says
that staying put is most likely, moving to a neighboring cell is plausible,
and a long jump is possible but unlikely. The posterior is updated every
prediction. No zone-specific correction is used.

Run:
    python live_link_ensemble_bayes_v5.py

Fast terminal-only test:
    python live_link_ensemble_bayes_v5.py --no-plot
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
# Spatial continuity tracker
# ---------------------------------------------------------------------------

GRID_POS = {
    "Z1": (0, 0), "Z2": (1, 0), "Z3": (2, 0),
    "Z4": (0, 1), "Z5": (1, 1), "Z6": (2, 1),
    "Z7": (0, 2), "Z8": (1, 2), "Z9": (2, 2),
}


def zones_adjacent(a: str, b: str) -> bool:
    """Exact or one grid cell away, including diagonal."""
    if a == b:
        return True
    if a not in GRID_POS or b not in GRID_POS:
        return False
    ax, ay = GRID_POS[a]
    bx, by = GRID_POS[b]
    return max(abs(ax - bx), abs(ay - by)) <= 1


class BayesianStateFilter:
    """
    Recursive Bayesian filter over 10 states:
        EMPTY, Z1, ..., Z9

    Observation evidence:
        P(EMPTY) from the binary occupancy RF
        (1 - P(EMPTY)) * P(Zi | occupied) from the late-fusion zone RF

    Dynamics:
        occupied -> same zone      high probability
        occupied -> adjacent zone  moderate probability
        occupied -> far zone       small probability
        occupied -> EMPTY          small probability

        EMPTY -> EMPTY             high probability
        EMPTY -> any occupied      remaining probability, distributed evenly

    The model output is not treated as a hard label. Every update combines
    the new RF evidence with the complete previous belief distribution.
    """

    def __init__(
        self,
        occupied_classes: list[str],
        *,
        stay_prob: float = 0.72,
        adjacent_total: float = 0.24,
        far_total: float = 0.02,
        occupied_to_empty: float = 0.02,
        empty_stay: float = 0.90,
        emission_power: float = 1.0,
        emission_floor: float = 1e-4,
    ):
        self.occupied_classes = list(occupied_classes)
        self.states = ["EMPTY"] + self.occupied_classes
        self.state_index = {s: i for i, s in enumerate(self.states)}

        self.stay_prob = float(stay_prob)
        self.adjacent_total = float(adjacent_total)
        self.far_total = float(far_total)
        self.occupied_to_empty = float(occupied_to_empty)
        self.empty_stay = float(empty_stay)
        self.emission_power = float(emission_power)
        self.emission_floor = float(emission_floor)

        occupied_mass = (
            self.stay_prob
            + self.adjacent_total
            + self.far_total
            + self.occupied_to_empty
        )

        if not np.isclose(occupied_mass, 1.0, atol=1e-6):
            raise ValueError(
                "Occupied transition probabilities must sum to 1.0: "
                f"stay({self.stay_prob}) + adjacent({self.adjacent_total}) + "
                f"far({self.far_total}) + empty({self.occupied_to_empty}) "
                f"= {occupied_mass}"
            )

        if not (0.0 <= self.empty_stay <= 1.0):
            raise ValueError("empty_stay must be between 0 and 1.")

        if self.emission_power <= 0:
            raise ValueError("emission_power must be > 0.")

        self.transition = self._build_transition()
        self.posterior = None

    def _build_transition(self) -> np.ndarray:
        n = len(self.states)
        T = np.zeros((n, n), dtype=np.float64)

        # EMPTY -> EMPTY / any occupied zone
        empty_i = self.state_index["EMPTY"]
        T[empty_i, empty_i] = self.empty_stay

        enter_mass = 1.0 - self.empty_stay
        if self.occupied_classes:
            each = enter_mass / len(self.occupied_classes)
            for z in self.occupied_classes:
                T[empty_i, self.state_index[z]] = each

        # Occupied state dynamics
        for z in self.occupied_classes:
            i = self.state_index[z]

            T[i, empty_i] = self.occupied_to_empty
            T[i, i] = self.stay_prob

            adjacent_zones = [
                q for q in self.occupied_classes
                if q != z and zones_adjacent(z, q)
            ]
            far_zones = [
                q for q in self.occupied_classes
                if q != z and not zones_adjacent(z, q)
            ]

            if adjacent_zones:
                each_adj = self.adjacent_total / len(adjacent_zones)
                for q in adjacent_zones:
                    T[i, self.state_index[q]] = each_adj
            else:
                T[i, i] += self.adjacent_total

            if far_zones:
                each_far = self.far_total / len(far_zones)
                for q in far_zones:
                    T[i, self.state_index[q]] = each_far
            else:
                T[i, i] += self.far_total

        # Numerical safety.
        T /= np.maximum(T.sum(axis=1, keepdims=True), 1e-12)
        return T

    def _emission(
        self,
        p_zone: np.ndarray,
        p_empty: float,
    ) -> np.ndarray:
        p_empty = float(np.clip(p_empty, 0.0, 1.0))

        p_zone = np.asarray(p_zone, dtype=np.float64)
        p_zone = np.maximum(p_zone, 0.0)
        if p_zone.sum() <= 0:
            p_zone = np.full(
                len(self.occupied_classes),
                1.0 / len(self.occupied_classes),
                dtype=np.float64,
            )
        else:
            p_zone = p_zone / p_zone.sum()

        evidence = np.empty(len(self.states), dtype=np.float64)
        evidence[0] = p_empty
        evidence[1:] = (1.0 - p_empty) * p_zone

        evidence = np.maximum(evidence, self.emission_floor)

        # Optional probability tempering. 1.0 means "use RF evidence as-is."
        evidence = evidence ** self.emission_power
        evidence /= evidence.sum()

        return evidence

    def update(
        self,
        p_zone: np.ndarray,
        p_empty: float,
    ) -> dict:
        emission = self._emission(p_zone, p_empty)

        if self.posterior is None:
            # Initialize from the first observation instead of an arbitrary zone.
            posterior = emission.copy()
            prior = np.full(
                len(self.states),
                1.0 / len(self.states),
                dtype=np.float64,
            )
        else:
            prior = self.posterior @ self.transition
            posterior = prior * emission

            s = posterior.sum()
            if s <= 0:
                posterior = emission.copy()
            else:
                posterior /= s

        self.posterior = posterior

        order = np.argsort(posterior)[::-1]
        best_i = int(order[0])
        second_i = int(order[1]) if len(order) > 1 else best_i

        best_state = self.states[best_i]
        best_prob = float(posterior[best_i])
        second_prob = float(posterior[second_i])
        posterior_margin = best_prob - second_prob

        occupied_post = posterior[1:].copy()
        occupied_sum = float(occupied_post.sum())

        if occupied_sum > 0:
            occupied_post /= occupied_sum
        else:
            occupied_post[:] = 1.0 / len(occupied_post)

        return {
            "state": best_state,
            "state_prob": best_prob,
            "posterior_margin": posterior_margin,
            "posterior": posterior.copy(),
            "occupied_posterior": occupied_post,
            "emission": emission.copy(),
            "prior": prior.copy(),
        }

    def top_states(self, n: int = 3):
        if self.posterior is None:
            return []
        idx = np.argsort(self.posterior)[::-1][:n]
        return [
            (self.states[int(i)], float(self.posterior[int(i)]))
            for i in idx
        ]


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
    stay_prob: float,
    adjacent_total: float,
    far_total: float,
    occupied_to_empty: float,
    empty_stay: float,
    emission_power: float,
) -> None:
    global LATEST_PRED, INFERENCE_ERROR

    occupied_classes = [str(x) for x in bundle["occupied_classes"]]
    zone_centers = {
        str(k): tuple(map(float, v))
        for k, v in bundle["zone_centers"].items()
    }

    zone_history = collections.deque(maxlen=consensus_n)
    empty_history = collections.deque(maxlen=consensus_n)

    bayes = BayesianStateFilter(
        occupied_classes,
        stay_prob=stay_prob,
        adjacent_total=adjacent_total,
        far_total=far_total,
        occupied_to_empty=occupied_to_empty,
        empty_stay=empty_stay,
        emission_power=emission_power,
    )

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

            # Keep optional very-short probability consensus. Default is 1,
            # which means the Bayes filter receives the newest RF probabilities.
            zone_history.append(p_zone_now)
            empty_history.append(p_empty_now)

            hist = list(zone_history)
            w = np.arange(1, len(hist) + 1, dtype=np.float64)
            w /= np.sum(w)

            p_zone_obs = np.sum(
                np.stack(hist, axis=0) * w[:, None],
                axis=0,
            )
            p_zone_obs /= np.maximum(p_zone_obs.sum(), 1e-12)

            p_empty_obs = float(
                np.sum(
                    np.asarray(list(empty_history), dtype=np.float64)
                    * w
                )
            )

            raw_i = int(np.argmax(p_zone_now))
            raw_zone = occupied_classes[raw_i]
            raw_zone_prob = float(p_zone_now[raw_i])

            if p_empty_now >= empty_threshold:
                raw_zone = "EMPTY"
                raw_state_prob = p_empty_now
            else:
                raw_state_prob = (1.0 - p_empty_now) * raw_zone_prob

            bayes_result = bayes.update(
                p_zone_obs,
                p_empty_obs,
            )

            bayes_zone = bayes_result["state"]
            bayes_prob = bayes_result["state_prob"]
            bayes_margin = bayes_result["posterior_margin"]
            p_zone_filtered = bayes_result["occupied_posterior"]

            # Filtered continuous XY uses the posterior conditional on occupied.
            x_m, y_m = weighted_xy(
                p_zone_filtered,
                occupied_classes,
                zone_centers,
            )

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

            posterior_top = bayes.top_states(3)

            result = {
                "bayes_zone": bayes_zone,
                "bayes_prob": bayes_prob,
                "bayes_margin": bayes_margin,
                "posterior_top": posterior_top,
                "now_zone": raw_zone,
                "raw_state_prob": float(raw_state_prob),
                "raw_zone_prob": raw_zone_prob,
                "p_zone_raw": p_zone_now,
                "p_zone": p_zone_filtered,
                "p_empty": p_empty_obs,
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
    ax.set_title("TriCSI live localization — Bayesian state filter V5")

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
        default=3.0,
        help="Live prediction refresh rate. Default 3 Hz (matches ~300 ms inference).",
    )
    parser.add_argument(
        "--consensus",
        type=int,
        default=1,
        help="Probability history before spatial tracker. Default 1 for low lag.",
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
    parser.add_argument(
        "--stay-prob",
        type=float,
        default=0.72,
        help="Occupied-state probability of staying in the same zone.",
    )
    parser.add_argument(
        "--adjacent-total",
        type=float,
        default=0.24,
        help="Total occupied-state probability assigned to adjacent zones.",
    )
    parser.add_argument(
        "--far-total",
        type=float,
        default=0.02,
        help="Total occupied-state probability assigned to non-adjacent zones.",
    )
    parser.add_argument(
        "--occupied-to-empty",
        type=float,
        default=0.02,
        help="Occupied-state transition probability to EMPTY.",
    )
    parser.add_argument(
        "--empty-stay",
        type=float,
        default=0.90,
        help="EMPTY-state probability of remaining EMPTY.",
    )
    parser.add_argument(
        "--emission-power",
        type=float,
        default=1.0,
        help="Power applied to RF evidence. 1.0 uses probabilities as-is.",
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

    occupied_transition_sum = (
        args.stay_prob
        + args.adjacent_total
        + args.far_total
        + args.occupied_to_empty
    )
    if not np.isclose(occupied_transition_sum, 1.0, atol=1e-6):
        raise SystemExit(
            "Bayesian occupied-state transition probabilities must sum to 1.0. "
            f"Current sum = {occupied_transition_sum:.6f}"
        )

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
    print("TriCSI LOW-LATENCY LIVE LOCALIZER V5 — BAYESIAN STATE FILTER")
    print("=" * 78)
    print(f"Port             : {args.port} @ {args.baud}")
    print(f"Model            : {args.model}")
    print(f"CSI window       : {args.window:.2f} s  (same as training)")
    print(f"Inference refresh: {args.infer_hz:.1f} Hz")
    print(f"Consensus        : {args.consensus} prediction(s)")
    print(f"Bayes stay       : {args.stay_prob:.2f}")
    print(f"Bayes adjacent   : {args.adjacent_total:.2f} total")
    print(f"Bayes far        : {args.far_total:.2f} total")
    print(f"Occupied->EMPTY  : {args.occupied_to_empty:.2f}")
    print(f"EMPTY stay       : {args.empty_stay:.2f}")
    print(f"Emission power   : {args.emission_power:.2f}")
    print(f"Raw EMPTY thresh : {empty_threshold:.2f}")
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
            args.stay_prob,
            args.adjacent_total,
            args.far_total,
            args.occupied_to_empty,
            args.empty_stay,
            args.emission_power,
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

                posterior_text = " ".join(
                    f"{state}={prob:.2f}"
                    for state, prob in pred["posterior_top"]
                )

                print(
                    f"{pred['bayes_zone']:5s} | "
                    f"NOW={pred['now_zone']:5s} | "
                    f"post={pred['bayes_prob']:.2f} "
                    f"pmargin={pred['bayes_margin']:.2f} "
                    f"rawP={pred['raw_state_prob']:.2f} "
                    f"empty={pred['p_empty']:.2f} | "
                    f"xy=({pred['x_m']:.2f},{pred['y_m']:.2f}) | "
                    f"L->{pred['left_zone']} R->{pred['right_zone']} | "
                    f"{lrate:.1f}/{rrate:.1f} Hz | "
                    f"infer={pred['inference_ms']:.0f}ms "
                    f"fresh={pred['fresh_ms']:.0f}ms | "
                    f"{posterior_text}"
                )

                if plot is not None:
                    plt, fig, dot, status = plot

                    if pred["bayes_zone"] == "EMPTY":
                        dot.set_offsets(np.empty((0, 2)))
                    elif (
                        math.isfinite(pred["x_m"])
                        and math.isfinite(pred["y_m"])
                    ):
                        dot.set_offsets(
                            np.asarray([[pred["x_m"], pred["y_m"]]])
                        )

                    top_post = " ".join(
                        f"{state}:{prob:.2f}"
                        for state, prob in pred["posterior_top"]
                    )
                    status.set_text(
                        f"BAYES: {pred['bayes_zone']}  P={pred['bayes_prob']:.2f}\n"
                        f"NOW:   {pred['now_zone']}  rawP={pred['raw_state_prob']:.2f}\n"
                        f"posterior margin={pred['bayes_margin']:.2f}\n"
                        f"EMPTY obs={pred['p_empty']:.2f}\n"
                        f"xy=({pred['x_m']:.2f}, {pred['y_m']:.2f}) m\n"
                        f"LEFT {pred['left_zone']} | RIGHT {pred['right_zone']}\n"
                        f"{top_post}\n"
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
        print("V5 stopped")
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
