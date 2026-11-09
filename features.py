from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from csi_io import load_trial_csv
from project_config import (
    HOP_SECONDS,
    MIN_PACKETS_PER_LINK,
    WINDOW_SECONDS,
)


def iq_to_amplitude(iq: np.ndarray) -> np.ndarray:
    """Hernandez/ESP-IDF CSI pairs are imaginary, real."""
    x = np.asarray(iq, dtype=np.float64)
    imaginary = x[:, 0::2]
    real = x[:, 1::2]
    return np.hypot(imaginary, real)


def _safe_iqr(values: np.ndarray, axis=0) -> np.ndarray:
    q75 = np.percentile(values, 75, axis=axis)
    q25 = np.percentile(values, 25, axis=axis)
    return q75 - q25


def link_feature_dict(
    iq: np.ndarray,
    rssi: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """
    Extract both STATIC and DYNAMIC CSI features.

    Static:
      - median absolute log-amplitude profile across all 64 LLTF carriers
      - median packet-centered spectral-shape profile

    Dynamic:
      - per-carrier std and IQR
      - adjacent-subcarrier difference
      - adjacent-frame Euclidean change
      - temporal variance

    Keeping both absolute and packet-centered shape is deliberate:
    movement detection benefited from removing common-mode gain, but static
    localization can contain useful absolute attenuation information.
    """
    amplitude = iq_to_amplitude(iq)
    log_amp = np.log(np.maximum(amplitude, 0.5))

    packet_center = np.median(log_amp, axis=1, keepdims=True)
    shape = log_amp - packet_center

    feature: dict[str, float] = {}

    # RSSI + overall amplitude statistics.
    feature[f"{prefix}_rssi_mean"] = float(np.mean(rssi))
    feature[f"{prefix}_rssi_std"] = float(np.std(rssi))
    feature[f"{prefix}_rssi_iqr"] = float(_safe_iqr(rssi, axis=0))
    feature[f"{prefix}_global_logamp_mean"] = float(np.mean(log_amp))
    feature[f"{prefix}_global_logamp_std"] = float(np.std(log_amp))

    # Full spectral fingerprint.
    raw_median = np.median(log_amp, axis=0)
    shape_median = np.median(shape, axis=0)
    shape_std = np.std(shape, axis=0)
    shape_iqr = _safe_iqr(shape, axis=0)

    for carrier in range(log_amp.shape[1]):
        feature[f"{prefix}_rawmed_c{carrier:02d}"] = float(raw_median[carrier])
        feature[f"{prefix}_shapemed_c{carrier:02d}"] = float(shape_median[carrier])
        feature[f"{prefix}_shapestd_c{carrier:02d}"] = float(shape_std[carrier])
        feature[f"{prefix}_shapeiqr_c{carrier:02d}"] = float(shape_iqr[carrier])

    # Wi-CaL-inspired dynamic descriptors.
    if len(shape) >= 2:
        frame_diff = np.diff(shape, axis=0)
        frame_euclidean = np.sqrt(np.mean(frame_diff * frame_diff, axis=1))
        feature[f"{prefix}_frame_euc_mean"] = float(np.mean(frame_euclidean))
        feature[f"{prefix}_frame_euc_std"] = float(np.std(frame_euclidean))
        feature[f"{prefix}_frame_euc_p90"] = float(np.percentile(frame_euclidean, 90))
    else:
        feature[f"{prefix}_frame_euc_mean"] = 0.0
        feature[f"{prefix}_frame_euc_std"] = 0.0
        feature[f"{prefix}_frame_euc_p90"] = 0.0

    adjacent = np.abs(np.diff(shape, axis=1))
    feature[f"{prefix}_adjacent_mean"] = float(np.mean(adjacent))
    feature[f"{prefix}_adjacent_std"] = float(np.std(adjacent))

    carrier_variance = np.var(shape, axis=0)
    feature[f"{prefix}_temporal_var_median"] = float(np.median(carrier_variance))
    feature[f"{prefix}_temporal_var_mean"] = float(np.mean(carrier_variance))

    return feature


def compute_window_features(
    left_iq: np.ndarray,
    left_rssi: np.ndarray,
    right_iq: np.ndarray,
    right_rssi: np.ndarray,
) -> dict[str, float]:
    out = {}
    out.update(link_feature_dict(left_iq, left_rssi, "L"))
    out.update(link_feature_dict(right_iq, right_rssi, "R"))

    # Cross-link features: useful when the same body location changes the two
    # paths differently.
    out["X_rssi_mean_difference"] = out["L_rssi_mean"] - out["R_rssi_mean"]
    out["X_global_logamp_difference"] = (
        out["L_global_logamp_mean"] - out["R_global_logamp_mean"]
    )
    out["X_frame_euc_difference"] = (
        out["L_frame_euc_mean"] - out["R_frame_euc_mean"]
    )
    out["X_temporal_var_difference"] = (
        out["L_temporal_var_median"] - out["R_temporal_var_median"]
    )
    out["X_frame_euc_sum"] = out["L_frame_euc_mean"] + out["R_frame_euc_mean"]
    out["X_temporal_var_sum"] = (
        out["L_temporal_var_median"] + out["R_temporal_var_median"]
    )
    return out


def iter_trial_windows(
    path: str | Path,
    *,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    min_packets_per_link: int = MIN_PACKETS_PER_LINK,
) -> Iterator[dict]:
    trial = load_trial_csv(path)
    left = trial["left"]
    right = trial["right"]
    meta = trial["metadata"]

    if len(left["time"]) == 0 or len(right["time"]) == 0:
        return

    # Use the common time interval observed by both links. Exact packet-to-
    # packet pairing is unnecessary because both streams share the AP clock.
    common_start = max(float(left["time"][0]), float(right["time"][0]))
    common_end = min(float(left["time"][-1]), float(right["time"][-1]))

    start = common_start
    window_index = 0

    while start + window_seconds <= common_end + 1e-9:
        end = start + window_seconds

        lm = (left["time"] >= start) & (left["time"] < end)
        rm = (right["time"] >= start) & (right["time"] < end)

        if int(np.sum(lm)) >= min_packets_per_link and int(np.sum(rm)) >= min_packets_per_link:
            features = compute_window_features(
                left["iq"][lm],
                left["rssi"][lm],
                right["iq"][rm],
                right["rssi"][rm],
            )

            try:
                x_m = float(meta["x_m"]) if meta["x_m"] != "" else np.nan
                y_m = float(meta["y_m"]) if meta["y_m"] != "" else np.nan
            except ValueError:
                x_m = y_m = np.nan

            yield {
                "features": features,
                "zone": meta["zone"],
                "x_m": x_m,
                "y_m": y_m,
                "session": meta["session"],
                "trial": str(meta["trial"]),
                "orientation": meta["orientation"],
                "trial_group": (
                    f"{meta['session']}|{meta['zone']}|"
                    f"{meta['trial']}|{meta['orientation']}"
                ),
                "source_file": meta["path"],
                "window_index": window_index,
                "window_start_s": start,
                "left_packets": int(np.sum(lm)),
                "right_packets": int(np.sum(rm)),
            }

        start += hop_seconds
        window_index += 1
