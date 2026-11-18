from __future__ import annotations

"""Shared feature extraction and evaluation utilities for three-link TriCSI data."""

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


LINK_TO_MAC = {
    "back_left": "02:00:00:00:00:01",
    "middle_back": "02:00:00:00:00:04",
    "back_right": "02:00:00:00:00:03",
}

LINK_SUBSETS = {
    "1tx_back_left": ("back_left",),
    "1tx_middle_back": ("middle_back",),
    "1tx_back_right": ("back_right",),
    "2tx_outer": ("back_left", "back_right"),
    "2tx_left_middle": ("back_left", "middle_back"),
    "2tx_middle_right": ("middle_back", "back_right"),
    "3tx_all": ("back_left", "middle_back", "back_right"),
}

CLASS_ORDER = ("EMPTY", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9")

DEFAULT_ZONE_COORDS = {
    "Z1": (0.4064, 1.5240),
    "Z2": (1.1176, 1.5240),
    "Z3": (1.8288, 1.5240),
    "Z4": (0.4064, 0.9144),
    "Z5": (1.1176, 0.9144),
    "Z6": (1.8288, 0.9144),
    "Z7": (0.4064, 0.3048),
    "Z8": (1.1176, 0.3048),
    "Z9": (1.8288, 0.3048),
}

# ESP32 LLTF: 52 populated complex subcarriers. The other 12 are guard/DC bins.
ACTIVE_SUBCARRIERS = tuple(list(range(6, 32)) + list(range(33, 59)))


def normalize_mac(value: str) -> str:
    return str(value).strip().upper().replace("-", ":")


@dataclass
class SessionFeatures:
    session_name: str
    source_dir: str
    metadata: pd.DataFrame
    features_by_link: dict[str, np.ndarray]
    feature_names_by_link: dict[str, list[str]]
    zone_coords: dict[str, tuple[float, float]]


def _natural_trial_key(path: Path) -> tuple[int, int, str]:
    upper = path.stem.upper()
    zone_match = re.search(r"(?:^|_)(EMPTY|Z[1-9])(?:_|$)", upper)
    trial_match = re.search(r"TRIAL[_-]?0*(\d+)", upper)
    zone = zone_match.group(1) if zone_match else "ZZZ"
    trial = int(trial_match.group(1)) if trial_match else 999
    zone_rank = CLASS_ORDER.index(zone) if zone in CLASS_ORDER else 999
    return zone_rank, trial, path.name


def discover_trial_files(session_dir: Path) -> list[Path]:
    files = sorted(session_dir.glob("*_trial*.csv"), key=_natural_trial_key)
    if not files:
        raise FileNotFoundError(f"No *_trial*.csv files found in {session_dir}")
    return files


def session_data_sha256(session_dir: str | Path) -> str:
    """Content fingerprint of the ordered trial CSV bytes (support files excluded)."""
    digest = hashlib.sha256()
    for path in discover_trial_files(Path(session_dir).expanduser().resolve()):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _read_session_manifest(session_dir: Path) -> tuple[str, dict[str, tuple[float, float]]]:
    manifest_path = session_dir / "session_manifest.json"
    session_name = session_dir.name
    coords = dict(DEFAULT_ZONE_COORDS)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        session_name = str(manifest.get("session", session_name))
        for zone, xy in manifest.get("zone_centers_m", {}).items():
            if zone in DEFAULT_ZONE_COORDS and isinstance(xy, list) and len(xy) == 2:
                coords[zone] = (float(xy[0]), float(xy[1]))
    return session_name, coords


def _infer_trial_identity(path: Path, frame: pd.DataFrame) -> tuple[str, int, str]:
    if "zone" in frame.columns and len(frame):
        zones = frame["zone"].dropna().astype(str).str.upper().unique()
        if len(zones) != 1 or zones[0] not in CLASS_ORDER:
            raise ValueError(f"{path.name}: expected one valid zone label, found {zones.tolist()}")
        zone = zones[0]
    else:
        match = re.search(r"(?:^|_)(EMPTY|Z[1-9])(?:_|$)", path.stem.upper())
        if not match:
            raise ValueError(f"{path.name}: cannot infer zone")
        zone = match.group(1)

    if "trial" in frame.columns and len(frame):
        trials = pd.to_numeric(frame["trial"], errors="coerce").dropna().astype(int).unique()
        if len(trials) != 1:
            raise ValueError(f"{path.name}: expected one trial number, found {trials.tolist()}")
        trial = int(trials[0])
    else:
        match = re.search(r"TRIAL[_-]?0*(\d+)", path.stem.upper())
        if not match:
            raise ValueError(f"{path.name}: cannot infer trial number")
        trial = int(match.group(1))

    if "orientation" in frame.columns and len(frame):
        orientations = frame["orientation"].dropna().astype(str).str.upper().unique()
        orientation = orientations[0] if len(orientations) == 1 else "NA"
    else:
        orientation = "NA" if zone == "EMPTY" else {1: "N", 2: "E", 3: "S", 4: "W", 5: "N"}.get(trial, "?")
    return zone, trial, orientation


def _feature_names() -> list[str]:
    names: list[str] = []
    for prefix in ("shape_median", "shape_iqr", "shape_motion"):
        names.extend(f"{prefix}_sc{index:02d}" for index in ACTIVE_SUBCARRIERS)
    names.extend(
        [
            "gain_median",
            "gain_iqr",
            "gain_motion",
            "rssi_median",
            "rssi_iqr",
            "rssi_motion",
        ]
    )
    return names


def _iqr(values: np.ndarray, axis=None) -> np.ndarray:
    return np.percentile(values, 75, axis=axis) - np.percentile(values, 25, axis=axis)


def _window_features(log_amplitude: np.ndarray, rssi: np.ndarray) -> np.ndarray:
    # Packet centering suppresses common receiver-gain changes while preserving
    # the frequency-selective CSI shape. RSSI is retained separately.
    packet_gain = np.median(log_amplitude, axis=1)
    shape = log_amplitude - packet_gain[:, None]

    shape_median = np.median(shape, axis=0)
    shape_iqr = _iqr(shape, axis=0)
    shape_motion = np.mean(np.abs(np.diff(shape, axis=0)), axis=0)

    gain_features = np.array(
        [
            np.median(packet_gain),
            _iqr(packet_gain),
            np.mean(np.abs(np.diff(packet_gain))),
        ],
        dtype=float,
    )
    rssi_features = np.array(
        [
            np.median(rssi),
            _iqr(rssi),
            np.mean(np.abs(np.diff(rssi))),
        ],
        dtype=float,
    )
    return np.concatenate([shape_median, shape_iqr, shape_motion, gain_features, rssi_features])


def extract_session_features(
    session_dir: str | Path,
    *,
    window_packets: int = 64,
    windows_per_trial: int = 7,
) -> SessionFeatures:
    session_dir = Path(session_dir).expanduser().resolve()
    if window_packets < 8:
        raise ValueError("window_packets must be at least 8")
    if windows_per_trial < 1:
        raise ValueError("windows_per_trial must be positive")

    session_name, zone_coords = _read_session_manifest(session_dir)
    trial_files = discover_trial_files(session_dir)
    expected = {(zone, trial) for zone in CLASS_ORDER for trial in range(1, 6)}
    observed: set[tuple[str, int]] = set()

    metadata_rows: list[dict] = []
    features: dict[str, list[np.ndarray]] = {link: [] for link in LINK_TO_MAC}
    names = _feature_names()

    for path in trial_files:
        frame = pd.read_csv(path, low_memory=False)
        zone, trial, orientation = _infer_trial_identity(path, frame)
        identity = (zone, trial)
        if identity in observed:
            raise ValueError(f"Duplicate trial {identity} in {session_dir}")
        observed.add(identity)

        iq_cols = [f"iq_{index:03d}" for index in range(128)]
        missing = [column for column in ("mac", "rssi", *iq_cols) if column not in frame.columns]
        if missing:
            raise ValueError(f"{path.name}: missing required columns: {missing[:8]}")

        macs = frame["mac"].astype(str).map(normalize_mac)
        per_link: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        packet_counts: dict[str, int] = {}
        for link, expected_mac in LINK_TO_MAC.items():
            link_frame = frame.loc[macs == expected_mac].copy()
            if link_frame.empty:
                raise ValueError(f"{path.name}: no packets for {link} ({expected_mac})")
            sort_column = "host_time_s" if "host_time_s" in link_frame.columns else "local_timestamp"
            if sort_column in link_frame.columns:
                link_frame = link_frame.sort_values(sort_column, kind="stable")

            raw = link_frame[iq_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            rssi = pd.to_numeric(link_frame["rssi"], errors="coerce").to_numpy(dtype=float)
            good = np.isfinite(raw).all(axis=1) & np.isfinite(rssi)
            raw = raw[good]
            rssi = rssi[good]
            if len(raw) < window_packets:
                raise ValueError(
                    f"{path.name}: {link} has only {len(raw)} valid packets; need {window_packets}"
                )

            imag = raw[:, 0::2]
            real = raw[:, 1::2]
            amplitude = np.hypot(real, imag)[:, ACTIVE_SUBCARRIERS]
            log_amplitude = np.log1p(amplitude)
            per_link[link] = (log_amplitude, rssi)
            packet_counts[link] = len(raw)

        fractions = np.linspace(0.0, 1.0, windows_per_trial)
        for window_index, fraction in enumerate(fractions):
            for link in LINK_TO_MAC:
                log_amplitude, rssi = per_link[link]
                max_start = len(log_amplitude) - window_packets
                start = int(round(float(fraction) * max_start))
                stop = start + window_packets
                features[link].append(_window_features(log_amplitude[start:stop], rssi[start:stop]))

            x_m, y_m = (np.nan, np.nan) if zone == "EMPTY" else zone_coords[zone]
            metadata_rows.append(
                {
                    "session": session_name,
                    "trial_uid": f"{session_name}|{zone}|{trial:02d}",
                    "trial_file": path.name,
                    "zone": zone,
                    "trial": trial,
                    "orientation": orientation,
                    "x_m": x_m,
                    "y_m": y_m,
                    "window_index": window_index,
                    "window_fraction": float(fraction),
                    **{f"packets_{link}": packet_counts[link] for link in LINK_TO_MAC},
                }
            )

    if observed != expected:
        missing_trials = sorted(expected - observed, key=lambda item: (CLASS_ORDER.index(item[0]), item[1]))
        extra_trials = sorted(observed - expected)
        raise ValueError(
            f"{session_name}: expected the 50-trial protocol. Missing={missing_trials}; extra={extra_trials}"
        )

    arrays = {link: np.vstack(rows).astype(np.float32) for link, rows in features.items()}
    feature_names = {link: [f"{link}__{name}" for name in names] for link in LINK_TO_MAC}
    return SessionFeatures(
        session_name=session_name,
        source_dir=str(session_dir),
        metadata=pd.DataFrame(metadata_rows),
        features_by_link=arrays,
        feature_names_by_link=feature_names,
        zone_coords=zone_coords,
    )


def combine_links(dataset: SessionFeatures, links: Sequence[str]) -> np.ndarray:
    unknown = [link for link in links if link not in dataset.features_by_link]
    if unknown:
        raise ValueError(f"Unknown links: {unknown}")
    return np.concatenate([dataset.features_by_link[link] for link in links], axis=1)


def combine_sessions(datasets: Iterable[SessionFeatures], links: Sequence[str]) -> tuple[np.ndarray, pd.DataFrame]:
    datasets = list(datasets)
    matrix = np.concatenate([combine_links(dataset, links) for dataset in datasets], axis=0)
    metadata = pd.concat([dataset.metadata for dataset in datasets], ignore_index=True)
    return matrix, metadata


def make_classifier(*, seed: int = 2026, n_estimators: int = 400) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def room_diagonal(zone_coords: dict[str, tuple[float, float]]) -> float:
    points = np.asarray(list(zone_coords.values()), dtype=float)
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def position_error(
    true_zone: str,
    predicted_zone: str,
    zone_coords: dict[str, tuple[float, float]],
) -> float:
    if true_zone == "EMPTY":
        return float("nan")
    if predicted_zone == "EMPTY":
        return room_diagonal(zone_coords)
    return float(np.linalg.norm(np.asarray(zone_coords[true_zone]) - np.asarray(zone_coords[predicted_zone])))


def evaluate_classifier(
    model: ExtraTreesClassifier,
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    zone_coords: dict[str, tuple[float, float]],
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    probabilities = model.predict_proba(matrix)
    classes = np.asarray(model.classes_, dtype=str)
    window_predictions = classes[np.argmax(probabilities, axis=1)]
    window_accuracy = float(accuracy_score(metadata["zone"], window_predictions))

    trial_rows: list[dict] = []
    for trial_uid, indexes in metadata.groupby("trial_uid", sort=False).groups.items():
        index_array = np.asarray(list(indexes), dtype=int)
        mean_probability = probabilities[index_array].mean(axis=0)
        predicted = str(classes[int(np.argmax(mean_probability))])
        row = metadata.iloc[index_array[0]]
        true_zone = str(row["zone"])
        error_m = position_error(true_zone, predicted, zone_coords)
        trial_rows.append(
            {
                "session": row["session"],
                "trial_uid": trial_uid,
                "trial_file": row["trial_file"],
                "true_zone": true_zone,
                "predicted_zone": predicted,
                "correct": int(predicted == true_zone),
                "position_error_m": error_m,
                "confidence": float(np.max(mean_probability)),
                **{f"p_{zone}": float(mean_probability[np.where(classes == zone)[0][0]]) for zone in classes},
            }
        )
    trial_frame = pd.DataFrame(trial_rows)
    occupied = trial_frame["true_zone"] != "EMPTY"
    empty = ~occupied
    errors = trial_frame.loc[occupied, "position_error_m"].to_numpy(dtype=float)
    metrics = {
        "window_accuracy": window_accuracy,
        "trial_accuracy": float(trial_frame["correct"].mean()),
        "occupied_trial_accuracy": float(trial_frame.loc[occupied, "correct"].mean()),
        "empty_recall": float(trial_frame.loc[empty, "correct"].mean()),
        "mean_position_error_m": float(np.mean(errors)),
        "median_position_error_m": float(np.median(errors)),
        "n_trials": int(len(trial_frame)),
        "n_windows": int(len(metadata)),
    }
    labels = list(CLASS_ORDER)
    cm = confusion_matrix(trial_frame["true_zone"], trial_frame["predicted_zone"], labels=labels)
    return metrics, trial_frame, cm


def stratified_paired_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    strata_column: str = "true_zone",
    iterations: int = 5000,
    seed: int = 2026,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups = [group[value_column].dropna().to_numpy(dtype=float) for _, group in frame.groupby(strata_column)]
    groups = [group for group in groups if len(group)]
    observed = float(np.mean(np.concatenate(groups)))
    samples = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        resampled = [rng.choice(group, size=len(group), replace=True) for group in groups]
        samples[iteration] = np.mean(np.concatenate(resampled))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"estimate": observed, "ci95_low": float(low), "ci95_high": float(high)}
