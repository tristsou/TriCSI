"""Non-pickle window caches and strict Empty-reference separation."""
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

LABELS = ["EMPTY"] + [f"Z{i}" for i in range(1, 10)]


@dataclass
class Windows:
    X: np.ndarray
    rssi: np.ndarray
    y_text: np.ndarray
    session: np.ndarray
    trial_number: np.ndarray
    orientation: np.ndarray
    group: np.ndarray
    source_file: np.ndarray
    window_index: np.ndarray
    window_start_s: np.ndarray
    left_packets: np.ndarray
    right_packets: np.ndarray

    def subset(self, mask):
        return Windows(**{f.name: getattr(self, f.name)[mask].copy() for f in fields(self)})

    def validate(self):
        n = len(self.X)
        if self.X.ndim != 4 or self.X.shape[1:] != (4, 32, 64) or not n:
            raise ValueError("Expected nonempty X with shape [N,4,32,64].")
        if self.rssi.shape != (n, 4):
            raise ValueError("RSSI must have shape [N,4].")
        if any(len(getattr(self, f.name)) != n for f in fields(self)):
            raise ValueError("Metadata length mismatch.")
        if not np.isfinite(self.X).all() or not np.isfinite(self.rssi).all():
            raise ValueError("Non-finite CSI or RSSI.")
        if set(self.y_text.astype(str)) - set(LABELS):
            raise ValueError("Unknown class labels.")
        for group in set(self.group.astype(str)):
            mask = self.group == group
            for name in ("session", "y_text", "orientation", "trial_number", "source_file"):
                if len(set(getattr(self, name)[mask].astype(str))) != 1:
                    raise ValueError(f"Group {group} mixes {name} values.")
        return self


def save_windows(path, raw):
    payload = {}
    for f in fields(Windows):
        value = np.asarray(getattr(raw, f.name))
        payload[f.name] = value.astype(str) if value.dtype.kind in "OU" else value
    Windows(**payload).validate()
    np.savez_compressed(path, **payload)


def load_windows(path):
    with np.load(path, allow_pickle=False) as payload:
        raw = Windows(**{f.name: payload[f.name].copy() for f in fields(Windows)})
    return raw.validate()


def combine(sets):
    if not sets:
        raise ValueError("No sessions supplied.")
    return Windows(**{f.name: np.concatenate([getattr(s, f.name) for s in sets])
                      for f in fields(Windows)}).validate()


def calibrate(raw, exclude_occupied=()):
    """Exactly Empty trial 1; no fallback to evaluation Empty recordings."""
    names = set(raw.session.astype(str))
    if len(names) != 1:
        raise ValueError("Calibrate one session at a time.")
    calibration = (raw.y_text == "EMPTY") & (raw.trial_number == 1)
    if not np.any(calibration):
        raise ValueError(f"{names}: missing dedicated Empty trial 1; no fallback permitted.")
    reference = raw.subset(calibration)
    out = raw.subset(np.ones(len(raw.X), dtype=bool))
    background = np.median(reference.X[:, [0, 2]], axis=(0, 2))
    out.X[:, 0] -= background[0]
    out.X[:, 2] -= background[1]
    out.rssi[:, 0] -= np.median(reference.rssi[:, 0])
    out.rssi[:, 2] -= np.median(reference.rssi[:, 2])
    score = ~calibration
    score &= ~((out.y_text != "EMPTY") & np.isin(out.trial_number, exclude_occupied))
    if not np.any(score):
        raise ValueError("No trials remain after calibration exclusions.")
    return out.subset(score), reference


def assert_disjoint(source, target):
    if set(source.session.astype(str)) & set(target.session.astype(str)):
        raise ValueError("Source and evaluation sessions overlap.")
    if set(source.group.astype(str)) & set(target.group.astype(str)):
        raise ValueError("Source and evaluation trial groups overlap.")
    if set(source.source_file.astype(str)) & set(target.source_file.astype(str)):
        raise ValueError("Source and evaluation files overlap.")


def require_all_classes(raw):
    missing = set(LABELS) - set(raw.y_text.astype(str))
    if missing:
        raise ValueError(f"Missing scored/training classes: {sorted(missing)}")


def safe_session(name):
    if not name or name in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError(f"Invalid session name {name!r}")
    return name


def read_sessions(cache, names):
    if len(names) != len(set(names)):
        raise ValueError("Duplicate session IDs.")
    result = {name: load_windows(Path(cache) / f"{safe_session(name)}.npz") for name in names}
    for name, raw in result.items():
        if set(raw.session.astype(str)) != {name}:
            raise ValueError(f"Cache {name} contains different session IDs.")
    return result
