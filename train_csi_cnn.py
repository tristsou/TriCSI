from __future__ import annotations

"""
TriCSI raw-CSI 2D CNN baseline.

This is the next model after the engineered-feature Random Forest.

INPUT PER 1-SECOND WINDOW
-------------------------
The two CSI links are resampled to a fixed 32 time steps. Each ESP32 LLTF
packet contains 64 complex subcarriers.

CNN image shape:
    4 x 32 x 64

Channels:
    0 = LEFT  absolute log-amplitude
    1 = LEFT  packet-centered log-amplitude
    2 = RIGHT absolute log-amplitude
    3 = RIGHT packet-centered log-amplitude

A small RSSI auxiliary branch is enabled by default. It receives four numbers:
    left mean RSSI, left std RSSI, right mean RSSI, right std RSSI

The network deliberately preserves BOTH absolute amplitude and centered shape,
because the RF ablation showed that absolute/subcarrier information remained
useful even though cross-session generalization was weak.

EVALUATION MODES
----------------
1) Five-fold whole-trial CV on one session:
       python train_csi_cnn.py --mode 5fold --session S1

2) Cross-session:
       python train_csi_cnn.py --mode cross-session --train-session S1 --test-session S2
       python train_csi_cnn.py --mode cross-session --train-session S2 --test-session S1

SCIENTIFIC SPLITTING
--------------------
- Outer test trials/sessions are NEVER used for fitting.
- Epoch count is selected using an INNER whole-trial validation split drawn
  only from the outer training data.
- After selecting the epoch count, the model is reinitialized and retrained
  on ALL outer-training trials for exactly that many epochs.
- Normalization statistics are always computed from training data only.
- Overlapping windows from the same 8-second recording never cross a split.

DEPENDENCY
----------
Requires PyTorch in the same virtual environment:
    python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
"""

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not installed in this virtual environment.\n"
        "Install the CPU build with:\n"
        "  python -m pip install torch==2.12.1 "
        "--index-url https://download.pytorch.org/whl/cpu\n"
        "Then rerun this script."
    ) from exc

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from csi_io import load_trial_csv
from project_config import (
    CLASS_ORDER,
    HOP_SECONDS,
    MIN_PACKETS_PER_LINK,
    MODEL_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    WINDOW_SECONDS,
    ZONE_CENTERS,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode can be slower, but this project is small enough that
    # repeatability is more valuable than a tiny speed gain.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Raw CSI preprocessing
# ---------------------------------------------------------------------------

def iq_to_log_amplitude(iq: np.ndarray) -> np.ndarray:
    """
    Hernandez / ESP-IDF CSI values are imaginary, real, imaginary, real, ...
    Return log amplitude with shape [packets, 64].
    """
    x = np.asarray(iq, dtype=np.float32)
    imaginary = x[:, 0::2]
    real = x[:, 1::2]
    amplitude = np.hypot(imaginary, real)

    # Match the original engineered-feature pipeline's conservative floor.
    return np.log(np.maximum(amplitude, 0.5)).astype(np.float32)


def interpolate_matrix(
    times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Linear interpolation along time.

    values shape:
        [packets, features]

    output:
        [target_steps, features]
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    target_times = np.asarray(target_times, dtype=np.float64)

    out = np.empty(
        (len(target_times), values.shape[1]),
        dtype=np.float32,
    )

    for j in range(values.shape[1]):
        out[:, j] = np.interp(
            target_times,
            times,
            values[:, j],
        ).astype(np.float32)

    return out


def interpolate_vector(
    times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    return np.interp(
        target_times,
        times,
        values,
    ).astype(np.float32)


@dataclass
class RawWindowSet:
    X: np.ndarray                 # [N, 4, T, 64]
    rssi: np.ndarray              # [N, 4] => Lmean,Lstd,Rmean,Rstd
    y_text: np.ndarray            # [N]
    session: np.ndarray           # [N]
    trial_number: np.ndarray      # [N]
    orientation: np.ndarray       # [N]
    group: np.ndarray             # [N]
    source_file: np.ndarray       # [N]
    window_index: np.ndarray      # [N]
    window_start_s: np.ndarray    # [N]
    left_packets: np.ndarray      # [N]
    right_packets: np.ndarray     # [N]


def trial_number(meta: dict[str, Any]) -> int:
    try:
        return int(str(meta["trial"]).strip())
    except Exception as exc:
        raise ValueError(
            f"Could not parse trial number {meta.get('trial')!r} "
            f"for {meta.get('path', '<unknown>')}"
        ) from exc


def extract_trial_raw_windows(
    path: Path,
    *,
    window_seconds: float,
    hop_seconds: float,
    target_steps: int,
    min_packets_per_link: int,
) -> list[dict[str, Any]]:
    """
    Convert one collected trial CSV into fixed-size raw CSI windows.
    """
    trial = load_trial_csv(path)
    left = trial["left"]
    right = trial["right"]
    meta = trial["metadata"]

    if len(left["time"]) == 0 or len(right["time"]) == 0:
        return []

    common_start = max(
        float(left["time"][0]),
        float(right["time"][0]),
    )
    common_end = min(
        float(left["time"][-1]),
        float(right["time"][-1]),
    )

    trial_no = trial_number(meta)
    group = (
        f"{meta['session']}|{meta['zone']}|"
        f"{trial_no}|{meta['orientation']}"
    )

    rows: list[dict[str, Any]] = []

    start = common_start
    window_index = 0

    while start + window_seconds <= common_end + 1e-9:
        end = start + window_seconds

        lm = (
            (left["time"] >= start)
            & (left["time"] < end)
        )
        rm = (
            (right["time"] >= start)
            & (right["time"] < end)
        )

        nl = int(np.sum(lm))
        nr = int(np.sum(rm))

        if (
            nl >= min_packets_per_link
            and nr >= min_packets_per_link
        ):
            left_times = left["time"][lm]
            right_times = right["time"][rm]

            left_abs = iq_to_log_amplitude(
                left["iq"][lm]
            )
            right_abs = iq_to_log_amplitude(
                right["iq"][rm]
            )

            # Packet-centered spectral shape removes common-mode amplitude
            # while preserving relative frequency/subcarrier structure.
            left_centered = (
                left_abs
                - np.median(
                    left_abs,
                    axis=1,
                    keepdims=True,
                )
            ).astype(np.float32)

            right_centered = (
                right_abs
                - np.median(
                    right_abs,
                    axis=1,
                    keepdims=True,
                )
            ).astype(np.float32)

            # Fixed target points at the centers of 32 equal temporal bins.
            target_times = (
                start
                + (
                    np.arange(
                        target_steps,
                        dtype=np.float64,
                    )
                    + 0.5
                )
                * (
                    window_seconds
                    / target_steps
                )
            )

            left_abs_i = interpolate_matrix(
                left_times,
                left_abs,
                target_times,
            )
            left_center_i = interpolate_matrix(
                left_times,
                left_centered,
                target_times,
            )
            right_abs_i = interpolate_matrix(
                right_times,
                right_abs,
                target_times,
            )
            right_center_i = interpolate_matrix(
                right_times,
                right_centered,
                target_times,
            )

            X = np.stack(
                [
                    left_abs_i,
                    left_center_i,
                    right_abs_i,
                    right_center_i,
                ],
                axis=0,
            ).astype(np.float32)

            # RSSI is intentionally a small auxiliary branch rather than a
            # 2-D image channel repeated across all 64 subcarriers.
            left_rssi_i = interpolate_vector(
                left_times,
                left["rssi"][lm],
                target_times,
            )
            right_rssi_i = interpolate_vector(
                right_times,
                right["rssi"][rm],
                target_times,
            )

            rssi_summary = np.asarray(
                [
                    np.mean(left_rssi_i),
                    np.std(left_rssi_i),
                    np.mean(right_rssi_i),
                    np.std(right_rssi_i),
                ],
                dtype=np.float32,
            )

            rows.append(
                {
                    "X": X,
                    "rssi": rssi_summary,
                    "zone": str(meta["zone"]),
                    "session": str(meta["session"]),
                    "trial_number": trial_no,
                    "orientation": str(
                        meta["orientation"]
                    ),
                    "group": group,
                    "source_file": str(path),
                    "window_index": window_index,
                    "window_start_s": float(start),
                    "left_packets": nl,
                    "right_packets": nr,
                }
            )

        start += hop_seconds
        window_index += 1

    return rows


def gather_raw_windows(
    paths: list[Path],
    *,
    window_seconds: float,
    hop_seconds: float,
    target_steps: int,
    min_packets_per_link: int,
) -> RawWindowSet:
    rows: list[dict[str, Any]] = []

    for i, path in enumerate(paths, 1):
        print(
            f"[{i:03d}/{len(paths):03d}] raw CSI: "
            f"{path.name}"
        )
        rows.extend(
            extract_trial_raw_windows(
                path,
                window_seconds=window_seconds,
                hop_seconds=hop_seconds,
                target_steps=target_steps,
                min_packets_per_link=min_packets_per_link,
            )
        )

    if not rows:
        raise RuntimeError(
            "No valid raw CSI windows were created."
        )

    return RawWindowSet(
        X=np.stack(
            [row["X"] for row in rows],
            axis=0,
        ).astype(np.float32),
        rssi=np.stack(
            [row["rssi"] for row in rows],
            axis=0,
        ).astype(np.float32),
        y_text=np.asarray(
            [row["zone"] for row in rows],
            dtype=object,
        ),
        session=np.asarray(
            [row["session"] for row in rows],
            dtype=object,
        ),
        trial_number=np.asarray(
            [row["trial_number"] for row in rows],
            dtype=np.int64,
        ),
        orientation=np.asarray(
            [row["orientation"] for row in rows],
            dtype=object,
        ),
        group=np.asarray(
            [row["group"] for row in rows],
            dtype=object,
        ),
        source_file=np.asarray(
            [row["source_file"] for row in rows],
            dtype=object,
        ),
        window_index=np.asarray(
            [row["window_index"] for row in rows],
            dtype=np.int64,
        ),
        window_start_s=np.asarray(
            [row["window_start_s"] for row in rows],
            dtype=np.float64,
        ),
        left_packets=np.asarray(
            [row["left_packets"] for row in rows],
            dtype=np.int64,
        ),
        right_packets=np.asarray(
            [row["right_packets"] for row in rows],
            dtype=np.int64,
        ),
    )


def trial_paths_for_session(
    raw_dir: Path,
    session: str,
) -> list[Path]:
    session_dir = raw_dir / session
    if not session_dir.exists():
        raise SystemExit(
            f"Session directory does not exist: "
            f"{session_dir}"
        )

    paths = sorted(
        p
        for p in session_dir.rglob("*.csv")
        if not p.name.startswith("schedule_")
    )

    if not paths:
        raise SystemExit(
            f"No trial CSV files found under "
            f"{session_dir}"
        )

    return paths


# ---------------------------------------------------------------------------
# Train-only normalization
# ---------------------------------------------------------------------------

@dataclass
class Normalizer:
    csi_mean: np.ndarray   # [4]
    csi_std: np.ndarray    # [4]
    rssi_mean: np.ndarray  # [4]
    rssi_std: np.ndarray   # [4]


def compute_normalizer(
    X: np.ndarray,
    rssi: np.ndarray,
) -> Normalizer:
    csi_mean = np.mean(
        X,
        axis=(0, 2, 3),
        dtype=np.float64,
    ).astype(np.float32)

    csi_std = np.std(
        X,
        axis=(0, 2, 3),
        dtype=np.float64,
    ).astype(np.float32)

    csi_std = np.maximum(
        csi_std,
        1e-4,
    )

    rssi_mean = np.mean(
        rssi,
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    rssi_std = np.std(
        rssi,
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    rssi_std = np.maximum(
        rssi_std,
        1e-4,
    )

    return Normalizer(
        csi_mean=csi_mean,
        csi_std=csi_std,
        rssi_mean=rssi_mean,
        rssi_std=rssi_std,
    )


# ---------------------------------------------------------------------------
# Dataset + augmentation
# ---------------------------------------------------------------------------

class CSIDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        rssi: np.ndarray,
        y_index: np.ndarray,
        normalizer: Normalizer,
        *,
        augment: bool,
        augmentation_strength: float,
    ) -> None:
        self.X = X
        self.rssi = rssi
        self.y = y_index
        self.normalizer = normalizer
        self.augment = augment
        self.augmentation_strength = (
            augmentation_strength
        )

    def __len__(self) -> int:
        return len(self.y)

    def _augment(
        self,
        x: torch.Tensor,
        rssi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.augmentation_strength

        if s <= 0:
            return x, rssi

        # 1) Session/gain-like offset. Apply only to absolute channels:
        #    left absolute = channel 0, right absolute = channel 2.
        if torch.rand(()) < 0.80:
            x[0] = x[0] + torch.randn(()) * (0.20 * s)
            x[2] = x[2] + torch.randn(()) * (0.20 * s)

        # 2) Smooth subcarrier-dependent distortion. This mimics a changed
        #    frequency response while preserving local structure.
        if torch.rand(()) < 0.70:
            n_carriers = x.shape[-1]
            knots = 8
            base = (
                torch.randn(
                    1,
                    1,
                    knots,
                    dtype=x.dtype,
                )
                * (0.10 * s)
            )
            smooth = torch.nn.functional.interpolate(
                base,
                size=n_carriers,
                mode="linear",
                align_corners=True,
            )[0, 0]
            smooth = smooth - torch.mean(smooth)

            # Different distortion on each radio link.
            left_smooth = smooth
            right_smooth = (
                torch.roll(
                    smooth,
                    shifts=int(
                        torch.randint(
                            low=-3,
                            high=4,
                            size=(1,),
                        ).item()
                    ),
                )
                * (
                    0.8
                    + 0.4
                    * torch.rand(())
                )
            )

            x[0] = x[0] + left_smooth[None, :]
            x[1] = x[1] + left_smooth[None, :]
            x[2] = x[2] + right_smooth[None, :]
            x[3] = x[3] + right_smooth[None, :]

        # 3) Small temporal shift.
        if torch.rand(()) < 0.50:
            shift = int(
                torch.randint(
                    low=-2,
                    high=3,
                    size=(1,),
                ).item()
            )
            if shift != 0:
                x = torch.roll(
                    x,
                    shifts=shift,
                    dims=1,
                )

        # 4) Packet/time masking. Since normalized zero means "training mean",
        #    masking to zero is a neutral dropout rather than an extreme value.
        if torch.rand(()) < 0.35:
            n_rows = int(
                torch.randint(
                    low=1,
                    high=4,
                    size=(1,),
                ).item()
            )
            start = int(
                torch.randint(
                    low=0,
                    high=max(
                        1,
                        x.shape[1] - n_rows + 1,
                    ),
                    size=(1,),
                ).item()
            )
            x[:, start:start + n_rows, :] = 0.0

        # 5) Subcarrier masking.
        if torch.rand(()) < 0.35:
            n_cols = int(
                torch.randint(
                    low=2,
                    high=7,
                    size=(1,),
                ).item()
            )
            start = int(
                torch.randint(
                    low=0,
                    high=max(
                        1,
                        x.shape[2] - n_cols + 1,
                    ),
                    size=(1,),
                ).item()
            )
            x[:, :, start:start + n_cols] = 0.0

        # 6) Small Gaussian noise.
        if torch.rand(()) < 0.80:
            x = (
                x
                + torch.randn_like(x)
                * (0.025 * s)
            )

        # 7) RSSI perturbation so the tiny auxiliary branch cannot memorize
        #    one session's exact RSSI offsets.
        if torch.rand(()) < 0.70:
            rssi = (
                rssi
                + torch.randn_like(rssi)
                * (0.08 * s)
            )

        return x, rssi

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(
            self.X[index].copy()
        ).float()

        rssi = torch.from_numpy(
            self.rssi[index].copy()
        ).float()

        csi_mean = torch.from_numpy(
            self.normalizer.csi_mean
        ).float()[:, None, None]

        csi_std = torch.from_numpy(
            self.normalizer.csi_std
        ).float()[:, None, None]

        rssi_mean = torch.from_numpy(
            self.normalizer.rssi_mean
        ).float()

        rssi_std = torch.from_numpy(
            self.normalizer.rssi_std
        ).float()

        x = (
            (x - csi_mean)
            / csi_std
        )

        rssi = (
            (rssi - rssi_mean)
            / rssi_std
        )

        if self.augment:
            x, rssi = self._augment(
                x,
                rssi,
            )

        y = torch.tensor(
            int(self.y[index]),
            dtype=torch.long,
        )

        return x, rssi, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SmallCSICNN(nn.Module):
    """
    Small time x subcarrier CNN.

    GroupNorm is used instead of BatchNorm so running batch statistics do not
    become another session-specific state.
    """

    def __init__(
        self,
        n_classes: int,
        *,
        use_rssi: bool,
        dropout: float,
    ) -> None:
        super().__init__()

        self.use_rssi = use_rssi

        self.backbone = nn.Sequential(
            nn.Conv2d(
                4,
                32,
                kernel_size=(3, 5),
                padding=(1, 2),
                bias=False,
            ),
            nn.GroupNorm(4, 32),
            nn.GELU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(4, 32),
            nn.GELU(),

            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=(3, 5),
                padding=(1, 2),
                bias=False,
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.MaxPool2d(
                kernel_size=(2, 2),
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(
                output_size=(1, 1),
            ),
        )

        if use_rssi:
            self.rssi_branch = nn.Sequential(
                nn.Linear(4, 16),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            head_in = 128 + 16
        else:
            self.rssi_branch = None
            head_in = 128

        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        rssi: torch.Tensor,
    ) -> torch.Tensor:
        z = self.backbone(x)
        z = torch.flatten(
            z,
            start_dim=1,
        )

        if self.use_rssi:
            assert (
                self.rssi_branch
                is not None
            )
            rz = self.rssi_branch(rssi)
            z = torch.cat(
                [z, rz],
                dim=1,
            )

        return self.head(z)


# ---------------------------------------------------------------------------
# Splitting utilities
# ---------------------------------------------------------------------------

def label_to_index_map(
    y_text: np.ndarray,
) -> tuple[list[str], dict[str, int]]:
    labels = [
        label
        for label in CLASS_ORDER
        if label in set(y_text)
    ]

    missing = set(CLASS_ORDER) - set(labels)
    if missing:
        raise ValueError(
            "Dataset is missing expected classes: "
            + ", ".join(
                sorted(missing)
            )
        )

    mapping = {
        label: i
        for i, label in enumerate(labels)
    }

    return labels, mapping


def encode_labels(
    y_text: np.ndarray,
    mapping: dict[str, int],
) -> np.ndarray:
    return np.asarray(
        [
            mapping[str(label)]
            for label in y_text
        ],
        dtype=np.int64,
    )


def choose_inner_validation_groups(
    y_text: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> set[str]:
    """
    Choose ONE whole trial per class from the available outer-training data.
    This inner split is used only to select the epoch count.
    """
    rng = np.random.default_rng(seed)
    val_groups: set[str] = set()

    for label in CLASS_ORDER:
        class_groups = sorted(
            set(
                groups[
                    y_text == label
                ]
            )
        )

        if len(class_groups) < 2:
            raise ValueError(
                f"Need at least 2 outer-training trials "
                f"for class {label}; found "
                f"{len(class_groups)}."
            )

        chosen = class_groups[
            int(
                rng.integers(
                    0,
                    len(class_groups),
                )
            )
        ]
        val_groups.add(chosen)

    return val_groups


def mask_for_groups(
    groups: np.ndarray,
    selected: set[str],
) -> np.ndarray:
    return np.asarray(
        [
            str(group) in selected
            for group in groups
        ],
        dtype=bool,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    batch_size: int
    max_epochs: int
    patience: int
    lr: float
    weight_decay: float
    label_smoothing: float
    dropout: float
    augmentation_strength: float
    use_rssi: bool
    device: str
    seed: int


def make_loader(
    X: np.ndarray,
    rssi: np.ndarray,
    y_index: np.ndarray,
    normalizer: Normalizer,
    *,
    augment: bool,
    cfg: TrainingConfig,
    shuffle: bool,
) -> DataLoader:
    dataset = CSIDataset(
        X,
        rssi,
        y_index,
        normalizer,
        augment=augment,
        augmentation_strength=(
            cfg.augmentation_strength
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,  # safest on Windows
        pin_memory=(
            cfg.device.startswith(
                "cuda"
            )
        ),
        generator=generator,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, rssi, y in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        rssi = rssi.to(
            device,
            non_blocking=True,
        )
        y = y.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        logits = model(
            x,
            rssi,
        )

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += (
            float(loss.item())
            * len(y)
        )

        total_correct += int(
            torch.sum(
                torch.argmax(
                    logits,
                    dim=1,
                )
                == y
            ).item()
        )

        total_count += len(y)

    return (
        total_loss
        / max(
            1,
            total_count,
        ),
        total_correct
        / max(
            1,
            total_count,
        ),
    )


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    preds: list[np.ndarray] = []

    for x, rssi, y in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        rssi = rssi.to(
            device,
            non_blocking=True,
        )

        logits = model(
            x,
            rssi,
        )

        probability = torch.softmax(
            logits,
            dim=1,
        )

        prediction = torch.argmax(
            probability,
            dim=1,
        )

        ys.append(
            y.numpy()
        )
        probs.append(
            probability.cpu().numpy()
        )
        preds.append(
            prediction.cpu().numpy()
        )

    return (
        np.concatenate(
            ys,
            axis=0,
        ),
        np.concatenate(
            preds,
            axis=0,
        ),
        np.concatenate(
            probs,
            axis=0,
        ),
    )


def macro_f1_from_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(
                range(n_classes)
            ),
            average="macro",
            zero_division=0,
        )
    )


def select_epoch_count(
    X: np.ndarray,
    rssi: np.ndarray,
    y_index: np.ndarray,
    y_text: np.ndarray,
    groups: np.ndarray,
    *,
    cfg: TrainingConfig,
    seed: int,
    n_classes: int,
) -> tuple[int, dict[str, Any]]:
    """
    Inner whole-trial validation selects the epoch count.
    """
    val_groups = (
        choose_inner_validation_groups(
            y_text,
            groups,
            seed=seed,
        )
    )

    val_mask = mask_for_groups(
        groups,
        val_groups,
    )
    train_mask = ~val_mask

    overlap = (
        set(groups[train_mask])
        & set(groups[val_mask])
    )
    if overlap:
        raise RuntimeError(
            "Inner train/validation group leakage "
            "detected."
        )

    normalizer = compute_normalizer(
        X[train_mask],
        rssi[train_mask],
    )

    train_loader = make_loader(
        X[train_mask],
        rssi[train_mask],
        y_index[train_mask],
        normalizer,
        augment=True,
        cfg=cfg,
        shuffle=True,
    )

    val_loader = make_loader(
        X[val_mask],
        rssi[val_mask],
        y_index[val_mask],
        normalizer,
        augment=False,
        cfg=cfg,
        shuffle=False,
    )

    set_seed(seed)

    device = torch.device(
        cfg.device
    )

    model = SmallCSICNN(
        n_classes,
        use_rssi=cfg.use_rssi,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=(
            cfg.label_smoothing
        )
    )

    best_epoch = 1
    best_f1 = -1.0
    best_val_accuracy = -1.0
    patience_left = cfg.patience
    history: list[dict[str, float]] = []

    for epoch in range(
        1,
        cfg.max_epochs + 1,
    ):
        train_loss, train_acc = (
            train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )
        )

        y_val, pred_val, _ = (
            predict_loader(
                model,
                val_loader,
                device,
            )
        )

        val_acc = float(
            accuracy_score(
                y_val,
                pred_val,
            )
        )

        val_f1 = (
            macro_f1_from_indices(
                y_val,
                pred_val,
                n_classes,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        improved = (
            val_f1
            > best_f1 + 1e-6
            or (
                abs(
                    val_f1
                    - best_f1
                )
                <= 1e-6
                and val_acc
                > best_val_accuracy
            )
        )

        if improved:
            best_f1 = val_f1
            best_val_accuracy = val_acc
            best_epoch = epoch
            patience_left = cfg.patience
        else:
            patience_left -= 1

        print(
            f"    epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} "
            f"acc={train_acc:.3f} | "
            f"val acc={val_acc:.3f} "
            f"macroF1={val_f1:.3f}"
        )

        if patience_left <= 0:
            print(
                f"    early stop; best epoch "
                f"was {best_epoch}"
            )
            break

    diagnostics = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_accuracy": (
            best_val_accuracy
        ),
        "inner_train_trials": int(
            len(
                set(
                    groups[train_mask]
                )
            )
        ),
        "inner_val_trials": int(
            len(
                set(
                    groups[val_mask]
                )
            )
        ),
        "inner_validation_groups": sorted(
            val_groups
        ),
        "history": history,
    }

    return best_epoch, diagnostics


def fit_full_training_data(
    X: np.ndarray,
    rssi: np.ndarray,
    y_index: np.ndarray,
    *,
    n_classes: int,
    epochs: int,
    cfg: TrainingConfig,
    seed: int,
) -> tuple[SmallCSICNN, Normalizer, list[dict[str, float]]]:
    """
    Reinitialize and train on ALL outer-training data for the epoch count chosen
    by the inner validation split.
    """
    normalizer = compute_normalizer(
        X,
        rssi,
    )

    train_loader = make_loader(
        X,
        rssi,
        y_index,
        normalizer,
        augment=True,
        cfg=cfg,
        shuffle=True,
    )

    set_seed(seed)

    device = torch.device(
        cfg.device
    )

    model = SmallCSICNN(
        n_classes,
        use_rssi=cfg.use_rssi,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=(
            cfg.label_smoothing
        )
    )

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        loss, accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "train_accuracy": accuracy,
            }
        )

        print(
            f"    refit epoch {epoch:03d}/{epochs:03d} | "
            f"loss={loss:.4f} "
            f"acc={accuracy:.3f}"
        )

    return (
        model,
        normalizer,
        history,
    )


def predict_arrays(
    model: nn.Module,
    normalizer: Normalizer,
    X: np.ndarray,
    rssi: np.ndarray,
    y_index: np.ndarray,
    *,
    cfg: TrainingConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    loader = make_loader(
        X,
        rssi,
        y_index,
        normalizer,
        augment=False,
        cfg=cfg,
        shuffle=False,
    )

    return predict_loader(
        model,
        loader,
        torch.device(
            cfg.device
        ),
    )


# ---------------------------------------------------------------------------
# Metrics + plots
# ---------------------------------------------------------------------------

def index_to_text(
    indices: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            labels[int(i)]
            for i in indices
        ],
        dtype=object,
    )


def classification_metrics(
    y_true_text: np.ndarray,
    y_pred_text: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(
                y_true_text,
                y_pred_text,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true_text,
                y_pred_text,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true_text,
                y_pred_text,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
    }


def probability_weighted_xy(
    probabilities: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    indices = []
    centers = []

    for i, label in enumerate(labels):
        if label in ZONE_CENTERS:
            indices.append(i)
            centers.append(
                ZONE_CENTERS[label]
            )

    if not indices:
        return np.full(
            (
                len(probabilities),
                2,
            ),
            np.nan,
            dtype=np.float64,
        )

    centers_array = np.asarray(
        centers,
        dtype=np.float64,
    )

    p = probabilities[:, indices]
    p = p / np.maximum(
        np.sum(
            p,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )

    return p @ centers_array


def xy_errors(
    y_true_text: np.ndarray,
    xy_pred: np.ndarray,
) -> np.ndarray:
    errors = np.full(
        len(y_true_text),
        np.nan,
        dtype=np.float64,
    )

    for i, (
        label,
        pred,
    ) in enumerate(
        zip(
            y_true_text,
            xy_pred,
        )
    ):
        if label not in ZONE_CENTERS:
            continue

        target = np.asarray(
            ZONE_CENTERS[label],
            dtype=np.float64,
        )

        errors[i] = float(
            np.linalg.norm(
                pred - target
            )
        )

    return errors


def summarize_xy(
    errors: np.ndarray,
) -> dict[str, float | None]:
    x = errors[
        np.isfinite(errors)
    ]

    if len(x) == 0:
        return {
            "xy_mean_m": None,
            "xy_median_m": None,
            "xy_p90_m": None,
        }

    return {
        "xy_mean_m": float(
            np.mean(x)
        ),
        "xy_median_m": float(
            np.median(x)
        ),
        "xy_p90_m": float(
            np.percentile(
                x,
                90,
            )
        ),
    }


def majority_vote_with_probability_tiebreak(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> str:
    counts = Counter(
        predictions.tolist()
    )
    best_count = max(
        counts.values()
    )
    tied = [
        label
        for label, count
        in counts.items()
        if count == best_count
    ]

    if len(tied) == 1:
        return tied[0]

    mean_prob = np.mean(
        probabilities,
        axis=0,
    )

    label_index = {
        label: i
        for i, label
        in enumerate(labels)
    }

    return max(
        tied,
        key=lambda label:
            mean_prob[
                label_index[label]
            ],
    )


def aggregate_trials(
    y_true_text: np.ndarray,
    pred_text: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    orientations: np.ndarray,
    source_files: np.ndarray,
    trial_numbers: np.ndarray,
    labels: list[str],
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows: list[dict[str, Any]] = []
    trial_true = []
    trial_majority = []
    trial_mean_prob = []

    for group in sorted(
        set(groups)
    ):
        idx = np.where(
            groups == group
        )[0]

        true_label = str(
            y_true_text[
                idx[0]
            ]
        )

        group_pred = pred_text[idx]
        group_prob = probabilities[idx]
        mean_prob = np.mean(
            group_prob,
            axis=0,
        )

        majority = (
            majority_vote_with_probability_tiebreak(
                group_pred,
                group_prob,
                labels,
            )
        )

        mean_prob_pred = labels[
            int(
                np.argmax(
                    mean_prob
                )
            )
        ]

        xy = probability_weighted_xy(
            mean_prob[
                None,
                :
            ],
            labels,
        )[0]

        err = xy_errors(
            np.asarray(
                [true_label],
                dtype=object,
            ),
            xy[
                None,
                :
            ],
        )[0]

        row = {
            "trial_group": str(group),
            "source_file": str(
                source_files[
                    idx[0]
                ]
            ),
            "zone_true": true_label,
            "trial": int(
                trial_numbers[
                    idx[0]
                ]
            ),
            "orientation": str(
                orientations[
                    idx[0]
                ]
            ),
            "n_windows": int(
                len(idx)
            ),
            "pred_majority": majority,
            "pred_mean_probability": (
                mean_prob_pred
            ),
            "correct_majority": int(
                majority
                == true_label
            ),
            "correct_mean_probability": int(
                mean_prob_pred
                == true_label
            ),
            "x_pred_m": float(
                xy[0]
            ),
            "y_pred_m": float(
                xy[1]
            ),
            "xy_error_m": (
                None
                if not np.isfinite(err)
                else float(err)
            ),
        }

        for i, label in enumerate(
            labels
        ):
            row[
                f"p_{label}"
            ] = float(
                mean_prob[i]
            )

        rows.append(row)
        trial_true.append(
            true_label
        )
        trial_majority.append(
            majority
        )
        trial_mean_prob.append(
            mean_prob_pred
        )

    return (
        rows,
        np.asarray(
            trial_true,
            dtype=object,
        ),
        np.asarray(
            trial_majority,
            dtype=object,
        ),
        np.asarray(
            trial_mean_prob,
            dtype=object,
        ),
    )


def save_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
    *,
    normalize: bool = False,
) -> np.ndarray:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    if normalize:
        denominator = cm.sum(
            axis=1,
            keepdims=True,
        )

        shown = np.divide(
            cm.astype(
                np.float64
            ),
            denominator,
            out=np.zeros_like(
                cm,
                dtype=np.float64,
            ),
            where=(
                denominator
                != 0
            ),
        )
    else:
        shown = cm

    fig, ax = plt.subplots(
        figsize=(9.5, 8.5)
    )

    image = ax.imshow(
        shown
    )

    ax.set_title(title)
    ax.set_xlabel(
        "Predicted zone"
    )
    ax.set_ylabel(
        "True zone"
    )

    ax.set_xticks(
        range(
            len(labels)
        ),
        labels=labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(
            len(labels)
        ),
        labels=labels,
    )

    for i in range(
        len(labels)
    ):
        for j in range(
            len(labels)
        ):
            if normalize:
                text = (
                    f"{shown[i, j]:.2f}"
                )
            else:
                text = str(
                    int(
                        cm[i, j]
                    )
                )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )

    plt.close(fig)
    plt.close("all")

    return cm


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(
            rows[0].keys()
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def per_zone_recall_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> list[dict[str, Any]]:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    rows = []

    for i, label in enumerate(
        labels
    ):
        support = int(
            np.sum(
                cm[i]
            )
        )

        rows.append(
            {
                "zone": label,
                "correct": int(
                    cm[i, i]
                ),
                "support": support,
                "recall": (
                    float(
                        cm[i, i]
                        / support
                    )
                    if support
                    else None
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# One outer train/test experiment
# ---------------------------------------------------------------------------

def run_outer_experiment(
    data: RawWindowSet,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    labels: list[str],
    label_map: dict[str, int],
    cfg: TrainingConfig,
    seed: int,
    run_name: str,
    save_model_path: Path | None,
) -> dict[str, Any]:
    train_groups = set(
        data.group[
            train_mask
        ]
    )
    test_groups = set(
        data.group[
            test_mask
        ]
    )

    overlap = (
        train_groups
        & test_groups
    )
    if overlap:
        raise RuntimeError(
            f"Outer split leakage in {run_name}: "
            f"{sorted(overlap)[:3]}"
        )

    y_all_index = encode_labels(
        data.y_text,
        label_map,
    )

    X_train = data.X[
        train_mask
    ]
    rssi_train = data.rssi[
        train_mask
    ]
    y_train_index = y_all_index[
        train_mask
    ]
    y_train_text = data.y_text[
        train_mask
    ]
    groups_train = data.group[
        train_mask
    ]

    print(
        f"\n[{run_name}] "
        f"selecting epoch count with an INNER "
        f"whole-trial validation split..."
    )

    best_epoch, inner_diag = (
        select_epoch_count(
            X_train,
            rssi_train,
            y_train_index,
            y_train_text,
            groups_train,
            cfg=cfg,
            seed=seed,
            n_classes=len(labels),
        )
    )

    print(
        f"[{run_name}] selected "
        f"{best_epoch} training epoch(s)."
    )

    print(
        f"[{run_name}] reinitializing and "
        f"training on ALL {len(train_groups)} "
        f"outer-training trials..."
    )

    model, normalizer, refit_history = (
        fit_full_training_data(
            X_train,
            rssi_train,
            y_train_index,
            n_classes=len(labels),
            epochs=best_epoch,
            cfg=cfg,
            seed=seed + 1000,
        )
    )

    y_test_index = y_all_index[
        test_mask
    ]

    (
        y_test_index_returned,
        pred_index,
        probabilities,
    ) = predict_arrays(
        model,
        normalizer,
        data.X[
            test_mask
        ],
        data.rssi[
            test_mask
        ],
        y_test_index,
        cfg=cfg,
    )

    if not np.array_equal(
        y_test_index_returned,
        y_test_index,
    ):
        raise RuntimeError(
            "Test loader changed sample order."
        )

    y_true_text = data.y_text[
        test_mask
    ]

    pred_text = index_to_text(
        pred_index,
        labels,
    )

    window_metrics = (
        classification_metrics(
            y_true_text,
            pred_text,
        )
    )

    xy_pred = probability_weighted_xy(
        probabilities,
        labels,
    )

    window_xy_errors = xy_errors(
        y_true_text,
        xy_pred,
    )

    window_metrics.update(
        summarize_xy(
            window_xy_errors
        )
    )

    (
        trial_rows,
        trial_true,
        trial_majority,
        trial_mean_prob,
    ) = aggregate_trials(
        y_true_text,
        pred_text,
        probabilities,
        data.group[
            test_mask
        ],
        data.orientation[
            test_mask
        ],
        data.source_file[
            test_mask
        ],
        data.trial_number[
            test_mask
        ],
        labels,
    )

    trial_majority_metrics = (
        classification_metrics(
            trial_true,
            trial_majority,
        )
    )

    trial_mean_prob_metrics = (
        classification_metrics(
            trial_true,
            trial_mean_prob,
        )
    )

    trial_xy_values = np.asarray(
        [
            (
                np.nan
                if row[
                    "xy_error_m"
                ] is None
                else float(
                    row[
                        "xy_error_m"
                    ]
                )
            )
            for row
            in trial_rows
        ],
        dtype=np.float64,
    )

    trial_mean_prob_metrics.update(
        summarize_xy(
            trial_xy_values
        )
    )

    if save_model_path is not None:
        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "model_state_dict": (
                model.cpu().state_dict()
            ),
            "labels": labels,
            "normalizer": {
                "csi_mean": (
                    normalizer
                    .csi_mean
                    .tolist()
                ),
                "csi_std": (
                    normalizer
                    .csi_std
                    .tolist()
                ),
                "rssi_mean": (
                    normalizer
                    .rssi_mean
                    .tolist()
                ),
                "rssi_std": (
                    normalizer
                    .rssi_std
                    .tolist()
                ),
            },
            "model_config": {
                "use_rssi": (
                    cfg.use_rssi
                ),
                "dropout": (
                    cfg.dropout
                ),
                "target_steps": int(
                    data.X.shape[2]
                ),
                "subcarriers": int(
                    data.X.shape[3]
                ),
                "input_channels": int(
                    data.X.shape[1]
                ),
            },
            "training": {
                "selected_epochs": (
                    best_epoch
                ),
                "run_name": run_name,
                "outer_train_trials": (
                    len(train_groups)
                ),
            },
        }

        torch.save(
            checkpoint,
            save_model_path,
        )

        model.to(
            torch.device(
                cfg.device
            )
        )

    return {
        "run_name": run_name,
        "selected_epochs": (
            best_epoch
        ),
        "inner_validation": (
            inner_diag
        ),
        "refit_history": (
            refit_history
        ),
        "train_trials": int(
            len(train_groups)
        ),
        "test_trials": int(
            len(test_groups)
        ),
        "train_windows": int(
            np.sum(train_mask)
        ),
        "test_windows": int(
            np.sum(test_mask)
        ),
        "window_metrics": (
            window_metrics
        ),
        "trial_majority_metrics": (
            trial_majority_metrics
        ),
        "trial_mean_probability_metrics": (
            trial_mean_prob_metrics
        ),
        "y_true_text": (
            y_true_text
        ),
        "pred_text": (
            pred_text
        ),
        "probabilities": (
            probabilities
        ),
        "window_xy_errors": (
            window_xy_errors
        ),
        "trial_rows": (
            trial_rows
        ),
        "trial_true": (
            trial_true
        ),
        "trial_majority": (
            trial_majority
        ),
        "trial_mean_probability": (
            trial_mean_prob
        ),
        "test_indices": np.where(
            test_mask
        )[0],
    }


# ---------------------------------------------------------------------------
# Five-fold CV
# ---------------------------------------------------------------------------

def run_5fold(
    data: RawWindowSet,
    *,
    session: str,
    cfg: TrainingConfig,
    output_dir: Path,
) -> None:
    labels, label_map = (
        label_to_index_map(
            data.y_text
        )
    )

    print("\nTRIAL/FOLD VALIDATION")
    for label in labels:
        trials = sorted(
            set(
                data.trial_number[
                    data.y_text
                    == label
                ]
            )
        )

        print(
            f"  {label:5s}: "
            f"{trials}"
        )

        if trials != [
            1,
            2,
            3,
            4,
            5,
        ]:
            raise SystemExit(
                f"{label}: expected trials "
                f"[1,2,3,4,5], found "
                f"{trials}"
            )

    fold_rows = []
    oof_window_rows = []
    oof_trial_rows = []

    all_true = []
    all_pred = []
    all_prob = []
    all_trial_true = []
    all_trial_majority = []
    all_trial_mean_prob = []

    for fold in range(
        1,
        6,
    ):
        print(
            "\n"
            + "=" * 78
        )
        print(
            f"CNN FOLD {fold}/5 — "
            f"hold out trial {fold}"
        )
        print(
            "=" * 78
        )

        test_mask = (
            data.trial_number
            == fold
        )
        train_mask = ~test_mask

        result = run_outer_experiment(
            data,
            train_mask,
            test_mask,
            labels=labels,
            label_map=label_map,
            cfg=cfg,
            seed=(
                cfg.seed
                + fold * 17
            ),
            run_name=f"fold_{fold}",
            save_model_path=None,
        )

        wm = result[
            "window_metrics"
        ]
        tm = result[
            "trial_mean_probability_metrics"
        ]

        print(
            f"\nFold {fold} RESULT | "
            f"window acc={wm['accuracy']:.3f} "
            f"macroF1={wm['macro_f1']:.3f} "
            f"medianXY={wm['xy_median_m']:.3f} m | "
            f"trial mean-prob acc="
            f"{tm['accuracy']:.3f}"
        )

        fold_rows.append(
            {
                "fold": fold,
                "selected_epochs": (
                    result[
                        "selected_epochs"
                    ]
                ),
                "window_accuracy": (
                    wm[
                        "accuracy"
                    ]
                ),
                "window_balanced_accuracy": (
                    wm[
                        "balanced_accuracy"
                    ]
                ),
                "window_macro_f1": (
                    wm[
                        "macro_f1"
                    ]
                ),
                "window_xy_median_m": (
                    wm[
                        "xy_median_m"
                    ]
                ),
                "window_xy_p90_m": (
                    wm[
                        "xy_p90_m"
                    ]
                ),
                "trial_majority_accuracy": (
                    result[
                        "trial_majority_metrics"
                    ][
                        "accuracy"
                    ]
                ),
                "trial_mean_probability_accuracy": (
                    tm[
                        "accuracy"
                    ]
                ),
                "trial_mean_probability_macro_f1": (
                    tm[
                        "macro_f1"
                    ]
                ),
            }
        )

        test_idx = result[
            "test_indices"
        ]

        for local_i, global_i in enumerate(
            test_idx
        ):
            row = {
                "fold": fold,
                "zone_true": str(
                    result[
                        "y_true_text"
                    ][
                        local_i
                    ]
                ),
                "zone_pred": str(
                    result[
                        "pred_text"
                    ][
                        local_i
                    ]
                ),
                "correct": int(
                    result[
                        "y_true_text"
                    ][
                        local_i
                    ]
                    == result[
                        "pred_text"
                    ][
                        local_i
                    ]
                ),
                "trial": int(
                    data.trial_number[
                        global_i
                    ]
                ),
                "orientation": str(
                    data.orientation[
                        global_i
                    ]
                ),
                "trial_group": str(
                    data.group[
                        global_i
                    ]
                ),
                "source_file": str(
                    data.source_file[
                        global_i
                    ]
                ),
                "window_index": int(
                    data.window_index[
                        global_i
                    ]
                ),
                "window_start_s": float(
                    data.window_start_s[
                        global_i
                    ]
                ),
                "xy_error_m": (
                    None
                    if not np.isfinite(
                        result[
                            "window_xy_errors"
                        ][
                            local_i
                        ]
                    )
                    else float(
                        result[
                            "window_xy_errors"
                        ][
                            local_i
                        ]
                    )
                ),
            }

            for class_i, label in enumerate(
                labels
            ):
                row[
                    f"p_{label}"
                ] = float(
                    result[
                        "probabilities"
                    ][
                        local_i,
                        class_i,
                    ]
                )

            oof_window_rows.append(
                row
            )

        for row in result[
            "trial_rows"
        ]:
            oof_trial_rows.append(
                {
                    "fold": fold,
                    **row,
                }
            )

        all_true.extend(
            result[
                "y_true_text"
            ].tolist()
        )
        all_pred.extend(
            result[
                "pred_text"
            ].tolist()
        )
        all_prob.append(
            result[
                "probabilities"
            ]
        )

        all_trial_true.extend(
            result[
                "trial_true"
            ].tolist()
        )
        all_trial_majority.extend(
            result[
                "trial_majority"
            ].tolist()
        )
        all_trial_mean_prob.extend(
            result[
                "trial_mean_probability"
            ].tolist()
        )

    all_true_arr = np.asarray(
        all_true,
        dtype=object,
    )
    all_pred_arr = np.asarray(
        all_pred,
        dtype=object,
    )

    all_trial_true_arr = np.asarray(
        all_trial_true,
        dtype=object,
    )
    all_trial_majority_arr = np.asarray(
        all_trial_majority,
        dtype=object,
    )
    all_trial_mean_prob_arr = np.asarray(
        all_trial_mean_prob,
        dtype=object,
    )

    aggregate_window = (
        classification_metrics(
            all_true_arr,
            all_pred_arr,
        )
    )

    aggregate_trial_majority = (
        classification_metrics(
            all_trial_true_arr,
            all_trial_majority_arr,
        )
    )

    aggregate_trial_mean_prob = (
        classification_metrics(
            all_trial_true_arr,
            all_trial_mean_prob_arr,
        )
    )

    fold_accuracy = np.asarray(
        [
            row[
                "window_accuracy"
            ]
            for row
            in fold_rows
        ],
        dtype=np.float64,
    )

    fold_f1 = np.asarray(
        [
            row[
                "window_macro_f1"
            ]
            for row
            in fold_rows
        ],
        dtype=np.float64,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_dir
        / "fold_metrics.csv",
        fold_rows,
    )

    write_csv(
        output_dir
        / "oof_window_predictions.csv",
        oof_window_rows,
    )

    write_csv(
        output_dir
        / "oof_trial_predictions.csv",
        oof_trial_rows,
    )

    save_confusion(
        all_true_arr,
        all_pred_arr,
        labels,
        output_dir
        / "confusion_matrix_window.png",
        f"Raw CSI CNN 5-fold OOF — "
        f"{session} (window)",
    )

    save_confusion(
        all_true_arr,
        all_pred_arr,
        labels,
        output_dir
        / "confusion_matrix_window_normalized.png",
        f"Raw CSI CNN 5-fold OOF — "
        f"{session} normalized",
        normalize=True,
    )

    save_confusion(
        all_trial_true_arr,
        all_trial_mean_prob_arr,
        labels,
        output_dir
        / "confusion_matrix_trial_mean_probability.png",
        f"Raw CSI CNN 5-fold OOF — "
        f"{session} (trial mean probability)",
    )

    write_csv(
        output_dir
        / "per_zone_recall_window.csv",
        per_zone_recall_rows(
            all_true_arr,
            all_pred_arr,
            labels,
        ),
    )

    write_csv(
        output_dir
        / "per_zone_recall_trial.csv",
        per_zone_recall_rows(
            all_trial_true_arr,
            all_trial_mean_prob_arr,
            labels,
        ),
    )

    summary = {
        "mode": "5fold",
        "session": session,
        "model": "SmallCSICNN",
        "input_shape": [
            4,
            int(
                data.X.shape[2]
            ),
            int(
                data.X.shape[3]
            ),
        ],
        "use_rssi_aux": (
            cfg.use_rssi
        ),
        "augmentation_strength": (
            cfg.augmentation_strength
        ),
        "fold_window_accuracy_mean": (
            float(
                np.mean(
                    fold_accuracy
                )
            )
        ),
        "fold_window_accuracy_std": (
            float(
                np.std(
                    fold_accuracy,
                    ddof=1,
                )
            )
        ),
        "fold_window_macro_f1_mean": (
            float(
                np.mean(
                    fold_f1
                )
            )
        ),
        "fold_window_macro_f1_std": (
            float(
                np.std(
                    fold_f1,
                    ddof=1,
                )
            )
        ),
        "aggregate_oof_window": (
            aggregate_window
        ),
        "aggregate_oof_trial_majority": (
            aggregate_trial_majority
        ),
        "aggregate_oof_trial_mean_probability": (
            aggregate_trial_mean_prob
        ),
        "folds": fold_rows,
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "RAW-CSI CNN 5-FOLD SUMMARY"
    )
    print(
        "=" * 78
    )
    print(
        f"Window accuracy: "
        f"{np.mean(fold_accuracy):.3f} "
        f"± {np.std(fold_accuracy, ddof=1):.3f}"
    )
    print(
        f"Window macro F1: "
        f"{np.mean(fold_f1):.3f} "
        f"± {np.std(fold_f1, ddof=1):.3f}"
    )
    print(
        f"Aggregate OOF window accuracy: "
        f"{aggregate_window['accuracy']:.3f}"
    )
    print(
        f"Aggregate OOF trial majority: "
        f"{aggregate_trial_majority['accuracy']:.3f}"
    )
    print(
        f"Aggregate OOF trial mean probability: "
        f"{aggregate_trial_mean_prob['accuracy']:.3f}"
    )
    print(
        f"\nSaved: {output_dir}"
    )


# ---------------------------------------------------------------------------
# Cross-session
# ---------------------------------------------------------------------------

def combine_raw_sets(
    a: RawWindowSet,
    b: RawWindowSet,
) -> RawWindowSet:
    return RawWindowSet(
        X=np.concatenate(
            [a.X, b.X],
            axis=0,
        ),
        rssi=np.concatenate(
            [a.rssi, b.rssi],
            axis=0,
        ),
        y_text=np.concatenate(
            [a.y_text, b.y_text],
            axis=0,
        ),
        session=np.concatenate(
            [a.session, b.session],
            axis=0,
        ),
        trial_number=np.concatenate(
            [
                a.trial_number,
                b.trial_number,
            ],
            axis=0,
        ),
        orientation=np.concatenate(
            [
                a.orientation,
                b.orientation,
            ],
            axis=0,
        ),
        group=np.concatenate(
            [a.group, b.group],
            axis=0,
        ),
        source_file=np.concatenate(
            [
                a.source_file,
                b.source_file,
            ],
            axis=0,
        ),
        window_index=np.concatenate(
            [
                a.window_index,
                b.window_index,
            ],
            axis=0,
        ),
        window_start_s=np.concatenate(
            [
                a.window_start_s,
                b.window_start_s,
            ],
            axis=0,
        ),
        left_packets=np.concatenate(
            [
                a.left_packets,
                b.left_packets,
            ],
            axis=0,
        ),
        right_packets=np.concatenate(
            [
                a.right_packets,
                b.right_packets,
            ],
            axis=0,
        ),
    )


def run_cross_session(
    train_data: RawWindowSet,
    test_data: RawWindowSet,
    *,
    train_session: str,
    test_session: str,
    cfg: TrainingConfig,
    output_dir: Path,
) -> None:
    data = combine_raw_sets(
        train_data,
        test_data,
    )

    labels, label_map = (
        label_to_index_map(
            data.y_text
        )
    )

    train_mask = (
        data.session
        == train_session
    )
    test_mask = (
        data.session
        == test_session
    )

    if (
        not np.any(
            train_mask
        )
        or not np.any(
            test_mask
        )
    ):
        raise RuntimeError(
            "Cross-session masks are empty."
        )

    save_model_path = (
        MODEL_DIR
        / (
            f"csi_cnn_"
            f"{train_session.lower()}_to_"
            f"{test_session.lower()}.pt"
        )
    )

    result = run_outer_experiment(
        data,
        train_mask,
        test_mask,
        labels=labels,
        label_map=label_map,
        cfg=cfg,
        seed=cfg.seed + 701,
        run_name=(
            f"{train_session}_to_"
            f"{test_session}"
        ),
        save_model_path=(
            save_model_path
        ),
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    wm = result[
        "window_metrics"
    ]
    tm = result[
        "trial_mean_probability_metrics"
    ]

    save_confusion(
        result[
            "y_true_text"
        ],
        result[
            "pred_text"
        ],
        labels,
        output_dir
        / "confusion_matrix_window.png",
        f"Raw CSI CNN "
        f"{train_session}->{test_session} "
        f"(window)",
    )

    save_confusion(
        result[
            "y_true_text"
        ],
        result[
            "pred_text"
        ],
        labels,
        output_dir
        / "confusion_matrix_window_normalized.png",
        f"Raw CSI CNN "
        f"{train_session}->{test_session} "
        f"normalized",
        normalize=True,
    )

    save_confusion(
        result[
            "trial_true"
        ],
        result[
            "trial_mean_probability"
        ],
        labels,
        output_dir
        / "confusion_matrix_trial_mean_probability.png",
        f"Raw CSI CNN "
        f"{train_session}->{test_session} "
        f"(trial mean probability)",
    )

    write_csv(
        output_dir
        / "per_zone_recall_window.csv",
        per_zone_recall_rows(
            result[
                "y_true_text"
            ],
            result[
                "pred_text"
            ],
            labels,
        ),
    )

    write_csv(
        output_dir
        / "per_zone_recall_trial.csv",
        per_zone_recall_rows(
            result[
                "trial_true"
            ],
            result[
                "trial_mean_probability"
            ],
            labels,
        ),
    )

    write_csv(
        output_dir
        / "trial_predictions.csv",
        result[
            "trial_rows"
        ],
    )

    test_indices = result[
        "test_indices"
    ]

    window_rows = []

    for local_i, global_i in enumerate(
        test_indices
    ):
        row = {
            "zone_true": str(
                result[
                    "y_true_text"
                ][
                    local_i
                ]
            ),
            "zone_pred": str(
                result[
                    "pred_text"
                ][
                    local_i
                ]
            ),
            "correct": int(
                result[
                    "y_true_text"
                ][
                    local_i
                ]
                == result[
                    "pred_text"
                ][
                    local_i
                ]
            ),
            "trial": int(
                data.trial_number[
                    global_i
                ]
            ),
            "orientation": str(
                data.orientation[
                    global_i
                ]
            ),
            "trial_group": str(
                data.group[
                    global_i
                ]
            ),
            "source_file": str(
                data.source_file[
                    global_i
                ]
            ),
            "window_index": int(
                data.window_index[
                    global_i
                ]
            ),
            "window_start_s": float(
                data.window_start_s[
                    global_i
                ]
            ),
            "xy_error_m": (
                None
                if not np.isfinite(
                    result[
                        "window_xy_errors"
                    ][
                        local_i
                    ]
                )
                else float(
                    result[
                        "window_xy_errors"
                    ][
                        local_i
                    ]
                )
            ),
        }

        for class_i, label in enumerate(
            labels
        ):
            row[
                f"p_{label}"
            ] = float(
                result[
                    "probabilities"
                ][
                    local_i,
                    class_i,
                ]
            )

        window_rows.append(row)

    write_csv(
        output_dir
        / "window_predictions.csv",
        window_rows,
    )

    (
        output_dir
        / "classification_report_window.txt"
    ).write_text(
        classification_report(
            result[
                "y_true_text"
            ],
            result[
                "pred_text"
            ],
            labels=labels,
            zero_division=0,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "classification_report_trial.txt"
    ).write_text(
        classification_report(
            result[
                "trial_true"
            ],
            result[
                "trial_mean_probability"
            ],
            labels=labels,
            zero_division=0,
        ),
        encoding="utf-8",
    )

    summary = {
        "mode": "cross-session",
        "train_session": (
            train_session
        ),
        "test_session": (
            test_session
        ),
        "model": "SmallCSICNN",
        "input_shape": [
            4,
            int(
                data.X.shape[2]
            ),
            int(
                data.X.shape[3]
            ),
        ],
        "use_rssi_aux": (
            cfg.use_rssi
        ),
        "augmentation_strength": (
            cfg.augmentation_strength
        ),
        "selected_epochs": (
            result[
                "selected_epochs"
            ]
        ),
        "train_trials": (
            result[
                "train_trials"
            ]
        ),
        "test_trials": (
            result[
                "test_trials"
            ]
        ),
        "train_windows": (
            result[
                "train_windows"
            ]
        ),
        "test_windows": (
            result[
                "test_windows"
            ]
        ),
        "window_metrics": (
            wm
        ),
        "trial_majority_metrics": (
            result[
                "trial_majority_metrics"
            ]
        ),
        "trial_mean_probability_metrics": (
            tm
        ),
        "inner_validation": (
            result[
                "inner_validation"
            ]
        ),
        "saved_model": str(
            save_model_path
        ),
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "RAW-CSI CNN CROSS-SESSION RESULTS"
    )
    print(
        "=" * 78
    )
    print(
        f"{train_session}->{test_session}"
    )
    print(
        f"Window level: "
        f"accuracy={wm['accuracy']:.3f} "
        f"macroF1={wm['macro_f1']:.3f} "
        f"medianXY={wm['xy_median_m']:.3f} m "
        f"p90XY={wm['xy_p90_m']:.3f} m"
    )
    print(
        f"Trial majority vote: "
        f"accuracy="
        f"{result['trial_majority_metrics']['accuracy']:.3f} "
        f"macroF1="
        f"{result['trial_majority_metrics']['macro_f1']:.3f}"
    )
    print(
        f"Trial mean probability: "
        f"accuracy={tm['accuracy']:.3f} "
        f"macroF1={tm['macro_f1']:.3f}"
    )
    print(
        f"Selected epochs: "
        f"{result['selected_epochs']}"
    )
    print(
        f"Saved model: "
        f"{save_model_path}"
    )
    print(
        f"Saved results: "
        f"{output_dir}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_device(
    requested: str,
) -> str:
    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "--device cuda was requested, "
                "but torch.cuda.is_available() "
                "is False."
            )
        return "cuda"

    if requested == "cpu":
        return "cpu"

    raise SystemExit(
        f"Unsupported --device "
        f"{requested!r}. "
        f"Use auto, cpu, or cuda."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Raw dual-link CSI 2D CNN "
            "for TriCSI localization."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "5fold",
            "cross-session",
        ],
        required=True,
    )

    parser.add_argument(
        "--session",
        default="S1",
        help=(
            "Session used by --mode 5fold."
        ),
    )

    parser.add_argument(
        "--train-session",
        default="S1",
    )

    parser.add_argument(
        "--test-session",
        default="S2",
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
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
        "--target-steps",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--min-packets",
        type=int,
        default=MIN_PACKETS_PER_LINK,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--augmentation-strength",
        type=float,
        default=1.0,
        help=(
            "0 disables augmentation. "
            "1 is the recommended baseline."
        ),
    )

    parser.add_argument(
        "--no-rssi",
        action="store_true",
        help=(
            "Disable the small RSSI "
            "auxiliary branch."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    device = resolve_device(
        args.device
    )

    set_seed(
        args.seed
    )

    cfg = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=(
            args.label_smoothing
        ),
        dropout=args.dropout,
        augmentation_strength=(
            args.augmentation_strength
        ),
        use_rssi=not args.no_rssi,
        device=device,
        seed=args.seed,
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "TriCSI RAW-CSI 2D CNN"
    )
    print(
        "=" * 78
    )
    print(
        f"PyTorch       : "
        f"{torch.__version__}"
    )
    print(
        f"Device        : "
        f"{device}"
    )
    print(
        f"Window / hop  : "
        f"{args.window:.2f} s / "
        f"{args.hop:.2f} s"
    )
    print(
        f"Resample      : "
        f"{args.target_steps} "
        f"time steps x 64 subcarriers"
    )
    print(
        f"Input         : "
        f"4 x {args.target_steps} x 64"
    )
    print(
        f"RSSI aux      : "
        f"{cfg.use_rssi}"
    )
    print(
        f"Augmentation  : "
        f"{cfg.augmentation_strength}"
    )
    print(
        f"Max epochs    : "
        f"{cfg.max_epochs}"
    )
    print(
        f"Patience      : "
        f"{cfg.patience}"
    )
    print(
        "=" * 78
    )

    start_time = time.time()

    if args.mode == "5fold":
        paths = trial_paths_for_session(
            args.raw_dir,
            args.session,
        )

        print(
            f"\nLoading raw CSI for "
            f"{args.session} "
            f"({len(paths)} trials)..."
        )

        data = gather_raw_windows(
            paths,
            window_seconds=args.window,
            hop_seconds=args.hop,
            target_steps=args.target_steps,
            min_packets_per_link=(
                args.min_packets
            ),
        )

        print(
            f"\nCreated "
            f"{len(data.y_text)} "
            f"raw CSI windows."
        )
        print(
            f"Independent trials: "
            f"{len(set(data.group))}"
        )

        output_dir = (
            RESULTS_DIR
            / (
                f"cnn_5fold_"
                f"{args.session.lower()}"
            )
        )

        run_5fold(
            data,
            session=args.session,
            cfg=cfg,
            output_dir=output_dir,
        )

    else:
        if (
            args.train_session
            == args.test_session
        ):
            raise SystemExit(
                "--train-session and "
                "--test-session must differ."
            )

        train_paths = (
            trial_paths_for_session(
                args.raw_dir,
                args.train_session,
            )
        )

        test_paths = (
            trial_paths_for_session(
                args.raw_dir,
                args.test_session,
            )
        )

        print(
            f"\nLoading TRAIN raw CSI "
            f"{args.train_session} "
            f"({len(train_paths)} trials)..."
        )

        train_data = gather_raw_windows(
            train_paths,
            window_seconds=args.window,
            hop_seconds=args.hop,
            target_steps=args.target_steps,
            min_packets_per_link=(
                args.min_packets
            ),
        )

        print(
            f"\nLoading TEST raw CSI "
            f"{args.test_session} "
            f"({len(test_paths)} trials)..."
        )

        test_data = gather_raw_windows(
            test_paths,
            window_seconds=args.window,
            hop_seconds=args.hop,
            target_steps=args.target_steps,
            min_packets_per_link=(
                args.min_packets
            ),
        )

        print(
            f"\nTRAIN windows: "
            f"{len(train_data.y_text)}"
        )
        print(
            f"TEST windows : "
            f"{len(test_data.y_text)}"
        )

        output_dir = (
            RESULTS_DIR
            / (
                f"cnn_cross_session_"
                f"{args.train_session.lower()}_to_"
                f"{args.test_session.lower()}"
            )
        )

        run_cross_session(
            train_data,
            test_data,
            train_session=(
                args.train_session
            ),
            test_session=(
                args.test_session
            ),
            cfg=cfg,
            output_dir=output_dir,
        )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"\nTotal elapsed: "
        f"{elapsed / 60.0:.1f} min"
    )


if __name__ == "__main__":
    main()
