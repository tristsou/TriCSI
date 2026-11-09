from __future__ import annotations

"""
TriCSI-Anchor: background-relative, multi-link, cross-session/cross-orientation metric learning with two-anchor target prototypes.

This is the frozen final research model.

Research design
---------------
The model addresses the dominant practical failure observed in CSI localization:
temporal/session domain shift.

It combines four ideas:

1. BACKGROUND-RELATIVE CSI
   For each session and each link, subtract the per-subcarrier median EMPTY
   log-amplitude. This removes much of the static/session-specific radio
   fingerprint while preserving person-induced perturbations.

2. LINK-SPECIFIC ENCODERS
   LEFT and RIGHT links keep independent small CNN encoders. They are not
   hard-assigned to room columns. A lightweight learned gate fuses their
   embeddings sample-by-sample.

3. SPATIALLY AWARE CROSS-SESSION EMBEDDING
   The embedding is trained with:
       zone classification
       occupancy classification
       coordinate regression
       pairwise spatial similarity loss
   The spatial loss depends only on physical zone distance, so the same location
   across different sessions is explicitly encouraged to map to the same place
   in embedding space.

4. TWO-ANCHOR TARGET-SESSION PROTOTYPES
   At a new session, collect:
       one EMPTY baseline
       one labeled trial per occupied zone
   Freeze the neural network. No gradient update is performed.
   Each target zone is represented by the mean embedding of that one calibration
   trial. Test windows are classified by cosine similarity to these prototypes.

This makes calibration cheap and avoids retraining the network in every session.

Modes
-----
Development sanity check using EXISTING S4-S6 only:
    python train_tricsi_anchor_final.py --mode loso

Final frozen evaluation after collecting S7 with collect_s7_tricsi_adapt.py:
    python train_tricsi_anchor_final.py --mode final --test-session S7

The S7 collector reserves trial01 for calibration and trials02-05 for evaluation.
"""

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required.\n"
        "Install the CPU build in the active .venv, then rerun."
    ) from exc

from sklearn.metrics import accuracy_score, f1_score

from train_csi_cnn import (
    RawWindowSet,
    gather_raw_windows,
    trial_paths_for_session,
)

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


SOURCE_SESSIONS = ("S4", "S5", "S6", "S7")
TARGET_STEPS = 32
OCCUPIED = [z for z in CLASS_ORDER if z != "EMPTY"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Raw session utilities
# ---------------------------------------------------------------------------

def load_session(
    raw_dir: Path,
    session: str,
) -> RawWindowSet:
    paths = trial_paths_for_session(raw_dir, session)

    print(
        f"\nLoading {session}: {len(paths)} trial files"
    )

    raw = gather_raw_windows(
        paths,
        window_seconds=WINDOW_SECONDS,
        hop_seconds=HOP_SECONDS,
        target_steps=TARGET_STEPS,
        min_packets_per_link=MIN_PACKETS_PER_LINK,
    )

    return raw


def subset(
    raw: RawWindowSet,
    mask: np.ndarray,
) -> RawWindowSet:
    names = raw.__dataclass_fields__.keys()
    kwargs = {
        name: getattr(raw, name)[mask]
        for name in names
    }
    return RawWindowSet(**kwargs)


def combine(
    sets: list[RawWindowSet],
) -> RawWindowSet:
    if not sets:
        raise ValueError("No RawWindowSet objects supplied.")

    names = sets[0].__dataclass_fields__.keys()
    kwargs = {
        name: np.concatenate(
            [getattr(s, name) for s in sets],
            axis=0,
        )
        for name in names
    }
    return RawWindowSet(**kwargs)


@dataclass
class Background:
    left_logamp: np.ndarray   # [64]
    right_logamp: np.ndarray  # [64]
    left_rssi: float
    right_rssi: float


def estimate_background(
    raw: RawWindowSet,
    *,
    calibration_only: bool,
) -> Background:
    mask = raw.y_text == "EMPTY"

    if calibration_only:
        first = mask & (raw.trial_number == 1)
        if np.any(first):
            mask = first

    if not np.any(mask):
        raise RuntimeError(
            "No EMPTY windows available for session background calibration."
        )

    left = np.median(
        raw.X[mask, 0, :, :],
        axis=(0, 1),
    ).astype(np.float32)

    right = np.median(
        raw.X[mask, 2, :, :],
        axis=(0, 1),
    ).astype(np.float32)

    lr = float(
        np.median(
            raw.rssi[mask, 0]
        )
    )
    rr = float(
        np.median(
            raw.rssi[mask, 2]
        )
    )

    return Background(
        left_logamp=left,
        right_logamp=right,
        left_rssi=lr,
        right_rssi=rr,
    )


def apply_background(
    raw: RawWindowSet,
    bg: Background,
) -> RawWindowSet:
    X = raw.X.copy().astype(np.float32)
    rssi = raw.rssi.copy().astype(np.float32)

    # Absolute channels become session-relative person/background residuals.
    X[:, 0, :, :] -= bg.left_logamp[None, None, :]
    X[:, 2, :, :] -= bg.right_logamp[None, None, :]

    # Packet-centered channels already remove common mode and are retained.
    # RSSI means become relative to session EMPTY baseline; std terms unchanged.
    rssi[:, 0] -= bg.left_rssi
    rssi[:, 2] -= bg.right_rssi

    return RawWindowSet(
        X=X,
        rssi=rssi,
        y_text=raw.y_text.copy(),
        session=raw.session.copy(),
        trial_number=raw.trial_number.copy(),
        orientation=raw.orientation.copy(),
        group=raw.group.copy(),
        source_file=raw.source_file.copy(),
        window_index=raw.window_index.copy(),
        window_start_s=raw.window_start_s.copy(),
        left_packets=raw.left_packets.copy(),
        right_packets=raw.right_packets.copy(),
    )


@dataclass
class Norm:
    csi_mean: np.ndarray
    csi_std: np.ndarray
    rssi_mean: np.ndarray
    rssi_std: np.ndarray


def compute_norm(raw: RawWindowSet) -> Norm:
    mean = np.mean(
        raw.X,
        axis=(0, 2, 3),
        dtype=np.float64,
    ).astype(np.float32)

    std = np.std(
        raw.X,
        axis=(0, 2, 3),
        dtype=np.float64,
    ).astype(np.float32)

    std = np.maximum(std, 1e-4)

    rmean = np.mean(
        raw.rssi,
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    rstd = np.std(
        raw.rssi,
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    rstd = np.maximum(rstd, 1e-4)

    return Norm(
        csi_mean=mean,
        csi_std=std,
        rssi_mean=rmean,
        rssi_std=rstd,
    )


# ---------------------------------------------------------------------------
# Labels / geometry
# ---------------------------------------------------------------------------

LABELS = list(CLASS_ORDER)
LABEL_TO_INDEX = {
    z: i for i, z in enumerate(LABELS)
}

ORIENTATION_ORDER = ["N", "E", "S", "W", "NA"]
ORI_TO_INDEX = {o: i for i, o in enumerate(ORIENTATION_ORDER)}

xy_array = np.asarray(
    [ZONE_CENTERS[z] for z in OCCUPIED],
    dtype=np.float32,
)

XY_MIN = xy_array.min(axis=0)
XY_MAX = xy_array.max(axis=0)
XY_RANGE = np.maximum(XY_MAX - XY_MIN, 1e-6)


def zone_xy_norm(zone: str) -> np.ndarray:
    if zone == "EMPTY":
        return np.asarray([0.5, 0.5], dtype=np.float32)

    xy = np.asarray(ZONE_CENTERS[zone], dtype=np.float32)
    return ((xy - XY_MIN) / XY_RANGE).astype(np.float32)


def zone_distance_matrix() -> np.ndarray:
    n = len(LABELS)
    out = np.zeros((n, n), dtype=np.float32)

    occ_xy = {
        z: np.asarray(ZONE_CENTERS[z], dtype=np.float32)
        for z in OCCUPIED
    }

    max_d = 0.0
    for a in OCCUPIED:
        for b in OCCUPIED:
            max_d = max(
                max_d,
                float(np.linalg.norm(occ_xy[a] - occ_xy[b])),
            )

    for i, a in enumerate(LABELS):
        for j, b in enumerate(LABELS):
            if a == "EMPTY" or b == "EMPTY":
                out[i, j] = 0.0 if a == b else 1.0
            else:
                out[i, j] = (
                    float(np.linalg.norm(occ_xy[a] - occ_xy[b]))
                    / max(max_d, 1e-6)
                )

    return out


ZONE_DIST = zone_distance_matrix()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AdaptDataset(Dataset):
    def __init__(
        self,
        raw: RawWindowSet,
        norm: Norm,
        *,
        augment: bool,
    ):
        self.raw = raw
        self.norm = norm
        self.augment = augment

        self.session_names = sorted(
            set(str(x) for x in raw.session)
        )
        self.session_map = {
            s: i for i, s in enumerate(self.session_names)
        }

    def __len__(self):
        return len(self.raw.y_text)

    def __getitem__(self, i):
        x = torch.from_numpy(
            self.raw.X[i].copy()
        ).float()

        rssi = torch.from_numpy(
            self.raw.rssi[i].copy()
        ).float()

        cmean = torch.from_numpy(
            self.norm.csi_mean
        ).float()[:, None, None]

        cstd = torch.from_numpy(
            self.norm.csi_std
        ).float()[:, None, None]

        rmean = torch.from_numpy(
            self.norm.rssi_mean
        ).float()

        rstd = torch.from_numpy(
            self.norm.rssi_std
        ).float()

        x = (x - cmean) / cstd
        rssi = (rssi - rmean) / rstd

        if self.augment:
            # Small time jitter.
            if torch.rand(()) < 0.35:
                shift = int(torch.randint(-1, 2, (1,)).item())
                x = torch.roll(x, shifts=shift, dims=1)

            # Mild subcarrier masking.
            if torch.rand(()) < 0.25:
                width = int(torch.randint(2, 6, (1,)).item())
                start = int(
                    torch.randint(
                        0,
                        max(1, x.shape[2] - width + 1),
                        (1,),
                    ).item()
                )
                x[:, :, start:start + width] = 0.0

            # Small residual noise only: background calibration already handles
            # the main session offset, so augmentation stays conservative.
            if torch.rand(()) < 0.60:
                x += 0.02 * torch.randn_like(x)

        zone = str(self.raw.y_text[i])
        y = LABEL_TO_INDEX[zone]
        occ = 0.0 if zone == "EMPTY" else 1.0
        xy = zone_xy_norm(zone)

        orientation = str(self.raw.orientation[i])
        if orientation not in ORI_TO_INDEX:
            orientation = "NA"

        return (
            x,
            rssi,
            torch.tensor(y, dtype=torch.long),
            torch.tensor(occ, dtype=torch.float32),
            torch.from_numpy(xy).float(),
            torch.tensor(
                self.session_map[str(self.raw.session[i])],
                dtype=torch.long,
            ),
            torch.tensor(
                ORI_TO_INDEX[orientation],
                dtype=torch.long,
            ),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LinkEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                16,
                kernel_size=(3, 5),
                padding=(1, 2),
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(
                16,
                32,
                kernel_size=(3, 5),
                padding=(1, 2),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(
                32,
                48,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((2, 4)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 2 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, dim),
        )

    def forward(self, x):
        return self.fc(self.net(x))


class TriCSIAdaptNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embed_dim: int = 64,
    ):
        super().__init__()

        self.left = LinkEncoder(2, embed_dim)
        self.right = LinkEncoder(2, embed_dim)

        self.rssi = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, embed_dim),
        )

        self.zone_head = nn.Linear(embed_dim, n_classes)
        self.occ_head = nn.Linear(embed_dim, 1)
        self.xy_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
        rssi,
        *,
        link_dropout: bool = False,
    ):
        e_l = self.left(x[:, 0:2])
        e_r = self.right(x[:, 2:4])
        e_rssi = self.rssi(rssi)

        if self.training and link_dropout and torch.rand(()) < 0.15:
            # Drop exactly one link embedding, never both.
            if torch.rand(()) < 0.5:
                e_l = torch.zeros_like(e_l)
            else:
                e_r = torch.zeros_like(e_r)

        gate_in = torch.cat([e_l, e_r, e_rssi], dim=1)
        w = torch.softmax(self.gate(gate_in), dim=1)

        fused_in = torch.cat(
            [
                e_l * w[:, 0:1],
                e_r * w[:, 1:2],
                e_rssi,
            ],
            dim=1,
        )

        emb = self.fuse(fused_in)
        emb = F.normalize(emb, p=2, dim=1)

        return {
            "embedding": emb,
            "zone_logits": self.zone_head(emb),
            "occ_logit": self.occ_head(emb).squeeze(1),
            "xy": self.xy_head(emb),
            "link_weights": w,
        }


# ---------------------------------------------------------------------------
# Spatial loss
# ---------------------------------------------------------------------------

def spatial_similarity_loss(
    emb,
    labels,
    sessions,
    orientations,
):
    """
    Geometry-aware metric loss with explicit nuisance invariance.

    Same-zone pairs are pulled together. Pairs from different sessions and
    different orientations receive extra weight. A same-zone pair that differs
    in BOTH session and orientation gets the largest weight. Nearby physical
    zones remain more similar than far zones.
    """
    n = emb.shape[0]
    if n < 2:
        return emb.new_tensor(0.0)

    sim = emb @ emb.T
    sim01 = (sim + 1.0) * 0.5

    d = torch.from_numpy(ZONE_DIST).to(emb.device)
    pair_d = d[labels[:, None], labels[None, :]]
    target = torch.exp(-0.5 * (pair_d / 0.35) ** 2)

    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    valid = ~eye

    same_zone = labels[:, None] == labels[None, :]
    cross_session = sessions[:, None] != sessions[None, :]
    cross_orientation = orientations[:, None] != orientations[None, :]

    weights = torch.ones_like(target)
    weights = torch.where(
        same_zone & cross_session,
        torch.full_like(weights, 3.0),
        weights,
    )
    weights = torch.where(
        same_zone & cross_orientation,
        torch.maximum(weights, torch.full_like(weights, 3.0)),
        weights,
    )
    weights = torch.where(
        same_zone & cross_session & cross_orientation,
        torch.full_like(weights, 6.0),
        weights,
    )

    err = (sim01 - target) ** 2
    return torch.sum(err[valid] * weights[valid]) / torch.sum(weights[valid])


# ---------------------------------------------------------------------------
# Training / embedding
# ---------------------------------------------------------------------------

def make_loader(raw, norm, batch_size, augment, shuffle):
    ds = AdaptDataset(
        raw,
        norm,
        augment=augment,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=augment,
    )


def train_model(
    source_raw: RawWindowSet,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    set_seed(seed)

    norm = compute_norm(source_raw)

    model = TriCSIAdaptNet(
        n_classes=len(LABELS),
        embed_dim=64,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    loader = make_loader(
        source_raw,
        norm,
        batch_size,
        augment=True,
        shuffle=True,
    )

    model.train()

    for epoch in range(1, epochs + 1):
        sums = {
            "total": 0.0,
            "ce": 0.0,
            "occ": 0.0,
            "xy": 0.0,
            "spatial": 0.0,
            "n": 0,
        }

        for x, rssi, y, occ, xy, session, orientation in loader:
            optimizer.zero_grad(set_to_none=True)

            out = model(
                x,
                rssi,
                link_dropout=True,
            )

            ce = F.cross_entropy(
                out["zone_logits"],
                y,
            )

            occ_loss = F.binary_cross_entropy_with_logits(
                out["occ_logit"],
                occ,
            )

            occupied = occ > 0.5
            if torch.any(occupied):
                xy_loss = F.smooth_l1_loss(
                    out["xy"][occupied],
                    xy[occupied],
                )
            else:
                xy_loss = ce.new_tensor(0.0)

            spatial = spatial_similarity_loss(
                out["embedding"],
                y,
                session,
                orientation,
            )

            loss = (
                ce
                + 0.25 * occ_loss
                + 0.40 * xy_loss
                + 0.55 * spatial
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

            b = len(y)
            sums["total"] += float(loss.item()) * b
            sums["ce"] += float(ce.item()) * b
            sums["occ"] += float(occ_loss.item()) * b
            sums["xy"] += float(xy_loss.item()) * b
            sums["spatial"] += float(spatial.item()) * b
            sums["n"] += b

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            n = max(1, sums["n"])
            print(
                f"epoch {epoch:03d}/{epochs} | "
                f"loss={sums['total']/n:.4f} "
                f"ce={sums['ce']/n:.4f} "
                f"occ={sums['occ']/n:.4f} "
                f"xy={sums['xy']/n:.4f} "
                f"spatial={sums['spatial']/n:.4f}"
            )

    return model.eval(), norm


@torch.no_grad()
def embed_raw(
    model,
    raw: RawWindowSet,
    norm: Norm,
    batch_size: int = 256,
):
    loader = make_loader(
        raw,
        norm,
        batch_size,
        augment=False,
        shuffle=False,
    )

    embeddings = []
    occ_probs = []
    xy_preds = []
    link_weights = []

    for x, rssi, *_ in loader:
        out = model(x, rssi)
        embeddings.append(
            out["embedding"].cpu().numpy()
        )
        occ_probs.append(
            torch.sigmoid(
                out["occ_logit"]
            ).cpu().numpy()
        )
        xy_preds.append(
            out["xy"].cpu().numpy()
        )
        link_weights.append(
            out["link_weights"].cpu().numpy()
        )

    return {
        "embedding": np.concatenate(embeddings, axis=0),
        "occ_prob": np.concatenate(occ_probs, axis=0),
        "xy_pred": np.concatenate(xy_preds, axis=0),
        "link_weights": np.concatenate(link_weights, axis=0),
    }


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def make_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
):
    protos = []

    for zone in OCCUPIED:
        mask = labels == zone
        if not np.any(mask):
            raise RuntimeError(
                f"No calibration embeddings for {zone}."
            )

        p = np.mean(
            embeddings[mask],
            axis=0,
        )
        p = p / max(
            np.linalg.norm(p),
            1e-12,
        )
        protos.append(p)

    return np.stack(protos, axis=0)


def prototype_probabilities(
    embeddings,
    prototypes,
    temperature: float = 0.12,
):
    e = normalize_rows(embeddings)
    p = normalize_rows(prototypes)

    logits = (e @ p.T) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)

    q = np.exp(logits)
    q /= np.maximum(
        q.sum(axis=1, keepdims=True),
        1e-12,
    )
    return q


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def trial_key(raw: RawWindowSet, i: int) -> str:
    return (
        f"{raw.session[i]}|"
        f"{raw.y_text[i]}|"
        f"{int(raw.trial_number[i])}|"
        f"{raw.orientation[i]}"
    )


def zone_xy(zone: str):
    return np.asarray(
        ZONE_CENTERS[zone],
        dtype=np.float64,
    )


def adjacent(a: str, b: str) -> bool:
    if a == "EMPTY" or b == "EMPTY":
        return a == b

    ia = int(a[1:]) - 1
    ib = int(b[1:]) - 1

    ra, ca = divmod(ia, 3)
    rb, cb = divmod(ib, 3)

    return max(
        abs(ra - rb),
        abs(ca - cb),
    ) <= 1


def evaluate_target(
    model,
    norm,
    source_raw,
    target_raw,
):
    src_emb = embed_raw(
        model,
        source_raw,
        norm,
    )

    tgt_emb = embed_raw(
        model,
        target_raw,
        norm,
    )

    # Source prototypes = zero-shot baseline.
    src_occ = source_raw.y_text != "EMPTY"
    source_protos = make_prototypes(
        src_emb["embedding"][src_occ],
        source_raw.y_text[src_occ],
    )

    # S8 target calibration: two short occupied anchors per zone,
    # trial01=N and trial02=E. No target gradient retraining.
    cal_mask = (
        (target_raw.y_text != "EMPTY")
        & (target_raw.trial_number <= 2)
    )

    target_protos = make_prototypes(
        tgt_emb["embedding"][cal_mask],
        target_raw.y_text[cal_mask],
    )

    # S8 occupied final test = trials03-06.
    # S8 EMPTY final test = trials02-05.
    test_mask = (
        (
            (target_raw.y_text != "EMPTY")
            & (target_raw.trial_number >= 3)
        )
        |
        (
            (target_raw.y_text == "EMPTY")
            & (target_raw.trial_number >= 2)
        )
    )

    test_idx = np.where(test_mask)[0]

    zero_probs = prototype_probabilities(
        tgt_emb["embedding"][test_idx],
        source_protos,
    )

    one_probs = prototype_probabilities(
        tgt_emb["embedding"][test_idx],
        target_protos,
    )

    occ_prob = tgt_emb["occ_prob"][test_idx]

    groups = np.asarray(
        [
            trial_key(
                target_raw,
                int(i),
            )
            for i in test_idx
        ],
        dtype=object,
    )

    y_true = []
    y_zero = []
    y_one = []
    xy_errors = []
    adjacent_hits = []
    link_weight_rows = []

    for g in sorted(set(groups)):
        local = np.where(groups == g)[0]
        global_i = test_idx[local[0]]

        true_zone = str(
            target_raw.y_text[
                global_i
            ]
        )

        mean_occ = float(
            np.mean(
                occ_prob[local]
            )
        )

        mean_zero = np.mean(
            zero_probs[local],
            axis=0,
        )
        mean_one = np.mean(
            one_probs[local],
            axis=0,
        )

        if mean_occ < 0.5:
            pred_zero = "EMPTY"
            pred_one = "EMPTY"
        else:
            pred_zero = OCCUPIED[
                int(np.argmax(mean_zero))
            ]
            pred_one = OCCUPIED[
                int(np.argmax(mean_one))
            ]

        y_true.append(true_zone)
        y_zero.append(pred_zero)
        y_one.append(pred_one)

        if true_zone != "EMPTY":
            centers = np.asarray(
                [ZONE_CENTERS[z] for z in OCCUPIED],
                dtype=np.float64,
            )

            xy = mean_one @ centers
            err = float(
                np.linalg.norm(
                    xy - zone_xy(true_zone)
                )
            )
            xy_errors.append(err)
            adjacent_hits.append(
                int(
                    pred_one != "EMPTY"
                    and adjacent(
                        true_zone,
                        pred_one,
                    )
                )
            )

        lw = np.mean(
            tgt_emb["link_weights"][
                test_idx[local]
            ],
            axis=0,
        )

        link_weight_rows.append(
            {
                "group": g,
                "true_zone": true_zone,
                "pred_zero": pred_zero,
                "pred_one": pred_one,
                "mean_occ_prob": mean_occ,
                "left_weight": float(lw[0]),
                "right_weight": float(lw[1]),
            }
        )

    y_true = np.asarray(y_true, dtype=object)
    y_zero = np.asarray(y_zero, dtype=object)
    y_one = np.asarray(y_one, dtype=object)

    errors = np.asarray(
        xy_errors,
        dtype=np.float64,
    )

    per_zone = {}

    for zone in LABELS:
        mask = y_true == zone
        if np.any(mask):
            per_zone[zone] = {
                "n": int(np.sum(mask)),
                "zero_shot": float(
                    np.mean(
                        y_zero[mask] == zone
                    )
                ),
                "two_anchor": float(
                    np.mean(
                        y_one[mask] == zone
                    )
                ),
            }

    metrics = {
        "zero_shot_trial_accuracy": float(
            accuracy_score(
                y_true,
                y_zero,
            )
        ),
        "two_anchor_trial_accuracy": float(
            accuracy_score(
                y_true,
                y_one,
            )
        ),
        "zero_shot_macro_f1": float(
            f1_score(
                y_true,
                y_zero,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "two_anchor_macro_f1": float(
            f1_score(
                y_true,
                y_one,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "occupied_exact_or_adjacent": (
            float(np.mean(adjacent_hits))
            if adjacent_hits
            else float("nan")
        ),
        "median_xy_error_m": (
            float(np.median(errors))
            if len(errors)
            else float("nan")
        ),
        "p90_xy_error_m": (
            float(np.percentile(errors, 90))
            if len(errors)
            else float("nan")
        ),
        "per_zone": per_zone,
    }

    return metrics, link_weight_rows


# ---------------------------------------------------------------------------
# Session preparation
# ---------------------------------------------------------------------------

def prepare_source_session(raw):
    bg = estimate_background(
        raw,
        calibration_only=False,
    )
    return apply_background(
        raw,
        bg,
    )


def prepare_target_session(raw):
    bg = estimate_background(
        raw,
        calibration_only=True,
    )
    return apply_background(
        raw,
        bg,
    )


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def print_metrics(title, m):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(
        f"zero-shot exact : "
        f"{m['zero_shot_trial_accuracy']:.3f}"
    )
    print(
        f"TWO-ANCHOR exact  : "
        f"{m['two_anchor_trial_accuracy']:.3f}"
    )
    print(
        f"two-anchor macroF1: "
        f"{m['two_anchor_macro_f1']:.3f}"
    )
    print(
        f"adjacent occ    : "
        f"{m['occupied_exact_or_adjacent']:.3f}"
    )
    print(
        f"median XY       : "
        f"{m['median_xy_error_m']:.3f} m"
    )
    print(
        f"P90 XY          : "
        f"{m['p90_xy_error_m']:.3f} m"
    )

    print("\nper-zone trial recall")
    for zone in LABELS:
        if zone in m["per_zone"]:
            z = m["per_zone"][zone]
            print(
                f"  {zone:5s} "
                f"zero={z['zero_shot']:.3f} "
                f"one={z['two_anchor']:.3f} "
                f"(n={z['n']})"
            )


def run_loso(args):
    raw_map = {
        s: load_session(args.raw_dir, s)
        for s in SOURCE_SESSIONS
    }

    results = []

    for fold, test_session in enumerate(
        SOURCE_SESSIONS
    ):
        train_sessions = [
            s
            for s in SOURCE_SESSIONS
            if s != test_session
        ]

        print(
            "\n\n"
            + "#" * 78
        )
        print(
            f"LOSO: {'+'.join(train_sessions)} "
            f"-> {test_session}"
        )
        print(
            "#" * 78
        )

        source_sets = [
            prepare_source_session(
                raw_map[s]
            )
            for s in train_sessions
        ]

        source = combine(
            source_sets
        )

        target = prepare_target_session(
            raw_map[test_session]
        )

        model, norm = train_model(
            source,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + fold * 1000,
        )

        metrics, _ = evaluate_target(
            model,
            norm,
            source,
            target,
        )

        print_metrics(
            f"{'+'.join(train_sessions)} -> "
            f"{test_session}",
            metrics,
        )

        results.append(
            {
                "train": "+".join(
                    train_sessions
                ),
                "test": test_session,
                **{
                    k: v
                    for k, v in metrics.items()
                    if k != "per_zone"
                },
            }
        )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "TriCSI-Adapt LOSO SUMMARY"
    )
    print(
        "=" * 78
    )

    for r in results:
        print(
            f"{r['train']:5s}->{r['test']} | "
            f"zero={r['zero_shot_trial_accuracy']:.3f} | "
            f"two-anchor={r['two_anchor_trial_accuracy']:.3f} | "
            f"adj={r['occupied_exact_or_adjacent']:.3f} | "
            f"med={r['median_xy_error_m']:.3f}m | "
            f"p90={r['p90_xy_error_m']:.3f}m"
        )

    mean_zero = float(
        np.mean(
            [
                r[
                    "zero_shot_trial_accuracy"
                ]
                for r in results
            ]
        )
    )

    mean_one = float(
        np.mean(
            [
                r[
                    "two_anchor_trial_accuracy"
                ]
                for r in results
            ]
        )
    )

    print(
        f"\nMean zero-shot exact : "
        f"{mean_zero:.3f}"
    )
    print(
        f"Mean TWO-ANCHOR exact  : "
        f"{mean_one:.3f}"
    )
    print(
        f"Calibration gain      : "
        f"{mean_one - mean_zero:+.3f}"
    )

    out_dir = (
        RESULTS_DIR
        / "tricsi_anchor_final"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        out_dir
        / "loso_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "results": results,
                "mean_zero_shot": mean_zero,
                "mean_two_anchor": mean_one,
                "architecture_frozen": True,
            },
            f,
            indent=2,
        )

    print(
        "\nThis is a sanity check, not a tuning loop."
    )
    print(
        "Do not sweep hyperparameters from these folds. "
        "The architecture is frozen before S7."
    )


def run_final(args):
    source_raw_map = {
        s: load_session(args.raw_dir, s)
        for s in SOURCE_SESSIONS
    }

    target_raw = load_session(
        args.raw_dir,
        args.test_session,
    )

    source = combine(
        [
            prepare_source_session(
                source_raw_map[s]
            )
            for s in SOURCE_SESSIONS
        ]
    )

    target = prepare_target_session(
        target_raw
    )

    print(
        "\nTraining frozen final encoder "
        "on S4+S5+S6+S7..."
    )

    model, norm = train_model(
        source,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    metrics, rows = evaluate_target(
        model,
        norm,
        source,
        target,
    )

    print_metrics(
        f"FINAL S4+S5+S6+S7 -> "
        f"{args.test_session}",
        metrics,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "tricsi_anchor_s4_s5_s6_s7.pt"
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": LABELS,
            "occupied": OCCUPIED,
            "norm": {
                "csi_mean": norm.csi_mean,
                "csi_std": norm.csi_std,
                "rssi_mean": norm.rssi_mean,
                "rssi_std": norm.rssi_std,
            },
            "zone_centers": ZONE_CENTERS,
            "window_seconds": WINDOW_SECONDS,
            "target_steps": TARGET_STEPS,
            "method": (
                "background_relative_dual_link_"
                "spatial_metric_two_anchor_prototypes"
            ),
        },
        model_path,
    )

    out_dir = (
        RESULTS_DIR
        / "tricsi_anchor_final"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        out_dir
        / f"final_{args.test_session}_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    csv_path = (
        out_dir
        / f"final_{args.test_session}_trial_predictions.csv"
    )

    if rows:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(rows)

    print(
        f"\nSaved model: {model_path}"
    )
    print(
        f"Saved final metrics: {out_dir}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["final"],
        default="final",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
    )
    parser.add_argument(
        "--test-session",
        default="S8",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=70,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=26082026,
    )

    args = parser.parse_args()

    print(
        "\n"
        + "=" * 78
    )
    print(
        "TriCSI-Anchor FINAL MODEL"
    )
    print(
        "=" * 78
    )
    print(
        "background-relative amplitude"
    )
    print(
        "dual link-specific CNN encoders"
    )
    print(
        "learned sample-wise link fusion"
    )
    print(
        "spatial cross-session metric loss"
    )
    print(
        "two-anchor target-session prototypes"
    )
    print(
        "=" * 78
    )

    run_final(args)


if __name__ == "__main__":
    main()
