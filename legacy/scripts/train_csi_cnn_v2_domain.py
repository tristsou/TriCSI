from __future__ import annotations

"""
TriCSI Multi-Session Domain-Invariant CNN V2

Why this model exists
---------------------
CNN V1:
- S1 5-fold window accuracy: ~62.5%
- S1 -> S2 window accuracy:  ~9.4% (chance for 10 classes)

Zone-conditioned diagnostics showed that the global CSI distribution moved only
modestly, but the class/zone fingerprints were not stable:
- absolute same-zone nearest match: 3/10
- centered same-zone nearest match: 2/10

So V2 does NOT try to solve the problem with stronger global normalization.
Instead it trains on multiple DEVELOPMENT sessions simultaneously and explicitly
encourages the learned embedding to keep location information while discarding
session identity.

Loss:
    total =
        zone_cross_entropy
        + lambda_domain * session_adversarial_loss
        + lambda_align  * same-zone_cross-session_centroid_loss

The session adversary uses a gradient reversal layer:
- session head tries to predict S1 vs S2
- encoder receives the opposite gradient, so it learns to make session identity
  harder to recover

The centroid loss:
- for each zone in a batch, compute S1 embedding centroid and S2 centroid
- pull same-zone S1/S2 centroids together

Input:
    4 x 32 x 64
    [L absolute logamp, L packet-centered, R absolute logamp, R packet-centered]

RSSI is intentionally excluded from V2:
- RF NO_RSSI ~= ALL cross-session
- RIGHT RSSI shifted ~ -7 dB between S1/S2
- so RSSI is an unnecessary session cue

Modes
-----
A) Development CV on S1+S2:
   Hold out trial number k from BOTH sessions.
   This is NOT an unseen-session test. It checks whether domain-invariant
   training preserves localization on unseen trials from known development
   sessions.

    python train_csi_cnn_v2_domain.py --mode cv --sessions S1 S2

B) Final untouched-session test:
   After architecture/hyperparameters are frozen, train on S1+S2 and test S3.

    python train_csi_cnn_v2_domain.py --mode final --sessions S1 S2 --test-session S3

Scientific rule:
- Do not tune based on S3 if S3 is intended to be the final untouched test.
- If you inspect S3 and then modify the model, S3 becomes development data and
  a future S4 should be reserved as the final test.
"""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, Sampler

# Reuse the already-tested raw CSI extraction functions from CNN V1.
from train_csi_cnn import (
    RawWindowSet,
    compute_normalizer,
    gather_raw_windows,
    index_to_text,
    label_to_index_map,
    probability_weighted_xy,
    summarize_xy,
    trial_paths_for_session,
    xy_errors,
)

from project_config import (
    CLASS_ORDER,
    HOP_SECONDS,
    MIN_PACKETS_PER_LINK,
    MODEL_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    WINDOW_SECONDS,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Combine multiple RawWindowSet objects
# ---------------------------------------------------------------------------

def combine_sets(sets: list[RawWindowSet]) -> RawWindowSet:
    if not sets:
        raise ValueError("No datasets supplied.")

    def cat(name: str):
        return np.concatenate(
            [getattr(x, name) for x in sets],
            axis=0,
        )

    return RawWindowSet(
        X=cat("X"),
        rssi=cat("rssi"),
        y_text=cat("y_text"),
        session=cat("session"),
        trial_number=cat("trial_number"),
        orientation=cat("orientation"),
        group=cat("group"),
        source_file=cat("source_file"),
        window_index=cat("window_index"),
        window_start_s=cat("window_start_s"),
        left_packets=cat("left_packets"),
        right_packets=cat("right_packets"),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DomainDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        domain: np.ndarray,
        normalizer,
        *,
        augment: bool,
        augmentation_strength: float,
    ) -> None:
        self.X = X
        self.y = y
        self.domain = domain
        self.normalizer = normalizer
        self.augment = augment
        self.s = augmentation_strength

    def __len__(self) -> int:
        return len(self.y)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        s = self.s
        if not self.augment or s <= 0:
            return x

        # Same basic physical perturbations as V1, but modest; the main V2
        # change is multi-session alignment, not simply stronger augmentation.
        if torch.rand(()) < 0.75:
            x[0] += torch.randn(()) * (0.15 * s)
            x[2] += torch.randn(()) * (0.15 * s)

        if torch.rand(()) < 0.60:
            n_carriers = x.shape[-1]
            knots = 8
            base = torch.randn(1, 1, knots) * (0.07 * s)
            smooth = F.interpolate(
                base,
                size=n_carriers,
                mode="linear",
                align_corners=True,
            )[0, 0]
            smooth -= smooth.mean()

            left = smooth
            right = torch.roll(
                smooth,
                shifts=int(torch.randint(-3, 4, (1,)).item()),
            ) * (0.8 + 0.4 * torch.rand(()))

            x[0] += left[None, :]
            x[1] += left[None, :]
            x[2] += right[None, :]
            x[3] += right[None, :]

        if torch.rand(()) < 0.40:
            shift = int(torch.randint(-2, 3, (1,)).item())
            if shift:
                x = torch.roll(x, shifts=shift, dims=1)

        if torch.rand(()) < 0.30:
            width = int(torch.randint(2, 7, (1,)).item())
            start = int(
                torch.randint(
                    0,
                    max(1, x.shape[2] - width + 1),
                    (1,),
                ).item()
            )
            x[:, :, start:start + width] = 0.0

        if torch.rand(()) < 0.70:
            x += torch.randn_like(x) * (0.02 * s)

        return x

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.X[i].copy()).float()

        mean = torch.from_numpy(
            self.normalizer.csi_mean
        ).float()[:, None, None]

        std = torch.from_numpy(
            self.normalizer.csi_std
        ).float()[:, None, None]

        x = (x - mean) / std
        x = self._augment(x)

        return (
            x,
            torch.tensor(int(self.y[i]), dtype=torch.long),
            torch.tensor(int(self.domain[i]), dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Balanced batch sampler
# ---------------------------------------------------------------------------

class BalancedClassDomainBatchSampler(Sampler[list[int]]):
    """
    Every batch contains equal examples from every class x domain cell.

    With 10 zones x 2 sessions and batch_size=80:
        4 windows from each of 20 cells
    """

    def __init__(
        self,
        y: np.ndarray,
        domain: np.ndarray,
        *,
        n_classes: int,
        n_domains: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.y = y
        self.domain = domain
        self.n_classes = n_classes
        self.n_domains = n_domains
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        self.cells: dict[tuple[int, int], np.ndarray] = {}
        for c in range(n_classes):
            for d in range(n_domains):
                idx = np.where((y == c) & (domain == d))[0]
                if len(idx) == 0:
                    raise ValueError(
                        f"No samples for class={c}, domain={d}"
                    )
                self.cells[(c, d)] = idx

        n_cells = n_classes * n_domains
        self.per_cell = max(1, batch_size // n_cells)
        self.actual_batch = self.per_cell * n_cells
        self.n_batches = max(
            1,
            math.ceil(len(y) / self.actual_batch),
        )

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        for _ in range(self.n_batches):
            batch = []
            for c in range(self.n_classes):
                for d in range(self.n_domains):
                    pool = self.cells[(c, d)]
                    chosen = rng.choice(
                        pool,
                        size=self.per_cell,
                        replace=(len(pool) < self.per_cell),
                    )
                    batch.extend(chosen.tolist())

            rng.shuffle(batch)
            yield batch


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


def grad_reverse(x: torch.Tensor, strength: float) -> torch.Tensor:
    return GradientReverseFunction.apply(x, strength)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DomainInvariantCNN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_domains: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.embedding = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.zone_head = nn.Linear(64, n_classes)

        self.domain_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, n_domains),
        )

    def forward(self, x: torch.Tensor, grl_strength: float = 1.0):
        z = self.backbone(x).flatten(1)
        emb = self.embedding(z)

        zone_logits = self.zone_head(emb)
        domain_logits = self.domain_head(
            grad_reverse(emb, grl_strength)
        )

        return zone_logits, domain_logits, emb


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def cross_session_centroid_loss(
    emb: torch.Tensor,
    y: torch.Tensor,
    domain: torch.Tensor,
    *,
    n_classes: int,
    n_domains: int,
) -> torch.Tensor:
    """
    Pull same-zone centroids across sessions together.

    For two sessions:
        || mean(S1, zone c) - mean(S2, zone c) ||^2
    """
    losses = []

    for c in range(n_classes):
        centroids = []

        for d in range(n_domains):
            mask = (y == c) & (domain == d)
            if torch.any(mask):
                centroids.append(emb[mask].mean(dim=0))

        if len(centroids) >= 2:
            # Generalized pairwise centroid agreement.
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    losses.append(
                        F.mse_loss(centroids[i], centroids[j])
                    )

    if not losses:
        return emb.sum() * 0.0

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
    }


def save_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    *,
    path: Path,
    title: str,
    normalize: bool,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        shown = np.divide(
            cm.astype(float),
            denom,
            out=np.zeros_like(cm, dtype=float),
            where=denom != 0,
        )
    else:
        shown = cm

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(shown)
    ax.set_title(title)
    ax.set_xlabel("Predicted zone")
    ax.set_ylabel("True zone")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{shown[i,j]:.2f}" if normalize else str(int(cm[i,j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plt.close("all")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@dataclass
class Config:
    batch_size: int
    max_epochs: int
    patience: int
    lr: float
    weight_decay: float
    dropout: float
    augmentation_strength: float
    lambda_domain: float
    lambda_align: float
    grl_strength: float
    seed: int
    device: str


def make_domain_map(sessions: list[str]) -> dict[str, int]:
    return {s: i for i, s in enumerate(sessions)}


def encode_labels(y_text: np.ndarray, label_map: dict[str, int]) -> np.ndarray:
    return np.asarray([label_map[str(y)] for y in y_text], dtype=np.int64)


def encode_domains(session: np.ndarray, domain_map: dict[str, int]) -> np.ndarray:
    return np.asarray([domain_map[str(s)] for s in session], dtype=np.int64)


def make_train_loader(
    X: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
    normalizer,
    *,
    n_classes: int,
    n_domains: int,
    cfg: Config,
) -> DataLoader:
    ds = DomainDataset(
        X,
        y,
        domain,
        normalizer,
        augment=True,
        augmentation_strength=cfg.augmentation_strength,
    )

    sampler = BalancedClassDomainBatchSampler(
        y,
        domain,
        n_classes=n_classes,
        n_domains=n_domains,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=0,
    )


def make_eval_loader(
    X: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
    normalizer,
    *,
    cfg: Config,
) -> DataLoader:
    ds = DomainDataset(
        X,
        y,
        domain,
        normalizer,
        augment=False,
        augmentation_strength=0.0,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )


def train_one_epoch(
    model: DomainInvariantCNN,
    loader: DataLoader,
    optimizer,
    *,
    cfg: Config,
    n_classes: int,
    n_domains: int,
) -> dict[str, float]:
    model.train()

    totals = defaultdict_float()
    count = 0

    for x, y, d in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        d = d.to(cfg.device)

        optimizer.zero_grad(set_to_none=True)

        zone_logits, domain_logits, emb = model(
            x,
            grl_strength=cfg.grl_strength,
        )

        zone_loss = F.cross_entropy(
            zone_logits,
            y,
            label_smoothing=0.05,
        )

        domain_loss = F.cross_entropy(
            domain_logits,
            d,
        )

        align_loss = cross_session_centroid_loss(
            emb,
            y,
            d,
            n_classes=n_classes,
            n_domains=n_domains,
        )

        loss = (
            zone_loss
            + cfg.lambda_domain * domain_loss
            + cfg.lambda_align * align_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        n = len(y)
        count += n

        totals["loss"] += float(loss.item()) * n
        totals["zone_loss"] += float(zone_loss.item()) * n
        totals["domain_loss"] += float(domain_loss.item()) * n
        totals["align_loss"] += float(align_loss.item()) * n
        totals["zone_correct"] += int(
            (zone_logits.argmax(1) == y).sum().item()
        )
        totals["domain_correct"] += int(
            (domain_logits.argmax(1) == d).sum().item()
        )

    return {
        "loss": totals["loss"] / count,
        "zone_loss": totals["zone_loss"] / count,
        "domain_loss": totals["domain_loss"] / count,
        "align_loss": totals["align_loss"] / count,
        "zone_accuracy": totals["zone_correct"] / count,
        "domain_accuracy": totals["domain_correct"] / count,
    }


def defaultdict_float():
    from collections import defaultdict
    return defaultdict(float)


@torch.no_grad()
def evaluate(
    model: DomainInvariantCNN,
    loader: DataLoader,
    *,
    cfg: Config,
    labels: list[str],
) -> dict[str, Any]:
    model.eval()

    y_all = []
    d_all = []
    pred_all = []
    dpred_all = []
    prob_all = []

    for x, y, d in loader:
        x = x.to(cfg.device)

        zone_logits, domain_logits, _ = model(
            x,
            grl_strength=0.0,
        )

        probs = torch.softmax(zone_logits, dim=1)
        pred = probs.argmax(1)
        dpred = domain_logits.argmax(1)

        y_all.append(y.numpy())
        d_all.append(d.numpy())
        pred_all.append(pred.cpu().numpy())
        dpred_all.append(dpred.cpu().numpy())
        prob_all.append(probs.cpu().numpy())

    y_idx = np.concatenate(y_all)
    d_idx = np.concatenate(d_all)
    pred_idx = np.concatenate(pred_all)
    dpred_idx = np.concatenate(dpred_all)
    probs = np.concatenate(prob_all)

    y_text = index_to_text(y_idx, labels)
    pred_text = index_to_text(pred_idx, labels)

    return {
        "y_idx": y_idx,
        "d_idx": d_idx,
        "pred_idx": pred_idx,
        "domain_pred_idx": dpred_idx,
        "probabilities": probs,
        "y_text": y_text,
        "pred_text": pred_text,
        "zone_metrics": class_metrics(y_text, pred_text),
        "domain_accuracy": float(
            accuracy_score(d_idx, dpred_idx)
        ),
    }


def choose_inner_trial(
    allowed_trial_numbers: list[int],
    seed: int,
) -> int:
    rng = np.random.default_rng(seed)
    return int(rng.choice(allowed_trial_numbers))


def select_epochs(
    data: RawWindowSet,
    outer_train_mask: np.ndarray,
    *,
    labels: list[str],
    label_map: dict[str, int],
    domain_map: dict[str, int],
    cfg: Config,
    seed: int,
) -> tuple[int, dict[str, Any]]:
    trial_numbers = sorted(
        set(data.trial_number[outer_train_mask].tolist())
    )

    if len(trial_numbers) < 2:
        raise RuntimeError("Need >=2 trial numbers for inner validation.")

    inner_val_trial = choose_inner_trial(trial_numbers, seed)

    inner_val = (
        outer_train_mask
        & (data.trial_number == inner_val_trial)
    )
    inner_train = outer_train_mask & ~inner_val

    y_idx = encode_labels(data.y_text, label_map)
    d_idx = encode_domains(data.session, domain_map)

    normalizer = compute_normalizer(
        data.X[inner_train],
        data.rssi[inner_train],
    )

    train_loader = make_train_loader(
        data.X[inner_train],
        y_idx[inner_train],
        d_idx[inner_train],
        normalizer,
        n_classes=len(labels),
        n_domains=len(domain_map),
        cfg=cfg,
    )

    val_loader = make_eval_loader(
        data.X[inner_val],
        y_idx[inner_val],
        d_idx[inner_val],
        normalizer,
        cfg=cfg,
    )

    set_seed(seed)

    model = DomainInvariantCNN(
        len(labels),
        len(domain_map),
        cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_epoch = 1
    best_f1 = -1.0
    best_acc = -1.0
    patience_left = cfg.patience
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            cfg=cfg,
            n_classes=len(labels),
            n_domains=len(domain_map),
        )

        va = evaluate(
            model,
            val_loader,
            cfg=cfg,
            labels=labels,
        )

        vf1 = va["zone_metrics"]["macro_f1"]
        vacc = va["zone_metrics"]["accuracy"]

        history.append(
            {
                "epoch": epoch,
                **tr,
                "val_zone_accuracy": vacc,
                "val_zone_macro_f1": vf1,
                "val_domain_accuracy": va["domain_accuracy"],
            }
        )

        print(
            f"    epoch {epoch:03d} | "
            f"zone train={tr['zone_accuracy']:.3f} "
            f"domain train={tr['domain_accuracy']:.3f} | "
            f"val zone={vacc:.3f} F1={vf1:.3f} "
            f"val domain={va['domain_accuracy']:.3f} | "
            f"align={tr['align_loss']:.4f}"
        )

        improved = (
            vf1 > best_f1 + 1e-6
            or (
                abs(vf1 - best_f1) <= 1e-6
                and vacc > best_acc
            )
        )

        if improved:
            best_f1 = vf1
            best_acc = vacc
            best_epoch = epoch
            patience_left = cfg.patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            print(f"    early stop; best epoch={best_epoch}")
            break

    return best_epoch, {
        "inner_validation_trial_number": inner_val_trial,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_accuracy": best_acc,
        "history": history,
    }


def fit_full(
    data: RawWindowSet,
    train_mask: np.ndarray,
    *,
    epochs: int,
    labels: list[str],
    label_map: dict[str, int],
    domain_map: dict[str, int],
    cfg: Config,
    seed: int,
):
    y_idx = encode_labels(data.y_text, label_map)
    d_idx = encode_domains(data.session, domain_map)

    normalizer = compute_normalizer(
        data.X[train_mask],
        data.rssi[train_mask],
    )

    loader = make_train_loader(
        data.X[train_mask],
        y_idx[train_mask],
        d_idx[train_mask],
        normalizer,
        n_classes=len(labels),
        n_domains=len(domain_map),
        cfg=cfg,
    )

    set_seed(seed)

    model = DomainInvariantCNN(
        len(labels),
        len(domain_map),
        cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history = []

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(
            model,
            loader,
            optimizer,
            cfg=cfg,
            n_classes=len(labels),
            n_domains=len(domain_map),
        )
        history.append({"epoch": epoch, **tr})

        print(
            f"    refit {epoch:03d}/{epochs:03d} | "
            f"zone={tr['zone_accuracy']:.3f} "
            f"domain={tr['domain_accuracy']:.3f} "
            f"align={tr['align_loss']:.4f}"
        )

    return model, normalizer, history


def evaluate_mask(
    model,
    normalizer,
    data,
    mask,
    *,
    labels,
    label_map,
    domain_map,
    cfg,
):
    y_idx = encode_labels(data.y_text, label_map)
    d_idx = encode_domains(data.session, domain_map)

    loader = make_eval_loader(
        data.X[mask],
        y_idx[mask],
        d_idx[mask],
        normalizer,
        cfg=cfg,
    )

    result = evaluate(
        model,
        loader,
        cfg=cfg,
        labels=labels,
    )

    xy = probability_weighted_xy(
        result["probabilities"],
        labels,
    )
    errors = xy_errors(
        result["y_text"],
        xy,
    )
    result["xy"] = summarize_xy(errors)
    result["xy_errors"] = errors

    return result


# ---------------------------------------------------------------------------
# Development 5-fold CV on S1+S2
# ---------------------------------------------------------------------------

def run_cv(
    data: RawWindowSet,
    sessions: list[str],
    *,
    cfg: Config,
    output_dir: Path,
) -> None:
    labels, label_map = label_to_index_map(data.y_text)
    domain_map = make_domain_map(sessions)

    fold_rows = []
    all_true = []
    all_pred = []

    for fold in range(1, 6):
        print("\n" + "=" * 78)
        print(
            f"DOMAIN CNN V2 FOLD {fold}/5 — "
            f"hold out trial {fold} from ALL development sessions"
        )
        print("=" * 78)

        test_mask = data.trial_number == fold
        train_mask = ~test_mask

        epochs, inner = select_epochs(
            data,
            train_mask,
            labels=labels,
            label_map=label_map,
            domain_map=domain_map,
            cfg=cfg,
            seed=cfg.seed + fold * 31,
        )

        print(
            f"[fold {fold}] selected {epochs} epochs; "
            f"refitting all outer-training trials..."
        )

        model, normalizer, _ = fit_full(
            data,
            train_mask,
            epochs=epochs,
            labels=labels,
            label_map=label_map,
            domain_map=domain_map,
            cfg=cfg,
            seed=cfg.seed + 1000 + fold,
        )

        ev = evaluate_mask(
            model,
            normalizer,
            data,
            test_mask,
            labels=labels,
            label_map=label_map,
            domain_map=domain_map,
            cfg=cfg,
        )

        m = ev["zone_metrics"]

        fold_rows.append(
            {
                "fold": fold,
                "selected_epochs": epochs,
                "window_accuracy": m["accuracy"],
                "window_macro_f1": m["macro_f1"],
                "window_balanced_accuracy": m["balanced_accuracy"],
                "domain_accuracy": ev["domain_accuracy"],
                "xy_median_m": ev["xy"]["xy_median_m"],
                "xy_p90_m": ev["xy"]["xy_p90_m"],
            }
        )

        all_true.extend(ev["y_text"].tolist())
        all_pred.extend(ev["pred_text"].tolist())

        print(
            f"Fold {fold}: zone acc={m['accuracy']:.3f} "
            f"macroF1={m['macro_f1']:.3f} "
            f"domain acc={ev['domain_accuracy']:.3f}"
        )

    all_true = np.asarray(all_true, dtype=object)
    all_pred = np.asarray(all_pred, dtype=object)
    overall = class_metrics(all_true, all_pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fold_metrics.csv", fold_rows)

    save_confusion(
        all_true,
        all_pred,
        labels,
        path=output_dir / "confusion_matrix_normalized.png",
        title="Domain-Invariant CNN V2 — S1+S2 5-fold development CV",
        normalize=True,
    )

    acc = np.asarray([r["window_accuracy"] for r in fold_rows])
    f1 = np.asarray([r["window_macro_f1"] for r in fold_rows])

    summary = {
        "mode": "development_cv",
        "sessions": sessions,
        "warning": (
            "This CV tests unseen trials from known development sessions. "
            "It is NOT an unseen-session estimate."
        ),
        "model": "DomainInvariantCNN_V2",
        "lambda_domain": cfg.lambda_domain,
        "lambda_align": cfg.lambda_align,
        "grl_strength": cfg.grl_strength,
        "augmentation_strength": cfg.augmentation_strength,
        "window_accuracy_mean": float(acc.mean()),
        "window_accuracy_std": float(acc.std(ddof=1)),
        "window_macro_f1_mean": float(f1.mean()),
        "window_macro_f1_std": float(f1.std(ddof=1)),
        "aggregate_window_metrics": overall,
        "folds": fold_rows,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("DOMAIN-INVARIANT CNN V2 DEVELOPMENT CV SUMMARY")
    print("=" * 78)
    print(
        f"Window accuracy: {acc.mean():.3f} ± {acc.std(ddof=1):.3f}"
    )
    print(
        f"Window macro F1: {f1.mean():.3f} ± {f1.std(ddof=1):.3f}"
    )
    print(
        f"Aggregate accuracy: {overall['accuracy']:.3f}"
    )
    print(
        "IMPORTANT: this is unseen-trial / known-session performance, "
        "not unseen-session performance."
    )
    print(f"Saved: {output_dir}")


# ---------------------------------------------------------------------------
# Final untouched-session test
# ---------------------------------------------------------------------------

def run_final(
    development_data: RawWindowSet,
    test_data: RawWindowSet,
    sessions: list[str],
    test_session: str,
    *,
    cfg: Config,
    output_dir: Path,
) -> None:
    # Epoch selection happens ONLY on S1/S2.
    labels, label_map = label_to_index_map(development_data.y_text)
    domain_map = make_domain_map(sessions)

    dev_mask = np.ones(len(development_data.y_text), dtype=bool)

    epochs, inner = select_epochs(
        development_data,
        dev_mask,
        labels=labels,
        label_map=label_map,
        domain_map=domain_map,
        cfg=cfg,
        seed=cfg.seed + 777,
    )

    print(
        f"\nSelected {epochs} epochs using DEVELOPMENT sessions only."
    )
    print("Refitting on all development sessions...")

    model, normalizer, history = fit_full(
        development_data,
        dev_mask,
        epochs=epochs,
        labels=labels,
        label_map=label_map,
        domain_map=domain_map,
        cfg=cfg,
        seed=cfg.seed + 1777,
    )

    # Test-domain IDs are not meaningful to the adversary because S3 was not a
    # training domain. For zone evaluation we only need a dummy domain array.
    # Build a separate simple eval dataset rather than pretending S3=S1/S2.
    y_test_idx = encode_labels(test_data.y_text, label_map)
    dummy_domain = np.zeros(len(y_test_idx), dtype=np.int64)

    test_loader = make_eval_loader(
        test_data.X,
        y_test_idx,
        dummy_domain,
        normalizer,
        cfg=cfg,
    )

    ev = evaluate(
        model,
        test_loader,
        cfg=cfg,
        labels=labels,
    )

    # Ignore domain accuracy on unseen S3; it has no interpretation.
    xy = probability_weighted_xy(ev["probabilities"], labels)
    errors = xy_errors(ev["y_text"], xy)
    xy_summary = summarize_xy(errors)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_confusion(
        ev["y_text"],
        ev["pred_text"],
        labels,
        path=output_dir / "confusion_matrix_normalized.png",
        title=f"Domain-Invariant CNN V2 — S1+S2 -> {test_session}",
        normalize=True,
    )

    m = ev["zone_metrics"]

    checkpoint_path = MODEL_DIR / (
        f"csi_cnn_v2_{'_'.join(s.lower() for s in sessions)}"
        f"_to_{test_session.lower()}.pt"
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "labels": labels,
            "development_sessions": sessions,
            "test_session": test_session,
            "selected_epochs": epochs,
            "config": {
                "lambda_domain": cfg.lambda_domain,
                "lambda_align": cfg.lambda_align,
                "grl_strength": cfg.grl_strength,
                "augmentation_strength": cfg.augmentation_strength,
                "dropout": cfg.dropout,
            },
            "normalizer": {
                "csi_mean": normalizer.csi_mean.tolist(),
                "csi_std": normalizer.csi_std.tolist(),
            },
        },
        checkpoint_path,
    )

    summary = {
        "mode": "final_untouched_session",
        "development_sessions": sessions,
        "test_session": test_session,
        "scientific_note": (
            "Epoch selection and all fitting used development sessions only. "
            "If model/hyperparameters are changed after inspecting this test, "
            "the test session is no longer untouched."
        ),
        "selected_epochs": epochs,
        "window_metrics": m,
        "xy": xy_summary,
        "inner_validation": inner,
        "saved_model": str(checkpoint_path),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("DOMAIN-INVARIANT CNN V2 FINAL SESSION RESULTS")
    print("=" * 78)
    print(
        f"Train {sessions} -> Test {test_session}"
    )
    print(
        f"Window accuracy={m['accuracy']:.3f} "
        f"macroF1={m['macro_f1']:.3f} "
        f"medianXY={xy_summary['xy_median_m']:.3f} m "
        f"p90XY={xy_summary['xy_p90_m']:.3f} m"
    )
    print(f"Selected epochs: {epochs}")
    print(f"Saved: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_sessions(
    raw_dir: Path,
    sessions: list[str],
    *,
    window: float,
    hop: float,
    target_steps: int,
    min_packets: int,
) -> RawWindowSet:
    sets = []

    for session in sessions:
        paths = trial_paths_for_session(raw_dir, session)
        print(
            f"\nLoading {session} raw CSI ({len(paths)} trials)..."
        )
        sets.append(
            gather_raw_windows(
                paths,
                window_seconds=window,
                hop_seconds=hop,
                target_steps=target_steps,
                min_packets_per_link=min_packets,
            )
        )

    return combine_sets(sets)


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=["cv", "final"],
        required=True,
    )
    p.add_argument(
        "--sessions",
        nargs="+",
        default=["S1", "S2"],
        help="Development sessions used for training/domain alignment.",
    )
    p.add_argument(
        "--test-session",
        default="S3",
        help="Untouched final session for --mode final.",
    )
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)

    p.add_argument("--window", type=float, default=WINDOW_SECONDS)
    p.add_argument("--hop", type=float, default=HOP_SECONDS)
    p.add_argument("--target-steps", type=int, default=32)
    p.add_argument(
        "--min-packets",
        type=int,
        default=MIN_PACKETS_PER_LINK,
    )

    p.add_argument("--batch-size", type=int, default=80)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--augmentation-strength", type=float, default=1.0)

    # Frozen V2 defaults based ONLY on S1/S2 diagnostics.
    p.add_argument("--lambda-domain", type=float, default=0.10)
    p.add_argument("--lambda-align", type=float, default=0.25)
    p.add_argument("--grl-strength", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu"], default="cpu")

    args = p.parse_args()

    if len(args.sessions) < 2:
        raise SystemExit(
            "V2 domain-invariant training requires at least 2 development sessions."
        )

    set_seed(args.seed)

    cfg = Config(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        augmentation_strength=args.augmentation_strength,
        lambda_domain=args.lambda_domain,
        lambda_align=args.lambda_align,
        grl_strength=args.grl_strength,
        seed=args.seed,
        device=args.device,
    )

    print("\n" + "=" * 78)
    print("TriCSI MULTI-SESSION DOMAIN-INVARIANT CNN V2")
    print("=" * 78)
    print(f"Development sessions : {args.sessions}")
    print("Input                : 4 x 32 x 64")
    print("RSSI branch          : OFF")
    print(f"lambda_domain        : {cfg.lambda_domain}")
    print(f"lambda_align         : {cfg.lambda_align}")
    print(f"GRL strength         : {cfg.grl_strength}")
    print(f"Device               : {cfg.device}")
    print("=" * 78)

    start = time.time()

    development_data = load_sessions(
        args.raw_dir,
        args.sessions,
        window=args.window,
        hop=args.hop,
        target_steps=args.target_steps,
        min_packets=args.min_packets,
    )

    print(
        f"\nDevelopment windows: {len(development_data.y_text)}"
    )
    print(
        f"Development trials : {len(set(development_data.group))}"
    )

    if args.mode == "cv":
        out = RESULTS_DIR / (
            "cnn_v2_domain_cv_"
            + "_".join(s.lower() for s in args.sessions)
        )

        run_cv(
            development_data,
            args.sessions,
            cfg=cfg,
            output_dir=out,
        )

    else:
        if args.test_session in args.sessions:
            raise SystemExit(
                "--test-session must not be one of --sessions."
            )

        test_data = load_sessions(
            args.raw_dir,
            [args.test_session],
            window=args.window,
            hop=args.hop,
            target_steps=args.target_steps,
            min_packets=args.min_packets,
        )

        out = RESULTS_DIR / (
            "cnn_v2_domain_"
            + "_".join(s.lower() for s in args.sessions)
            + "_to_"
            + args.test_session.lower()
        )

        run_final(
            development_data,
            test_data,
            args.sessions,
            args.test_session,
            cfg=cfg,
            output_dir=out,
        )

    print(
        f"\nTotal elapsed: {(time.time() - start)/60.0:.1f} min"
    )


if __name__ == "__main__":
    main()
