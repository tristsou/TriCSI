"""Matched training using the existing TriCSI-Anchor network and losses."""
import copy
import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import LABELS


def load_anchor(project_root):
    root = Path(project_root).resolve()
    sys.path.insert(0, str(root))
    anchor = importlib.import_module("train_tricsi_anchor_final")
    if Path(anchor.__file__).resolve().parent != root:
        raise ValueError("A different project's Anchor module is already loaded.")
    if list(anchor.LABELS) != LABELS:
        raise ValueError("This experiment expects Empty plus Z1-Z9 in the original order.")
    return anchor


class AugmentedWindows(Dataset):
    def __init__(self, raw, norm, anchor, method, generator, strength, seed):
        self.raw, self.norm, self.anchor = raw, norm, anchor
        self.method, self.generator = method, generator
        self.strength, self.seed, self.epoch = strength, seed, 0
        self.session_map = {s: i for i, s in enumerate(sorted(set(raw.session.astype(str))))}
        self.style = None

    def start_epoch(self, epoch):
        self.epoch = epoch
        if self.method in ("generic", "session"):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 71]))
            # One session-level style across all classes this epoch. Its
            # assignment is independent of the sample's label and orientation.
            self.style = self.generator.sample(rng, self.strength, self.method)

    def __len__(self):
        return len(self.raw.X)

    def __getitem__(self, i):
        raw = self.raw
        x = raw.X[i].copy()
        if self.method in ("generic", "session"):
            identity = int.from_bytes(hashlib.sha256(str(raw.group[i]).encode()).digest()[:8], "little")
            trial_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, identity]))
            # Half of trial views remain real. Every real trial stays available.
            if trial_rng.random() < 0.5:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, identity, int(raw.window_index[i]), 19]))
                x = self.generator.transform(x, self.style, rng)
        x = torch.from_numpy((x - self.norm.csi_mean[:, None, None]) / self.norm.csi_std[:, None, None]).float()
        rssi = torch.from_numpy((raw.rssi[i].copy() - self.norm.rssi_mean) / self.norm.rssi_std).float()
        # Existing input augmentation is exactly retained in B, C and D.
        if self.method != "none":
            if torch.rand(()) < 0.35:
                x = torch.roll(x, shifts=int(torch.randint(-1, 2, (1,)).item()), dims=1)
            if torch.rand(()) < 0.25:
                width = int(torch.randint(2, 6, (1,)).item())
                start = int(torch.randint(0, max(1, x.shape[2] - width + 1), (1,)).item())
                x[:, :, start:start + width] = 0
            if torch.rand(()) < 0.60:
                x += 0.02 * torch.randn_like(x)
        label = str(raw.y_text[i])
        orientation = str(raw.orientation[i])
        return (x, rssi, torch.tensor(LABELS.index(label)),
                torch.tensor(float(label != "EMPTY")),
                torch.from_numpy(self.anchor.zone_xy_norm(label)),
                torch.tensor(self.session_map[str(raw.session[i])]),
                torch.tensor(self.anchor.ORI_TO_INDEX.get(orientation, self.anchor.ORI_TO_INDEX["NA"])))


def train(source, anchor, generator, *, method, strength, seed, epochs, batch_size, lr,
          on_epoch=None, eval_every=5):
    if eval_every <= 0:
        raise ValueError("eval_every must be positive.")
    anchor.set_seed(seed)
    norm = anchor.compute_norm(source)  # Real source data only, identical across methods.
    model = anchor.TriCSIAdaptNet(n_classes=len(LABELS), embed_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dataset = AugmentedWindows(source, norm, anchor, method, generator, strength, seed)
    updates = 0
    best_rank, best_state, selection = None, None, None
    for epoch in range(epochs):
        dataset.start_epoch(epoch)
        # Independent sampler RNG ensures the same batches across conditions,
        # regardless of random numbers consumed inside input augmentation.
        sampler = torch.Generator().manual_seed(seed + epoch * 100003)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=sampler,
                            num_workers=0, drop_last=False)
        model.train()
        total = count = 0
        for x, rssi, y, occ, xy, session, orientation in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(x, rssi, link_dropout=True)
            ce = F.cross_entropy(out["zone_logits"], y)
            occupancy = F.binary_cross_entropy_with_logits(out["occ_logit"], occ)
            occupied = occ > 0.5
            position = F.smooth_l1_loss(out["xy"][occupied], xy[occupied]) if occupied.any() else ce.new_tensor(0.)
            spatial = anchor.spatial_similarity_loss(out["embedding"], y, session, orientation)
            loss = ce + 0.25 * occupancy + 0.40 * position + 0.55 * spatial
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            updates += 1
            total += float(loss.detach()) * len(y)
            count += len(y)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"{method} seed={seed} epoch={epoch+1}/{epochs} loss={total/count:.4f}", flush=True)
        if on_epoch is not None and (epoch == 0 or (epoch + 1) % eval_every == 0 or epoch + 1 == epochs):
            was_training = model.training
            # Even an unshuffled inference DataLoader consumes a PyTorch seed.
            # Restore RNG state so validation cannot change subsequent updates.
            with torch.random.fork_rng(devices=[]):
                model.eval()
                result = on_epoch(model, norm, epoch + 1, updates, total/count)
            model.train(was_training)
            rank = tuple(float(v) for v in result["rank"])
            if not rank or not np.isfinite(rank).all():
                raise ValueError("Invalid development checkpoint score.")
            # Strict comparison retains the earlier epoch on an exact tie.
            if best_rank is None or rank > best_rank:
                best_rank, best_state = rank, copy.deepcopy(model.state_dict())
                selection = {"selected_epoch": epoch + 1, "selected_updates": updates,
                             "selected_development": result}
    if best_state is not None:
        model.load_state_dict(best_state)
    if selection is None:
        selection = {"selected_epoch": epochs, "selected_updates": updates,
                     "selected_development": None}
    selection["max_epochs_run"] = epochs
    return model.eval(), norm, updates, selection


@torch.no_grad()
def source_prototypes(model, norm, source, anchor):
    embedding = anchor.embed_raw(model, source, norm)["embedding"]
    mask = source.y_text != "EMPTY"
    return anchor.make_prototypes(embedding[mask], source.y_text[mask])


@torch.no_grad()
def predict_trials(model, norm, prototypes, target, anchor):
    out = anchor.embed_raw(model, target, norm)
    zone_probs = anchor.prototype_probabilities(out["embedding"], prototypes)
    rows = []
    for group in sorted(set(target.group.astype(str))):
        mask = target.group == group
        i = np.flatnonzero(mask)[0]
        occ = float(np.mean(out["occ_prob"][mask]))
        q = np.mean(zone_probs[mask], axis=0)
        pred = "EMPTY" if occ < 0.5 else anchor.OCCUPIED[int(np.argmax(q))]
        rows.append({"session": str(target.session[i]), "group": group,
                     "true": str(target.y_text[i]), "pred": pred,
                     "orientation": str(target.orientation[i]), "occupied_probability": occ})
    return rows


def save_checkpoint(path, model, norm, prototypes, **metadata):
    torch.save({"state_dict": model.state_dict(),
                "norm": {k: getattr(norm, k).tolist() for k in ("csi_mean", "csi_std", "rssi_mean", "rssi_std")},
                "prototypes": torch.as_tensor(prototypes), "metadata": metadata}, path)


def load_checkpoint(path, anchor):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = anchor.TriCSIAdaptNet(n_classes=len(LABELS), embed_dim=64)
    model.load_state_dict(saved["state_dict"], strict=True)
    norm = anchor.Norm(**{k: np.asarray(v, dtype=np.float32) for k, v in saved["norm"].items()})
    return model.eval(), norm, saved["prototypes"].numpy(), saved["metadata"]
