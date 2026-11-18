"""Empirical, low-rank CSI session generator, fitted on source data only.

This generates perturbations of measured log-amplitude representations. It
does not solve electromagnetic propagation or create verified human positions.
Source label preservation is a hypothesis to validate on real held-out sessions.
"""
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np


@dataclass
class SessionGenerator:
    basis: list
    marginal_std: list
    noise_std: list
    residual_scale: list
    gain_std: float
    matched_source_pairs: int
    source_sessions: list

    @classmethod
    def fit(cls, source, calibration, max_rank=4):
        """Both arguments must contain source sessions only; source is calibrated.

        A matched observation is a *whole-trial* summary. Overlapping windows
        are aggregated before estimating cross-session residual differences.
        Rank is capped because source-session counts are small.
        """
        sessions = sorted(set(source.session.astype(str)))
        if len(sessions) < 2:
            raise ValueError("Generator requires at least two source sessions.")
        if set(calibration.session.astype(str)) != set(sessions):
            raise ValueError("Calibration must contain exactly the source sessions.")
        if np.any(calibration.y_text != "EMPTY"):
            raise ValueError("Generator calibration must be Empty only.")
        profiles = {}
        for group in sorted(set(source.group.astype(str))):
            mask = source.group == group
            i = np.flatnonzero(mask)[0]
            key = (str(source.session[i]), str(source.y_text[i]),
                   str(source.orientation[i]))
            # Median window summaries avoid giving long trials more influence.
            v = np.median(source.X[mask][:, [0, 2]], axis=(0, 2))
            profiles.setdefault(key, []).append(v)
        profiles = {key: np.mean(values, axis=0) for key, values in profiles.items()}
        differences = []
        conditions = sorted({(k[1], k[2]) for k in profiles if k[1] != "EMPTY"})
        for a, b in combinations(sessions, 2):
            for label, orientation in conditions:
                ka, kb = (a, label, orientation), (b, label, orientation)
                if ka in profiles and kb in profiles:
                    differences.append((profiles[ka] - profiles[kb]) / np.sqrt(2))
        if len(differences) < 2:
            raise ValueError("Need >=2 matched source zone/orientation pairs; inspect inventory.")
        d = np.stack(differences).astype(np.float64)
        # Balanced signs give a zero-centered empirical perturbation distribution.
        # No claim is made that these differences are exclusively nuisance effects.
        matrix = np.concatenate([d, -d]).reshape(2 * len(d), -1)
        _, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
        rank = min(max_rank, int(np.sum(singular > 1e-10)))
        basis = (vectors[:rank] * (singular[:rank, None] / np.sqrt(len(matrix))))
        k = source.X.shape[-1]
        basis = basis.reshape(rank, 2, k)
        # Limit extreme perturbations relative to source signal variation.
        scale = np.maximum(np.std(source.X[:, [0, 2]], axis=(0, 2)), 1e-4)
        marginal = np.sqrt(np.sum(basis ** 2, axis=0))
        # Noise differences are within-window, within-source-Empty only.
        temporal = np.diff(calibration.X[:, [0, 2]], axis=2)
        noise = np.std(temporal, axis=(0, 2)) / np.sqrt(2)
        noise = np.minimum(noise, 0.2 * scale)
        gain = float(np.clip(np.median(marginal / scale) * 0.1, 0, 0.15))
        return cls(basis.tolist(), marginal.tolist(), noise.tolist(), scale.tolist(),
                   gain, len(differences), sessions)

    def to_dict(self):
        return asdict(self)

    def sample(self, rng, strength=1.0, method="session"):
        if strength < 0 or not np.isfinite(strength):
            raise ValueError("Augmentation strength must be finite and nonnegative.")
        std = np.asarray(self.marginal_std)
        if method == "session":
            b = np.asarray(self.basis).reshape(-1, *std.shape)
            offset = np.einsum("r,rlk->lk", rng.normal(size=len(b)), b)
        elif method == "generic":
            # Matched marginal variances, but no learned cross-bin/link structure.
            offset = rng.normal(size=std.shape) * std
        else:
            raise ValueError(f"Unknown generator method: {method}")
        cap = 0.5 * np.asarray(self.residual_scale)
        offset = np.clip(offset * strength, -cap, cap)
        gain = np.clip(1 + rng.normal(size=(2, 1)) * self.gain_std * strength, 0.5, 1.5)
        return {"offset": offset, "gain": gain, "strength": float(strength)}

    def transform(self, x, style, rng):
        """Alter calibrated absolute channels and consistently update spectral shape.

        The centered channels retain the original room-dependent baseline. RSSI
        is deliberately unchanged by this CSI-only intervention. This is not a
        physically complete synthetic radio packet.
        """
        result = np.asarray(x, dtype=np.float32).copy()
        strength = style["strength"]
        if strength == 0:
            return result
        residual = result[[0, 2]].astype(np.float64)
        noise_scale = np.asarray(self.noise_std)[:, None, :] * strength
        innovation = rng.normal(size=residual.shape) * noise_scale
        noise = np.zeros_like(innovation)
        rho = 0.8
        noise[:, 0] = innovation[:, 0]
        for t in range(1, residual.shape[1]):
            noise[:, t] = rho * noise[:, t - 1] + np.sqrt(1 - rho ** 2) * innovation[:, t]
        changed = (residual * style["gain"][:, None, :]
                   + style["offset"][:, None, :] + noise)
        delta = changed - residual
        result[[0, 2]] = changed
        centered = result[[1, 3]].astype(np.float64) + delta
        result[[1, 3]] = centered - np.median(centered, axis=-1, keepdims=True)
        if not np.all(np.isfinite(result)):
            raise ValueError("Generator produced non-finite values.")
        return result
