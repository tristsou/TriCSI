from __future__ import annotations

"""
Paper-guided CSI preprocessing for the TriCSI localization project.

Basis:
Hernandez & Bulut, "WiFi Sensing on the Edge: Signal Processing
Techniques and Challenges for Real-World Systems", IEEE Communications
Surveys & Tutorials, 2023.

Default output variants
-----------------------
1) amplitude_pca10
   CSI I/Q -> amplitude -> PCA(10)

2) hampel_pca10
   CSI I/Q -> amplitude -> Hampel outlier filter -> PCA(10)

3) savgol_pca10
   CSI I/Q -> amplitude -> Savitzky-Golay smoothing -> PCA(10)

Why three variants?
-------------------
The paper found PCA to be the most consistently useful dimensionality
reduction method, but it explicitly found denoising to be application
dependent. Therefore we do NOT silently stack several denoisers and call
that "better". We export clean, directly comparable variants instead.

Important leakage rule
----------------------
PCA is fitted ONLY on --fit-sessions. It is then applied to every
--apply-session. For a fair S1 -> S2 experiment use:

    python preprocess_csi_paper.py --fit-sessions S1 --apply-sessions S1 S2

S2 is transformed by the PCA learned from S1; S2 is never used to fit it.

For later S1+S2 development training you can create a different export:

    python preprocess_csi_paper.py --fit-sessions S1 S2 --apply-sessions S1 S2 --tag fit_S1_S2

Do not use an S1+S2-fitted PCA to claim a fair S1 -> S2 result.

Output
------
data/preprocessed_paper/<tag>/
    manifest.json
    pca_explained_variance.csv
    pca_explained_variance.png
    models/
       ...
    amplitude_pca10/S1/*.csv
    amplitude_pca10/S2/*.csv
    hampel_pca10/S1/*.csv
    hampel_pca10/S2/*.csv
    savgol_pca10/S1/*.csv
    savgol_pca10/S2/*.csv

Each output trial CSV preserves the original non-IQ metadata and adds:
    amp_00 ... amp_63      cleaned amplitude used for that variant
    pc_01  ... pc_10       PCA components

PCA already performs the mean-centering required by the PCA transform.
Per-component z-score/min-max scaling AFTER PCA is intentionally NOT
performed by default because the paper reports that feature scaling after
PCA reduced accuracy in all three of its evaluated scales.
"""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

try:
    from project_config import RAW_DATA_DIR
except Exception:
    RAW_DATA_DIR = Path("data") / "raw"


IQ_RE = re.compile(r"^iq_(\d+)$", re.IGNORECASE)
EPS = 1e-12


@dataclass
class TrialData:
    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]
    iq_columns: list[str]
    link_field: str
    amplitude_by_link: dict[str, np.ndarray]
    row_indices_by_link: dict[str, np.ndarray]


def find_case_insensitive(fieldnames: list[str], wanted: str) -> str:
    lookup = {x.lower(): x for x in fieldnames}
    if wanted.lower() not in lookup:
        raise ValueError(
            f"Required column '{wanted}' not found. "
            f"Available columns include: {fieldnames[:20]}"
        )
    return lookup[wanted.lower()]


def sorted_iq_columns(fieldnames: list[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for name in fieldnames:
        m = IQ_RE.match(name.strip())
        if m:
            found.append((int(m.group(1)), name))

    found.sort()
    columns = [name for _, name in found]

    if len(columns) < 128:
        raise ValueError(
            f"Expected at least 128 IQ columns, found {len(columns)}."
        )

    # The project stores LLTF in the first 128 raw signed values.
    return columns[:128]


def canonical_link(value: str) -> str | None:
    x = value.strip().lower()

    if x in {"left", "l", "tx_left", "tx-left"}:
        return "left"
    if x in {"right", "r", "tx_right", "tx-right"}:
        return "right"

    # Be permissive if the project uses labels containing LEFT/RIGHT.
    if "left" in x:
        return "left"
    if "right" in x:
        return "right"

    return None


def iq_rows_to_amplitude(
    rows: list[dict[str, str]],
    iq_columns: list[str],
    indices: np.ndarray,
) -> np.ndarray:
    raw = np.empty((len(indices), 128), dtype=np.float64)

    for out_i, row_i in enumerate(indices):
        row = rows[int(row_i)]
        raw[out_i] = [
            float(row[col])
            for col in iq_columns
        ]

    # ESP32-CSI-Tool LLTF raw order in this project:
    # imaginary, real, imaginary, real, ...
    imag = raw[:, 0::2]
    real = raw[:, 1::2]
    return np.hypot(imag, real)


def load_trial(path: Path) -> TrialData:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header: {path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    iq_columns = sorted_iq_columns(fieldnames)
    link_field = find_case_insensitive(fieldnames, "link")

    row_indices_by_link: dict[str, list[int]] = {
        "left": [],
        "right": [],
    }

    for i, row in enumerate(rows):
        link = canonical_link(row.get(link_field, ""))
        if link is not None:
            row_indices_by_link[link].append(i)

    idx = {
        link: np.asarray(v, dtype=np.int64)
        for link, v in row_indices_by_link.items()
    }

    for link in ("left", "right"):
        if len(idx[link]) == 0:
            raise ValueError(
                f"{path.name}: no rows found for {link} link."
            )

    amp = {
        link: iq_rows_to_amplitude(
            rows,
            iq_columns,
            idx[link],
        )
        for link in ("left", "right")
    }

    return TrialData(
        path=path,
        fieldnames=fieldnames,
        rows=rows,
        iq_columns=iq_columns,
        link_field=link_field,
        amplitude_by_link=amp,
        row_indices_by_link=idx,
    )


def discover_trials(raw_dir: Path, session: str) -> list[Path]:
    root = raw_dir / session
    if not root.exists():
        raise SystemExit(f"Missing session folder: {root}")

    paths = sorted(
        p for p in root.rglob("*.csv")
        if not p.name.lower().startswith("schedule")
    )

    if not paths:
        raise SystemExit(f"No trial CSV files found in {root}")

    return paths


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def hampel_filter_matrix(
    x: np.ndarray,
    *,
    window: int,
    threshold_mad: float,
) -> np.ndarray:
    """
    Trailing-window Hampel filter, applied independently to every subcarrier.

    Paper equation:
        keep x[t] if |x[t] - median| <= 3 * MAD
        otherwise replace with median.

    We retain threshold_mad as a CLI parameter; default is exactly 3.0.
    """
    if window < 1:
        return x.copy()

    y = x.copy()
    n, s = x.shape

    for t in range(n):
        start = max(0, t - window + 1)
        w = x[start:t + 1]

        med = np.median(w, axis=0)
        mad = np.median(np.abs(w - med), axis=0)

        delta = np.abs(x[t] - med)
        threshold = threshold_mad * mad

        # If MAD is exactly zero, only replace a value if it differs from
        # the constant local median. This avoids a 0/0 special case.
        is_outlier = np.where(
            mad > EPS,
            delta > threshold,
            delta > EPS,
        )

        y[t, is_outlier] = med[is_outlier]

    return y


def savgol_filter_matrix(
    x: np.ndarray,
    *,
    window: int,
    polyorder: int,
) -> np.ndarray:
    """
    Savitzky-Golay smoothing along TIME, independently for each subcarrier.
    """
    if len(x) < 3:
        return x.copy()

    w = int(window)
    if w % 2 == 0:
        w += 1

    # Window cannot exceed the sequence length and must be odd.
    max_w = len(x) if len(x) % 2 == 1 else len(x) - 1
    w = min(w, max_w)

    if w <= polyorder or w < 3:
        return x.copy()

    return savgol_filter(
        x,
        window_length=w,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    ).astype(np.float64)


def clean_amplitude(
    amplitude: np.ndarray,
    variant: str,
    *,
    hampel_window: int,
    hampel_threshold: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> np.ndarray:
    if variant == "amplitude_pca10":
        return amplitude.copy()

    if variant == "hampel_pca10":
        return hampel_filter_matrix(
            amplitude,
            window=hampel_window,
            threshold_mad=hampel_threshold,
        )

    if variant == "savgol_pca10":
        return savgol_filter_matrix(
            amplitude,
            window=savgol_window,
            polyorder=savgol_polyorder,
        )

    raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# PCA fit / transform
# ---------------------------------------------------------------------------

def fit_link_pca(
    matrices: list[np.ndarray],
    n_components: int,
) -> PCA:
    x = np.concatenate(matrices, axis=0)

    if x.shape[1] != 64:
        raise ValueError(f"Expected 64 subcarriers, got {x.shape[1]}.")

    if n_components > min(x.shape):
        raise ValueError(
            f"PCA n_components={n_components} exceeds "
            f"min(samples, features)={min(x.shape)}"
        )

    # sklearn PCA centers each subcarrier before projection. We intentionally
    # do NOT StandardScale subcarriers or whiten PCs.
    pca = PCA(
        n_components=n_components,
        svd_solver="full",
        whiten=False,
    )
    pca.fit(x)
    return pca


def transform_trial(
    trial: TrialData,
    cleaned: dict[str, np.ndarray],
    pcas: dict[str, PCA],
) -> dict[str, np.ndarray]:
    return {
        link: pcas[link].transform(cleaned[link])
        for link in ("left", "right")
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def output_fieldnames(
    trial: TrialData,
    n_components: int,
) -> list[str]:
    metadata = [
        x for x in trial.fieldnames
        if x not in set(trial.iq_columns)
    ]
    amp_cols = [f"amp_{i:02d}" for i in range(64)]
    pc_cols = [f"pc_{i+1:02d}" for i in range(n_components)]
    return metadata + amp_cols + pc_cols


def save_processed_trial(
    trial: TrialData,
    cleaned: dict[str, np.ndarray],
    transformed: dict[str, np.ndarray],
    output_path: Path,
    *,
    n_components: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = output_fieldnames(trial, n_components)
    iq_set = set(trial.iq_columns)

    # Map original row index -> position inside that link's cleaned matrix.
    lookup: dict[int, tuple[str, int]] = {}
    for link in ("left", "right"):
        for local_i, row_i in enumerate(trial.row_indices_by_link[link]):
            lookup[int(row_i)] = (link, local_i)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row_i, original in enumerate(trial.rows):
            link_info = lookup.get(row_i)
            if link_info is None:
                # Ignore rows that are not one of the two desired links.
                continue

            link, local_i = link_info

            out = {
                k: v
                for k, v in original.items()
                if k not in iq_set
            }

            amp = cleaned[link][local_i]
            pc = transformed[link][local_i]

            for j in range(64):
                out[f"amp_{j:02d}"] = f"{float(amp[j]):.8g}"

            for j in range(n_components):
                out[f"pc_{j+1:02d}"] = f"{float(pc[j]):.8g}"

            writer.writerow(out)


def write_explained_variance(
    rows: list[dict[str, Any]],
    output_csv: Path,
) -> None:
    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "link",
                "component",
                "explained_variance_ratio",
                "cumulative_explained_variance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_explained_variance(
    pca_models: dict[str, dict[str, PCA]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for variant, link_models in pca_models.items():
        for link, pca in link_models.items():
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            ax.plot(
                np.arange(1, len(cumulative) + 1),
                cumulative,
                marker="o",
                label=f"{variant} / {link}",
            )

    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.set_title("Paper-guided CSI PCA calibration")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-guided amplitude/denoising/PCA preprocessing."
    )
    parser.add_argument(
        "--fit-sessions",
        nargs="+",
        default=["S1"],
        help="Sessions allowed to fit PCA. Default: S1",
    )
    parser.add_argument(
        "--apply-sessions",
        nargs="+",
        default=["S1", "S2"],
        help="Sessions to transform/export. Default: S1 S2",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "preprocessed_paper",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Output subfolder name. Default is derived from fit sessions.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="Paper used d=10; accuracy plateaued around d=8. Default: 10",
    )

    # At ~33 packets/s, 7 packets ~= 0.21 s. These are conservative
    # project-specific starting windows, not claimed as the paper's optimum.
    parser.add_argument("--hampel-window", type=int, default=7)
    parser.add_argument("--hampel-threshold", type=float, default=3.0)
    parser.add_argument("--savgol-window", type=int, default=7)
    parser.add_argument("--savgol-polyorder", type=int, default=2)

    args = parser.parse_args()

    if args.pca_components < 1 or args.pca_components > 64:
        raise SystemExit("--pca-components must be between 1 and 64.")

    tag = args.tag
    if tag is None:
        tag = "fit_" + "_".join(args.fit_sessions)

    output_dir = args.output_root / tag
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        "amplitude_pca10",
        "hampel_pca10",
        "savgol_pca10",
    ]

    print("\n" + "=" * 78)
    print("PAPER-GUIDED CSI PREPROCESSING")
    print("=" * 78)
    print(f"Raw data          : {args.raw_dir}")
    print(f"PCA fit sessions  : {args.fit_sessions}")
    print(f"Apply sessions    : {args.apply_sessions}")
    print(f"PCA components    : {args.pca_components}")
    print("Post-PCA scaling  : NONE")
    print(
        f"Hampel            : trailing window={args.hampel_window}, "
        f"threshold={args.hampel_threshold} x MAD"
    )
    print(
        f"Savitzky-Golay    : window={args.savgol_window}, "
        f"polyorder={args.savgol_polyorder}"
    )
    print("=" * 78)

    # Cache every session needed either for fitting or output.
    needed_sessions = list(
        dict.fromkeys(args.fit_sessions + args.apply_sessions)
    )

    trials_by_session: dict[str, list[TrialData]] = {}

    for session in needed_sessions:
        paths = discover_trials(args.raw_dir, session)
        print(f"\nLoading {session}: {len(paths)} trial files")

        loaded = []
        for i, path in enumerate(paths, 1):
            print(f"  [{i:03d}/{len(paths):03d}] {path.name}")
            loaded.append(load_trial(path))

        trials_by_session[session] = loaded

    # Cache cleaned amplitude per variant/session/trial/link so we only
    # perform denoising once.
    cleaned_cache: dict[
        str,
        dict[str, list[dict[str, np.ndarray]]]
    ] = {}

    for variant in variants:
        cleaned_cache[variant] = {}

        print(f"\nPreparing variant: {variant}")

        for session in needed_sessions:
            cleaned_trials = []

            for trial in trials_by_session[session]:
                cleaned_trials.append(
                    {
                        link: clean_amplitude(
                            trial.amplitude_by_link[link],
                            variant,
                            hampel_window=args.hampel_window,
                            hampel_threshold=args.hampel_threshold,
                            savgol_window=args.savgol_window,
                            savgol_polyorder=args.savgol_polyorder,
                        )
                        for link in ("left", "right")
                    }
                )

            cleaned_cache[variant][session] = cleaned_trials

    # Fit separate LEFT and RIGHT PCA transforms using ONLY fit sessions.
    # This preserves the fact that the two TX->RX paths are physically
    # different links.
    pca_models: dict[str, dict[str, PCA]] = {}
    variance_rows: list[dict[str, Any]] = []

    for variant in variants:
        print(f"\nFitting PCA for {variant}...")

        pca_models[variant] = {}

        for link in ("left", "right"):
            matrices = []

            for session in args.fit_sessions:
                matrices.extend(
                    cleaned[link]
                    for cleaned in cleaned_cache[variant][session]
                )

            pca = fit_link_pca(
                matrices,
                n_components=args.pca_components,
            )
            pca_models[variant][link] = pca

            model_path = (
                model_dir
                / f"{variant}_{link}_pca{args.pca_components}.joblib"
            )
            joblib.dump(pca, model_path)

            cumulative = np.cumsum(pca.explained_variance_ratio_)
            print(
                f"  {link:5s}: PC1={pca.explained_variance_ratio_[0]:.3f}, "
                f"PC{args.pca_components} cumulative={cumulative[-1]:.3f}"
            )

            for i, (evr, cum) in enumerate(
                zip(
                    pca.explained_variance_ratio_,
                    cumulative,
                ),
                1,
            ):
                variance_rows.append(
                    {
                        "variant": variant,
                        "link": link,
                        "component": i,
                        "explained_variance_ratio": float(evr),
                        "cumulative_explained_variance": float(cum),
                    }
                )

    # Export transformed trial CSVs.
    total_files = 0

    for variant in variants:
        print(f"\nExporting {variant}...")

        for session in args.apply_sessions:
            session_trials = trials_by_session[session]
            session_clean = cleaned_cache[variant][session]

            for trial, cleaned in zip(session_trials, session_clean):
                transformed = transform_trial(
                    trial,
                    cleaned,
                    pca_models[variant],
                )

                out = (
                    output_dir
                    / variant
                    / session
                    / trial.path.name
                )

                save_processed_trial(
                    trial,
                    cleaned,
                    transformed,
                    out,
                    n_components=args.pca_components,
                )
                total_files += 1

    write_explained_variance(
        variance_rows,
        output_dir / "pca_explained_variance.csv",
    )

    plot_explained_variance(
        pca_models,
        output_dir / "pca_explained_variance.png",
    )

    manifest = {
        "paper": {
            "title": (
                "WiFi Sensing on the Edge: Signal Processing "
                "Techniques and Challenges for Real-World Systems"
            ),
            "authors": "Steven M. Hernandez and Eyuphan Bulut",
            "doi": "10.1109/COMST.2022.3209144",
        },
        "input": {
            "raw_dir": str(args.raw_dir),
            "fit_sessions": args.fit_sessions,
            "apply_sessions": args.apply_sessions,
            "lltf_raw_iq_values": 128,
            "complex_subcarriers": 64,
            "iq_order": "imaginary, real",
        },
        "processing": {
            "feature": "amplitude = sqrt(imag^2 + real^2)",
            "variants": variants,
            "pca_components": args.pca_components,
            "pca_whiten": False,
            "post_pca_scaling": "none",
            "hampel": {
                "window_packets": args.hampel_window,
                "threshold_mad": args.hampel_threshold,
            },
            "savitzky_golay": {
                "window_packets": args.savgol_window,
                "polyorder": args.savgol_polyorder,
            },
            "note": (
                "PCA fitted separately for LEFT and RIGHT links. "
                "This is a project-specific adaptation for the dual-link setup."
            ),
        },
        "leakage_rule": (
            "Only fit_sessions were used to fit PCA. "
            "apply_sessions were transformed using the fitted PCA."
        ),
        "output_files": {
            "trial_csv_count": total_files,
            "root": str(output_dir),
        },
    }

    with (output_dir / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Output root : {output_dir}")
    print(f"Trial CSVs  : {total_files}")
    print("Variants    :")
    for variant in variants:
        print(f"  - {variant}")
    print("Saved PCA models and explained-variance report.")
    print(
        "\nFor a fair S1->S2 test, use THIS fit_S1 export. "
        "Do not refit PCA on S2."
    )


if __name__ == "__main__":
    main()
