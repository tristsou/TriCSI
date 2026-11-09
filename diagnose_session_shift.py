from __future__ import annotations

"""
TriCSI S1-vs-S2 raw CSI session-shift diagnostic.

Purpose
-------
The engineered-feature RF generalized S1->S2 at ~31%, while the first raw
CSI CNN fell to ~9% cross-session and ~62.5% same-session 5-fold accuracy.
Before changing architecture, measure exactly how the raw CSI distribution
moves between sessions.

This script compares S1 and S2 for each link and each LLTF subcarrier.

Outputs include:
- absolute log-amplitude median profiles
- packet-centered spectral-shape median profiles
- per-subcarrier session shifts
- robust standardized shift ("robust effect size")
- global link gain shift
- residual frequency-dependent distortion after global gain removal
- left-minus-right differential fingerprint shift
- CSV/JSON summaries and plots

Run:
    python diagnose_session_shift.py --session-a S1 --session-b S2
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from csi_io import load_trial_csv
from project_config import RAW_DATA_DIR, RESULTS_DIR


MAD_TO_SIGMA = 1.4826
EPS = 1e-6


def trial_paths(raw_dir: Path, session: str) -> list[Path]:
    root = raw_dir / session
    if not root.exists():
        raise SystemExit(f"Missing session folder: {root}")

    paths = sorted(
        p for p in root.rglob("*.csv")
        if not p.name.startswith("schedule_")
    )
    if not paths:
        raise SystemExit(f"No trial CSVs found in {root}")
    return paths


def iq_to_logamp(iq: np.ndarray) -> np.ndarray:
    x = np.asarray(iq, dtype=np.float32)
    imag = x[:, 0::2]
    real = x[:, 1::2]
    amp = np.hypot(imag, real)
    return np.log(np.maximum(amp, 0.5)).astype(np.float32)


def collect_session(paths: list[Path]) -> dict[str, np.ndarray]:
    """
    Concatenate all packets in a session.

    Returns:
        L_abs, R_abs       [packets, 64]
        L_center, R_center [packets, 64]
        L_rssi, R_rssi     [packets]
    """
    out: dict[str, list[np.ndarray]] = {
        "L_abs": [],
        "R_abs": [],
        "L_center": [],
        "R_center": [],
        "L_rssi": [],
        "R_rssi": [],
    }

    for i, path in enumerate(paths, 1):
        print(f"[{i:03d}/{len(paths):03d}] {path.name}")
        trial = load_trial_csv(path)

        for prefix, side in (("L", trial["left"]), ("R", trial["right"])):
            if len(side["iq"]) == 0:
                continue

            abs_log = iq_to_logamp(side["iq"])
            centered = (
                abs_log
                - np.median(abs_log, axis=1, keepdims=True)
            ).astype(np.float32)

            out[f"{prefix}_abs"].append(abs_log)
            out[f"{prefix}_center"].append(centered)
            out[f"{prefix}_rssi"].append(
                np.asarray(side["rssi"], dtype=np.float32)
            )

    merged: dict[str, np.ndarray] = {}
    for key, arrays in out.items():
        if not arrays:
            raise RuntimeError(f"No data collected for {key}")
        merged[key] = np.concatenate(arrays, axis=0)

    return merged


def robust_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-column median, IQR, robust sigma.
    """
    med = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    iqr = q75 - q25
    mad = np.median(np.abs(x - med), axis=0)
    sigma = MAD_TO_SIGMA * mad
    return med, iqr, sigma


def pooled_robust_scale(
    sigma_a: np.ndarray,
    sigma_b: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        np.sqrt((sigma_a ** 2 + sigma_b ** 2) / 2.0),
        EPS,
    )


def summarize_vector(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_profiles(
    carriers: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    label_a: str,
    label_b: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(carriers, a, label=label_a)
    ax.plot(carriers, b, label=label_b)
    ax.set_xlabel("LLTF subcarrier index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plt.close("all")


def plot_shift(
    carriers: np.ndarray,
    shift: np.ndarray,
    *,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(carriers, shift)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("LLTF subcarrier index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plt.close("all")


def plot_effect(
    carriers: np.ndarray,
    effect: np.ndarray,
    *,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(carriers, effect)
    ax.axhline(1.0, linewidth=1)
    ax.set_xlabel("LLTF subcarrier index")
    ax.set_ylabel("|median shift| / pooled robust scale")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plt.close("all")


def analyze_representation(
    a: np.ndarray,
    b: np.ndarray,
    *,
    session_a: str,
    session_b: str,
    representation: str,
    link: str,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    med_a, iqr_a, sig_a = robust_stats(a)
    med_b, iqr_b, sig_b = robust_stats(b)

    raw_shift = med_b - med_a
    pooled = pooled_robust_scale(sig_a, sig_b)
    effect = np.abs(raw_shift) / pooled

    # A pure additive link-level gain shift will move all subcarriers together.
    # Remove the median shift across subcarriers and inspect what remains.
    global_shift = float(np.median(raw_shift))
    residual_shift = raw_shift - global_shift

    carriers = np.arange(a.shape[1])

    rows = []
    for c in carriers:
        rows.append(
            {
                "representation": representation,
                "link": link,
                "subcarrier": int(c),
                f"{session_a}_median": float(med_a[c]),
                f"{session_b}_median": float(med_b[c]),
                "median_shift_B_minus_A": float(raw_shift[c]),
                f"{session_a}_iqr": float(iqr_a[c]),
                f"{session_b}_iqr": float(iqr_b[c]),
                "pooled_robust_scale": float(pooled[c]),
                "robust_effect_size_abs": float(effect[c]),
                "residual_shift_after_global_gain": float(residual_shift[c]),
            }
        )

    safe = f"{representation}_{link}".lower()

    plot_profiles(
        carriers,
        med_a,
        med_b,
        label_a=session_a,
        label_b=session_b,
        ylabel="median log amplitude" if representation == "absolute" else "median centered log amplitude",
        title=f"{link} {representation}: {session_a} vs {session_b}",
        path=out_dir / f"{safe}_median_profiles.png",
    )

    plot_shift(
        carriers,
        raw_shift,
        ylabel=f"{session_b} - {session_a}",
        title=f"{link} {representation}: per-subcarrier session shift",
        path=out_dir / f"{safe}_session_shift.png",
    )

    plot_effect(
        carriers,
        effect,
        title=f"{link} {representation}: robust session-shift effect size",
        path=out_dir / f"{safe}_robust_effect_size.png",
    )

    summary = {
        "representation": representation,
        "link": link,
        "global_median_shift": global_shift,
        "absolute_raw_shift": summarize_vector(np.abs(raw_shift)),
        "absolute_residual_frequency_shift": summarize_vector(np.abs(residual_shift)),
        "robust_effect_size": summarize_vector(effect),
        "fraction_subcarriers_effect_gt_0_5": float(np.mean(effect > 0.5)),
        "fraction_subcarriers_effect_gt_1": float(np.mean(effect > 1.0)),
        "fraction_subcarriers_effect_gt_2": float(np.mean(effect > 2.0)),
    }

    return rows, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session-a", default="S1")
    p.add_argument("--session-b", default="S2")
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    args = p.parse_args()

    print("\n" + "=" * 78)
    print("TriCSI RAW CSI SESSION-SHIFT DIAGNOSTIC")
    print("=" * 78)
    print(f"Session A: {args.session_a}")
    print(f"Session B: {args.session_b}")
    print("=" * 78)

    paths_a = trial_paths(args.raw_dir, args.session_a)
    paths_b = trial_paths(args.raw_dir, args.session_b)

    print(f"\nLoading {args.session_a} ({len(paths_a)} trials)...")
    a = collect_session(paths_a)

    print(f"\nLoading {args.session_b} ({len(paths_b)} trials)...")
    b = collect_session(paths_b)

    out_dir = RESULTS_DIR / (
        f"session_shift_{args.session_a.lower()}_{args.session_b.lower()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nPACKET COUNTS")
    for key in ("L_abs", "R_abs"):
        print(
            f"  {key[0]} link: "
            f"{args.session_a}={len(a[key])}, "
            f"{args.session_b}={len(b[key])}"
        )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for representation, suffix in (
        ("absolute", "abs"),
        ("centered", "center"),
    ):
        for link in ("L", "R"):
            rows, summary = analyze_representation(
                a[f"{link}_{suffix}"],
                b[f"{link}_{suffix}"],
                session_a=args.session_a,
                session_b=args.session_b,
                representation=representation,
                link=link,
                out_dir=out_dir,
            )
            all_rows.extend(rows)
            summaries.append(summary)

    write_csv(
        out_dir / "per_subcarrier_session_shift.csv",
        all_rows,
    )

    # ------------------------------------------------------------------
    # Link-relative fingerprint:
    # median absolute log-amplitude profile of LEFT minus RIGHT.
    # This does not require packet-level synchronization; it compares the
    # session-level median profile of each link.
    # ------------------------------------------------------------------
    med_la = np.median(a["L_abs"], axis=0)
    med_ra = np.median(a["R_abs"], axis=0)
    med_lb = np.median(b["L_abs"], axis=0)
    med_rb = np.median(b["R_abs"], axis=0)

    diff_a = med_la - med_ra
    diff_b = med_lb - med_rb
    diff_shift = diff_b - diff_a

    carriers = np.arange(64)

    plot_profiles(
        carriers,
        diff_a,
        diff_b,
        label_a=args.session_a,
        label_b=args.session_b,
        ylabel="LEFT median - RIGHT median",
        title="Link-relative absolute CSI fingerprint",
        path=out_dir / "left_minus_right_fingerprint.png",
    )

    plot_shift(
        carriers,
        diff_shift,
        ylabel=f"{args.session_b} - {args.session_a}",
        title="Session shift of LEFT-minus-RIGHT fingerprint",
        path=out_dir / "left_minus_right_fingerprint_shift.png",
    )

    # ------------------------------------------------------------------
    # RSSI summary
    # ------------------------------------------------------------------
    rssi_summary = {}
    for link in ("L", "R"):
        xa = a[f"{link}_rssi"]
        xb = b[f"{link}_rssi"]
        rssi_summary[link] = {
            f"{args.session_a}_mean": float(np.mean(xa)),
            f"{args.session_a}_median": float(np.median(xa)),
            f"{args.session_a}_std": float(np.std(xa)),
            f"{args.session_b}_mean": float(np.mean(xb)),
            f"{args.session_b}_median": float(np.median(xb)),
            f"{args.session_b}_std": float(np.std(xb)),
            "median_shift_B_minus_A": float(np.median(xb) - np.median(xa)),
        }

    # ------------------------------------------------------------------
    # Overall diagnostic interpretation flags.
    # ------------------------------------------------------------------
    by_key = {
        (s["representation"], s["link"]): s
        for s in summaries
    }

    absolute_effect_mean = float(
        np.mean(
            [
                by_key[("absolute", "L")]["robust_effect_size"]["mean"],
                by_key[("absolute", "R")]["robust_effect_size"]["mean"],
            ]
        )
    )

    centered_effect_mean = float(
        np.mean(
            [
                by_key[("centered", "L")]["robust_effect_size"]["mean"],
                by_key[("centered", "R")]["robust_effect_size"]["mean"],
            ]
        )
    )

    absolute_residual_mean = float(
        np.mean(
            [
                by_key[("absolute", "L")]["absolute_residual_frequency_shift"]["mean"],
                by_key[("absolute", "R")]["absolute_residual_frequency_shift"]["mean"],
            ]
        )
    )

    overall = {
        "session_a": args.session_a,
        "session_b": args.session_b,
        "packets": {
            args.session_a: {
                "left": int(len(a["L_abs"])),
                "right": int(len(a["R_abs"])),
            },
            args.session_b: {
                "left": int(len(b["L_abs"])),
                "right": int(len(b["R_abs"])),
            },
        },
        "representations": summaries,
        "rssi": rssi_summary,
        "left_minus_right_fingerprint_shift": summarize_vector(np.abs(diff_shift)),
        "mean_absolute_representation_effect_size": absolute_effect_mean,
        "mean_centered_representation_effect_size": centered_effect_mean,
        "mean_absolute_residual_frequency_shift_after_global_gain": absolute_residual_mean,
        "interpretation_note": (
            "A large absolute effect size with much smaller centered effect size "
            "suggests mostly gain/common-mode shift. A large centered effect size "
            "or large residual frequency-dependent shift means the subcarrier "
            "shape itself changed between sessions. A large LEFT-minus-RIGHT "
            "fingerprint shift means even link-relative structure is unstable."
        ),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # Compact console report.
    print("\n" + "=" * 78)
    print("SESSION-SHIFT SUMMARY")
    print("=" * 78)

    for s in summaries:
        print(
            f"{s['representation']:8s} {s['link']} | "
            f"global shift={s['global_median_shift']:+.3f} | "
            f"effect mean={s['robust_effect_size']['mean']:.3f} "
            f"median={s['robust_effect_size']['median']:.3f} | "
            f">1σ carriers={100*s['fraction_subcarriers_effect_gt_1']:.1f}% | "
            f"residual freq shift mean="
            f"{s['absolute_residual_frequency_shift']['mean']:.3f}"
        )

    print("\nRSSI MEDIAN SHIFT")
    for link in ("L", "R"):
        print(
            f"  {link}: "
            f"{rssi_summary[link]['median_shift_B_minus_A']:+.2f} dB"
        )

    print(
        "\nLEFT-minus-RIGHT fingerprint "
        f"| abs shift mean="
        f"{overall['left_minus_right_fingerprint_shift']['mean']:.3f} "
        f"median="
        f"{overall['left_minus_right_fingerprint_shift']['median']:.3f}"
    )

    print(f"\nSaved results: {out_dir}")
    print("Key files:")
    print("  summary.json")
    print("  per_subcarrier_session_shift.csv")
    print("  absolute_L_median_profiles.png")
    print("  absolute_R_median_profiles.png")
    print("  centered_L_median_profiles.png")
    print("  centered_R_median_profiles.png")
    print("  left_minus_right_fingerprint_shift.png")


if __name__ == "__main__":
    main()
