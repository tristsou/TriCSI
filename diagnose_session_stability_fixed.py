from __future__ import annotations

"""
Deep TriCSI session-stability diagnostic.

Goal
----
Explain WHY S3 is less internally stable than S1/S2 by comparing raw CSI
and RSSI at the TRIAL level, without training another classifier.

It measures, for S1/S2/S3:
- packets and packet rate per link
- RSSI median / spread per link
- median absolute log-amplitude per link
- within-trial CSI variability
- 64-subcarrier trial fingerprints
- packet-centered ("shape") fingerprints
- LEFT-vs-RIGHT fingerprint relationship
- same-zone trial-to-trial distances
- North trial 1 vs North trial 5 distances
- drift versus collection order

Outputs
-------
results/session_stability/
    trial_metrics.csv
    same_zone_pair_distances.csv
    n1_n5_distances.csv
    session_summary.csv
    drift_correlations.csv
    fingerprint_*.png
    rssi_over_order_*.png
    amplitude_over_order_*.png
    same_zone_distance_boxplot.png
    n1_n5_distance_boxplot.png
    summary.json

Run
---
    python diagnose_session_stability.py

Notes
-----
- Reads the collected trial CSVs directly.
- Assumes iq_000 ... iq_127 are the first 128 raw LLTF I/Q values.
- Pair order is imag, real, matching the current TriCSI collection pipeline.
- File modification time is used only as a fallback estimate of collection
  order if an explicit timestamp/order cannot be inferred.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LEFT_MAC = "02:00:00:00:00:01"
DEFAULT_RIGHT_MAC = "02:00:00:00:00:03"
SESSIONS = ("S1", "S2", "S3")
ZONES = ("EMPTY", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_mac(x: str) -> str:
    return str(x).strip().upper().replace("-", ":")


def find_col(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def infer_zone(path: Path, df: pd.DataFrame) -> str:
    c = find_col(df.columns, ["zone", "class", "label"])
    if c is not None and len(df):
        v = str(df.iloc[0][c]).strip().upper()
        if v in ZONES:
            return v

    text = path.stem.upper()
    m = re.search(r"(?<![A-Z0-9])(EMPTY|Z[1-9])(?![A-Z0-9])", text)
    if m:
        return m.group(1)

    raise ValueError(f"Could not infer zone from {path.name}")


def infer_trial_number(path: Path, df: pd.DataFrame) -> int:
    c = find_col(df.columns, ["trial", "trial_number", "trial_num", "trial_id"])
    if c is not None and len(df):
        try:
            return int(float(df.iloc[0][c]))
        except Exception:
            pass

    text = path.stem
    patterns = [
        r"trial[_-]?0*([1-9]\d*)",
        r"(?:^|[_-])t0*([1-9]\d*)(?:[_-]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

    # Last resort: look for a standalone 1..5 token.
    toks = re.findall(r"(?:^|[_-])([1-5])(?:[_-]|$)", text)
    if toks:
        return int(toks[-1])

    raise ValueError(f"Could not infer trial number from {path.name}")


def infer_orientation(path: Path, df: pd.DataFrame, trial: int, zone: str) -> str:
    c = find_col(df.columns, ["orientation", "facing", "direction"])
    if c is not None and len(df):
        v = str(df.iloc[0][c]).strip().upper()
        if v:
            return v

    if zone == "EMPTY":
        return "NA"

    return {1: "N", 2: "E", 3: "S", 4: "W", 5: "N"}.get(trial, "?")


def numeric_series(df, col):
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def robust_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    q25, q75 = np.percentile(x, [25, 75])
    return float(q75 - q25)


def pearson_order(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


def euclid(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def standardized_distance_vectors(vectors):
    """
    Robustly standardize dimensions using all supplied trial vectors.
    Returns standardized array and center/scale.
    """
    X = np.asarray(vectors, dtype=float)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    scale = 1.4826 * mad
    std = np.nanstd(X, axis=0)
    scale = np.where(scale > 1e-8, scale, std)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (X - med) / scale, med, scale


# ---------------------------------------------------------------------------
# Trial parsing
# ---------------------------------------------------------------------------

def read_trial(
    path: Path,
    session: str,
    left_mac: str,
    right_mac: str,
):
    df = pd.read_csv(path)

    zone = infer_zone(path, df)
    trial = infer_trial_number(path, df)
    orientation = infer_orientation(path, df, trial, zone)

    mac_col = find_col(
        df.columns,
        ["mac", "source_mac", "src_mac", "transmitter_mac", "tx_mac"],
    )
    rssi_col = find_col(df.columns, ["rssi", "RSSI"])
    ts_col = find_col(
        df.columns,
        [
            "real_timestamp",
            "timestamp",
            "local_timestamp",
            "time",
            "time_s",
            "timestamp_s",
        ],
    )

    iq_cols = [
        c for c in df.columns
        if re.fullmatch(r"iq_\d{3}", str(c), flags=re.IGNORECASE)
    ]
    iq_cols = sorted(iq_cols, key=lambda c: int(re.search(r"(\d+)$", str(c)).group(1)))

    if len(iq_cols) < 128:
        raise ValueError(
            f"{path.name}: expected >=128 iq_### columns, found {len(iq_cols)}"
        )

    iq_cols = iq_cols[:128]

    if mac_col is None:
        raise ValueError(f"{path.name}: could not find source MAC column")

    macs = df[mac_col].astype(str).map(norm_mac)

    out = {
        "session": session,
        "zone": zone,
        "trial": int(trial),
        "orientation": orientation,
        "path": str(path),
        "mtime": float(path.stat().st_mtime),
        "links": {},
    }

    for link_name, target_mac in (
        ("LEFT", norm_mac(left_mac)),
        ("RIGHT", norm_mac(right_mac)),
    ):
        sub = df.loc[macs == target_mac].copy()

        if len(sub) == 0:
            raise ValueError(f"{path.name}: no rows for {link_name} {target_mac}")

        raw = sub[iq_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        # Drop rows with unusable I/Q.
        good = np.isfinite(raw).all(axis=1)
        raw = raw[good]
        sub = sub.iloc[np.where(good)[0]].copy()

        if len(raw) == 0:
            raise ValueError(f"{path.name}: no valid I/Q rows for {link_name}")

        imag = raw[:, 0::2]
        real = raw[:, 1::2]
        amp = np.sqrt(imag * imag + real * real)
        logamp = np.log1p(amp)

        # Packet-centered shape removes packet-wide common gain.
        centered = logamp - np.median(logamp, axis=1, keepdims=True)

        rssi = numeric_series(sub, rssi_col)
        if rssi is None:
            rssi = np.full(len(sub), np.nan)

        # Packet rate from timestamp if plausible.
        rate_hz = float("nan")
        duration_s = float("nan")

        ts = numeric_series(sub, ts_col)
        if ts is not None:
            ts = ts[np.isfinite(ts)]
            if len(ts) >= 2:
                ts = np.sort(ts)
                span = float(ts[-1] - ts[0])

                # Infer common units heuristically.
                if span > 1e8:       # likely microseconds over 8 s
                    span_s = span / 1e6
                elif span > 1e5:     # likely milliseconds
                    span_s = span / 1e3
                elif span > 0:
                    span_s = span
                else:
                    span_s = float("nan")

                if np.isfinite(span_s) and 0.5 <= span_s <= 120:
                    duration_s = span_s
                    rate_hz = (len(ts) - 1) / span_s

        # Fallback to expected collection duration from packet count / ~33.3 Hz.
        if not np.isfinite(rate_hz):
            rate_hz = len(raw) / 8.0
            duration_s = 8.0

        out["links"][link_name] = {
            "n_packets": int(len(raw)),
            "duration_s": float(duration_s),
            "rate_hz": float(rate_hz),
            "rssi_median": float(np.nanmedian(rssi)),
            "rssi_mean": float(np.nanmean(rssi)),
            "rssi_std": float(np.nanstd(rssi)),
            "rssi_iqr": robust_iqr(rssi),
            "logamp_global_median": float(np.nanmedian(logamp)),
            "logamp_global_std": float(np.nanstd(logamp)),
            "within_trial_temporal_std_mean": float(
                np.mean(np.std(logamp, axis=0))
            ),
            "fingerprint_abs": np.median(logamp, axis=0),
            "fingerprint_shape": np.median(centered, axis=0),
        }

    # Cross-link trial fingerprint.
    out["cross"] = {
        "abs_left_minus_right": (
            out["links"]["LEFT"]["fingerprint_abs"]
            - out["links"]["RIGHT"]["fingerprint_abs"]
        ),
        "shape_left_minus_right": (
            out["links"]["LEFT"]["fingerprint_shape"]
            - out["links"]["RIGHT"]["fingerprint_shape"]
        ),
    }

    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_all(raw_dir: Path, left_mac: str, right_mac: str):
    trials = []

    for session in SESSIONS:
        folder = raw_dir / session
        all_paths = sorted(folder.glob("*.csv"))

        if not all_paths:
            raise SystemExit(f"No CSVs found in {folder}")

        # The session folders also contain metadata CSVs such as
        # schedule_S1.csv. Those are not CSI trial files.
        paths = []
        skipped = []

        for path in all_paths:
            try:
                header = pd.read_csv(path, nrows=0)
                iq_cols = [
                    c for c in header.columns
                    if re.fullmatch(r"iq_\\d{3}", str(c), flags=re.IGNORECASE)
                ]

                if len(iq_cols) >= 128:
                    paths.append(path)
                else:
                    skipped.append(path.name)
            except Exception:
                skipped.append(path.name)

        print(
            f"Reading {session}: {len(paths)} CSI trial files "
            f"({len(skipped)} non-trial CSVs skipped)"
        )

        if skipped:
            print("  skipped:", ", ".join(skipped))

        if not paths:
            raise SystemExit(f"No CSI trial CSVs found in {folder}")

        for path in paths:
            try:
                trials.append(
                    read_trial(path, session, left_mac, right_mac)
                )
            except Exception as e:
                raise RuntimeError(f"Failed on {path}: {e}") from e

    # Assign per-session collection order using mtime.
    for session in SESSIONS:
        ss = [t for t in trials if t["session"] == session]
        ss_sorted = sorted(ss, key=lambda t: (t["mtime"], t["path"]))
        for order, t in enumerate(ss_sorted, 1):
            t["collection_order"] = order

    return trials


def trial_metric_rows(trials):
    rows = []

    for t in trials:
        base = {
            "session": t["session"],
            "zone": t["zone"],
            "trial": t["trial"],
            "orientation": t["orientation"],
            "collection_order": t["collection_order"],
            "filename": Path(t["path"]).name,
        }

        for link in ("LEFT", "RIGHT"):
            x = t["links"][link]
            rows.append(
                {
                    **base,
                    "link": link,
                    "n_packets": x["n_packets"],
                    "duration_s": x["duration_s"],
                    "rate_hz": x["rate_hz"],
                    "rssi_median": x["rssi_median"],
                    "rssi_mean": x["rssi_mean"],
                    "rssi_std": x["rssi_std"],
                    "rssi_iqr": x["rssi_iqr"],
                    "logamp_global_median": x["logamp_global_median"],
                    "logamp_global_std": x["logamp_global_std"],
                    "within_trial_temporal_std_mean": x[
                        "within_trial_temporal_std_mean"
                    ],
                }
            )

    return rows


def distance_analysis(trials):
    """
    Distances are robust-standardized separately for each representation/link
    using all 150 trials, so S1/S2/S3 remain on the same comparison scale.
    """
    reps = []

    for rep_name, getter in (
        ("LEFT_abs", lambda t: t["links"]["LEFT"]["fingerprint_abs"]),
        ("RIGHT_abs", lambda t: t["links"]["RIGHT"]["fingerprint_abs"]),
        ("LEFT_shape", lambda t: t["links"]["LEFT"]["fingerprint_shape"]),
        ("RIGHT_shape", lambda t: t["links"]["RIGHT"]["fingerprint_shape"]),
        ("LR_abs_diff", lambda t: t["cross"]["abs_left_minus_right"]),
        ("LR_shape_diff", lambda t: t["cross"]["shape_left_minus_right"]),
    ):
        X = [getter(t) for t in trials]
        Z, _, _ = standardized_distance_vectors(X)

        for i, t in enumerate(trials):
            reps.append(
                {
                    "rep": rep_name,
                    "trial_index": i,
                    "vector": Z[i],
                }
            )

    lookup = {(r["rep"], r["trial_index"]): r["vector"] for r in reps}

    pair_rows = []
    n_rows = []

    for rep_name in sorted(set(r["rep"] for r in reps)):
        for session in SESSIONS:
            ss_idx = [
                i for i, t in enumerate(trials)
                if t["session"] == session
            ]

            # All same-zone pairs within session.
            for zone in ZONES:
                idx = [
                    i for i in ss_idx
                    if trials[i]["zone"] == zone
                ]

                for a_pos in range(len(idx)):
                    for b_pos in range(a_pos + 1, len(idx)):
                        i = idx[a_pos]
                        j = idx[b_pos]
                        d = euclid(
                            lookup[(rep_name, i)],
                            lookup[(rep_name, j)],
                        )

                        pair_rows.append(
                            {
                                "representation": rep_name,
                                "session": session,
                                "zone": zone,
                                "trial_a": trials[i]["trial"],
                                "trial_b": trials[j]["trial"],
                                "orientation_a": trials[i]["orientation"],
                                "orientation_b": trials[j]["orientation"],
                                "same_orientation": int(
                                    trials[i]["orientation"]
                                    == trials[j]["orientation"]
                                ),
                                "distance": d,
                            }
                        )

                # Specific N1 vs N5.
                n1 = [
                    i for i in idx
                    if trials[i]["trial"] == 1
                ]
                n5 = [
                    i for i in idx
                    if trials[i]["trial"] == 5
                ]

                if n1 and n5:
                    i, j = n1[0], n5[0]
                    d = euclid(
                        lookup[(rep_name, i)],
                        lookup[(rep_name, j)],
                    )

                    n_rows.append(
                        {
                            "representation": rep_name,
                            "session": session,
                            "zone": zone,
                            "distance_N1_N5": d,
                        }
                    )

    return pair_rows, n_rows


def session_summary(metric_rows, pair_rows, n_rows):
    rows = []

    for session in SESSIONS:
        for link in ("LEFT", "RIGHT"):
            m = [
                r for r in metric_rows
                if r["session"] == session and r["link"] == link
            ]

            same_abs = [
                r["distance"]
                for r in pair_rows
                if r["session"] == session
                and r["representation"] == f"{link}_abs"
            ]
            same_shape = [
                r["distance"]
                for r in pair_rows
                if r["session"] == session
                and r["representation"] == f"{link}_shape"
            ]
            n_abs = [
                r["distance_N1_N5"]
                for r in n_rows
                if r["session"] == session
                and r["representation"] == f"{link}_abs"
            ]
            n_shape = [
                r["distance_N1_N5"]
                for r in n_rows
                if r["session"] == session
                and r["representation"] == f"{link}_shape"
            ]

            rows.append(
                {
                    "session": session,
                    "link": link,
                    "median_rate_hz": float(np.median([x["rate_hz"] for x in m])),
                    "median_rssi": float(np.median([x["rssi_median"] for x in m])),
                    "rssi_trial_range": float(
                        np.max([x["rssi_median"] for x in m])
                        - np.min([x["rssi_median"] for x in m])
                    ),
                    "median_logamp": float(
                        np.median([x["logamp_global_median"] for x in m])
                    ),
                    "logamp_trial_range": float(
                        np.max([x["logamp_global_median"] for x in m])
                        - np.min([x["logamp_global_median"] for x in m])
                    ),
                    "median_within_trial_temporal_std": float(
                        np.median(
                            [x["within_trial_temporal_std_mean"] for x in m]
                        )
                    ),
                    "median_same_zone_abs_distance": float(np.median(same_abs)),
                    "median_same_zone_shape_distance": float(np.median(same_shape)),
                    "median_N1_N5_abs_distance": float(np.median(n_abs)),
                    "median_N1_N5_shape_distance": float(np.median(n_shape)),
                }
            )

    return rows


def drift_rows(metric_rows):
    rows = []

    for session in SESSIONS:
        for link in ("LEFT", "RIGHT"):
            m = sorted(
                [
                    r for r in metric_rows
                    if r["session"] == session and r["link"] == link
                ],
                key=lambda r: r["collection_order"],
            )

            rows.append(
                {
                    "session": session,
                    "link": link,
                    "rssi_vs_order_r": pearson_order(
                        [x["rssi_median"] for x in m]
                    ),
                    "logamp_vs_order_r": pearson_order(
                        [x["logamp_global_median"] for x in m]
                    ),
                    "within_trial_std_vs_order_r": pearson_order(
                        [x["within_trial_temporal_std_mean"] for x in m]
                    ),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_over_order(metric_rows, out_dir: Path, field: str, ylabel: str, prefix: str):
    for link in ("LEFT", "RIGHT"):
        fig, ax = plt.subplots(figsize=(9, 5))

        for session in SESSIONS:
            m = sorted(
                [
                    r for r in metric_rows
                    if r["session"] == session and r["link"] == link
                ],
                key=lambda r: r["collection_order"],
            )

            ax.plot(
                [x["collection_order"] for x in m],
                [x[field] for x in m],
                marker="o",
                linewidth=1,
                markersize=3,
                label=session,
            )

        ax.set_xlabel("Collection order within session")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} over collection order — {link}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{link.lower()}.png", dpi=180)
        plt.close(fig)


def plot_distance_box(pair_rows, out_dir: Path):
    reps = ("LEFT_abs", "RIGHT_abs", "LEFT_shape", "RIGHT_shape")

    for rep in reps:
        data = []
        labels = []

        for session in SESSIONS:
            vals = [
                r["distance"]
                for r in pair_rows
                if r["representation"] == rep and r["session"] == session
            ]
            data.append(vals)
            labels.append(session)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(data, tick_labels=labels)
        ax.set_ylabel("Robust-standardized Euclidean distance")
        ax.set_title(f"Same-zone trial-to-trial distance — {rep}")
        fig.tight_layout()
        fig.savefig(out_dir / f"same_zone_distance_{rep}.png", dpi=180)
        plt.close(fig)


def plot_n_box(n_rows, out_dir: Path):
    reps = ("LEFT_abs", "RIGHT_abs", "LEFT_shape", "RIGHT_shape")

    for rep in reps:
        data = []
        labels = []

        for session in SESSIONS:
            vals = [
                r["distance_N1_N5"]
                for r in n_rows
                if r["representation"] == rep
                and r["session"] == session
                and r["zone"] != "EMPTY"
            ]
            data.append(vals)
            labels.append(session)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(data, tick_labels=labels)
        ax.set_ylabel("N1 ↔ N5 fingerprint distance")
        ax.set_title(f"North-repeatability distance — {rep}")
        fig.tight_layout()
        fig.savefig(out_dir / f"n1_n5_distance_{rep}.png", dpi=180)
        plt.close(fig)


def plot_fingerprint_heatmaps(trials, out_dir: Path):
    for session in SESSIONS:
        ss = sorted(
            [t for t in trials if t["session"] == session],
            key=lambda t: t["collection_order"],
        )

        for link in ("LEFT", "RIGHT"):
            mat = np.vstack(
                [t["links"][link]["fingerprint_abs"] for t in ss]
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(mat, aspect="auto")
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("Trial collection order")
            ax.set_title(f"{session} {link} median log-amplitude fingerprint")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(
                out_dir / f"fingerprint_{session.lower()}_{link.lower()}.png",
                dpi=180,
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows):
    if not rows:
        return

    keys = []
    seen = set()

    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data") / "raw",
    )
    p.add_argument("--left-mac", default=DEFAULT_LEFT_MAC)
    p.add_argument("--right-mac", default=DEFAULT_RIGHT_MAC)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "session_stability",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("TriCSI RAW SESSION-STABILITY DIAGNOSTIC")
    print("=" * 78)
    print(f"Raw data : {args.raw_dir}")
    print(f"LEFT MAC : {norm_mac(args.left_mac)}")
    print(f"RIGHT MAC: {norm_mac(args.right_mac)}")
    print("=" * 78)

    trials = load_all(
        args.raw_dir,
        args.left_mac,
        args.right_mac,
    )

    print(f"\nLoaded {len(trials)} trials.")

    metric_rows = trial_metric_rows(trials)
    pair_rows, n_rows = distance_analysis(trials)
    summary_rows = session_summary(metric_rows, pair_rows, n_rows)
    drift = drift_rows(metric_rows)

    write_csv(args.output_dir / "trial_metrics.csv", metric_rows)
    write_csv(args.output_dir / "same_zone_pair_distances.csv", pair_rows)
    write_csv(args.output_dir / "n1_n5_distances.csv", n_rows)
    write_csv(args.output_dir / "session_summary.csv", summary_rows)
    write_csv(args.output_dir / "drift_correlations.csv", drift)

    plot_over_order(
        metric_rows,
        args.output_dir,
        "rssi_median",
        "Median RSSI (dB)",
        "rssi_over_order",
    )
    plot_over_order(
        metric_rows,
        args.output_dir,
        "logamp_global_median",
        "Median log CSI amplitude",
        "amplitude_over_order",
    )
    plot_distance_box(pair_rows, args.output_dir)
    plot_n_box(n_rows, args.output_dir)
    plot_fingerprint_heatmaps(trials, args.output_dir)

    summary_json = {
        "session_summary": summary_rows,
        "drift_correlations": drift,
        "note": (
            "Distances are robust-standardized across all 150 trials for each "
            "representation. Larger same-zone or N1-N5 distance means poorer "
            "trial-to-trial CSI stability."
        ),
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("\n" + "=" * 78)
    print("SESSION-STABILITY SUMMARY")
    print("=" * 78)

    for session in SESSIONS:
        print(f"\n{session}")
        for link in ("LEFT", "RIGHT"):
            r = next(
                x for x in summary_rows
                if x["session"] == session and x["link"] == link
            )

            print(
                f"  {link:5s} | "
                f"rate={r['median_rate_hz']:.1f} Hz | "
                f"RSSI range={r['rssi_trial_range']:.1f} dB | "
                f"amp range={r['logamp_trial_range']:.3f} | "
                f"same-zone abs dist={r['median_same_zone_abs_distance']:.2f} | "
                f"N1-N5 abs dist={r['median_N1_N5_abs_distance']:.2f}"
            )

    print("\nDrift versus collection order (Pearson r):")
    for r in drift:
        print(
            f"  {r['session']} {r['link']:5s} | "
            f"RSSI r={r['rssi_vs_order_r']:+.2f} | "
            f"amplitude r={r['logamp_vs_order_r']:+.2f} | "
            f"within-trial variability r={r['within_trial_std_vs_order_r']:+.2f}"
        )

    print(f"\nSaved all diagnostics to: {args.output_dir}")
    print(
        "\nLook especially at whether S3 has much larger same-zone/N1-N5 "
        "distances than S1/S2, and whether one link shows stronger drift."
    )


if __name__ == "__main__":
    main()
