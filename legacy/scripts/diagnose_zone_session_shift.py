from __future__ import annotations

"""
TriCSI zone-conditioned S1-vs-S2 shift diagnostic.

Why this exists
---------------
The first session-shift diagnostic pooled ALL zones together and found only
small global per-subcarrier shifts. But the CNN still collapsed to chance
cross-session. Pooling across classes can hide a more important failure mode:

    S1 Z1 fingerprint may move toward S2 Z4
    S1 Z2 fingerprint may move toward S2 Z6
    ...

This script therefore compares CSI fingerprints CONDITIONED ON ZONE and,
where possible, matched by orientation.

It answers:
1) Is the same physical zone still nearest to itself across sessions?
2) Are some zones systematically "relabelled" by session shift?
3) Is the shift mostly in absolute amplitude or packet-centered shape?
4) Does orientation explain a large part of the mismatch?

Run:
    python diagnose_zone_session_shift.py --session-a S1 --session-b S2
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from csi_io import load_trial_csv
from project_config import CLASS_ORDER, RAW_DATA_DIR, RESULTS_DIR


EPS = 1e-8


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


def collect_session(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """
    Returns one entry per trial group keyed by a stable string.
    """
    trials: dict[str, dict[str, Any]] = {}

    for i, path in enumerate(paths, 1):
        print(f"[{i:03d}/{len(paths):03d}] {path.name}")
        trial = load_trial_csv(path)
        meta = trial["metadata"]

        zone = str(meta["zone"])
        orientation = str(meta["orientation"])
        trial_num = int(str(meta["trial"]).strip())
        session = str(meta["session"])

        left_abs = iq_to_logamp(trial["left"]["iq"])
        right_abs = iq_to_logamp(trial["right"]["iq"])

        left_center = (
            left_abs
            - np.median(left_abs, axis=1, keepdims=True)
        ).astype(np.float32)

        right_center = (
            right_abs
            - np.median(right_abs, axis=1, keepdims=True)
        ).astype(np.float32)

        key = f"{session}|{zone}|{trial_num}|{orientation}"

        trials[key] = {
            "session": session,
            "zone": zone,
            "trial": trial_num,
            "orientation": orientation,
            "source_file": str(path),
            "L_abs_profile": np.median(left_abs, axis=0).astype(np.float64),
            "R_abs_profile": np.median(right_abs, axis=0).astype(np.float64),
            "L_center_profile": np.median(left_center, axis=0).astype(np.float64),
            "R_center_profile": np.median(right_center, axis=0).astype(np.float64),
            "L_rssi_median": float(np.median(trial["left"]["rssi"])),
            "R_rssi_median": float(np.median(trial["right"]["rssi"])),
        }

    return trials


def zone_profiles(
    trials: dict[str, dict[str, Any]],
    representation: str,
) -> dict[str, np.ndarray]:
    """
    Concatenate LEFT and RIGHT 64-carrier median profiles => 128-D fingerprint.
    Aggregate trial fingerprints within each zone using median.
    """
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)

    for row in trials.values():
        if representation == "absolute":
            fp = np.concatenate(
                [row["L_abs_profile"], row["R_abs_profile"]]
            )
        elif representation == "centered":
            fp = np.concatenate(
                [row["L_center_profile"], row["R_center_profile"]]
            )
        else:
            raise ValueError(representation)

        grouped[row["zone"]].append(fp)

    return {
        zone: np.median(np.stack(v, axis=0), axis=0)
        for zone, v in grouped.items()
    }


def normalize_for_distance(
    a_profiles: dict[str, np.ndarray],
    b_profiles: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Standardize dimensions using pooled per-dimension robust-ish scale from
    all zone fingerprints across both sessions. This avoids a few large-value
    carriers dominating Euclidean distance.
    """
    labels = sorted(set(a_profiles) & set(b_profiles))
    all_rows = np.stack(
        [a_profiles[z] for z in labels] + [b_profiles[z] for z in labels],
        axis=0,
    )
    mean = np.mean(all_rows, axis=0)
    std = np.std(all_rows, axis=0)
    std = np.maximum(std, EPS)

    a = {z: (a_profiles[z] - mean) / std for z in labels}
    b = {z: (b_profiles[z] - mean) / std for z in labels}
    return a, b


def euclidean_matrix(
    a_profiles: dict[str, np.ndarray],
    b_profiles: dict[str, np.ndarray],
    labels: list[str],
) -> np.ndarray:
    out = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for i, za in enumerate(labels):
        for j, zb in enumerate(labels):
            out[i, j] = np.linalg.norm(a_profiles[za] - b_profiles[zb])
    return out


def cosine_distance_matrix(
    a_profiles: dict[str, np.ndarray],
    b_profiles: dict[str, np.ndarray],
    labels: list[str],
) -> np.ndarray:
    out = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for i, za in enumerate(labels):
        x = a_profiles[za]
        xn = np.linalg.norm(x)
        for j, zb in enumerate(labels):
            y = b_profiles[zb]
            yn = np.linalg.norm(y)
            denom = max(xn * yn, EPS)
            out[i, j] = 1.0 - float(np.dot(x, y) / denom)
    return out


def save_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    *,
    title: str,
    path: Path,
    fmt: str = ".2f",
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(matrix)
    ax.set_title(title)
    ax.set_xlabel("Session B zone")
    ax.set_ylabel("Session A zone")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, format(matrix[i, j], fmt),
                ha="center", va="center", fontsize=8
            )

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


def matched_orientation_trial_distances(
    trials_a: dict[str, dict[str, Any]],
    trials_b: dict[str, dict[str, Any]],
    representation: str,
) -> list[dict[str, Any]]:
    """
    Match by zone + orientation + trial number when available.
    With the current design, trial numbers/orientations are aligned across S1/S2.
    """
    index_b: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in trials_b.values():
        index_b[(row["zone"], row["trial"], row["orientation"])] = row

    rows = []

    for a in trials_a.values():
        key = (a["zone"], a["trial"], a["orientation"])
        if key not in index_b:
            continue

        b = index_b[key]

        if representation == "absolute":
            fa = np.concatenate([a["L_abs_profile"], a["R_abs_profile"]])
            fb = np.concatenate([b["L_abs_profile"], b["R_abs_profile"]])
        else:
            fa = np.concatenate([a["L_center_profile"], a["R_center_profile"]])
            fb = np.concatenate([b["L_center_profile"], b["R_center_profile"]])

        rows.append(
            {
                "zone": a["zone"],
                "trial": a["trial"],
                "orientation": a["orientation"],
                "representation": representation,
                "euclidean_raw": float(np.linalg.norm(fa - fb)),
                "L_rssi_shift_db": float(b["L_rssi_median"] - a["L_rssi_median"]),
                "R_rssi_shift_db": float(b["R_rssi_median"] - a["R_rssi_median"]),
            }
        )

    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session-a", default="S1")
    p.add_argument("--session-b", default="S2")
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    args = p.parse_args()

    print("\n" + "=" * 78)
    print("TriCSI ZONE-CONDITIONED SESSION-SHIFT DIAGNOSTIC")
    print("=" * 78)
    print(f"Session A: {args.session_a}")
    print(f"Session B: {args.session_b}")
    print("=" * 78)

    paths_a = trial_paths(args.raw_dir, args.session_a)
    paths_b = trial_paths(args.raw_dir, args.session_b)

    print(f"\nLoading {args.session_a}...")
    trials_a = collect_session(paths_a)

    print(f"\nLoading {args.session_b}...")
    trials_b = collect_session(paths_b)

    labels = [
        z for z in CLASS_ORDER
        if any(r["zone"] == z for r in trials_a.values())
        and any(r["zone"] == z for r in trials_b.values())
    ]

    out_dir = RESULTS_DIR / (
        f"zone_session_shift_{args.session_a.lower()}_{args.session_b.lower()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "session_a": args.session_a,
        "session_b": args.session_b,
        "labels": labels,
        "representations": {},
    }

    ranking_rows: list[dict[str, Any]] = []

    for representation in ("absolute", "centered"):
        print(f"\nAnalyzing {representation} fingerprints...")

        pa_raw = zone_profiles(trials_a, representation)
        pb_raw = zone_profiles(trials_b, representation)

        pa, pb = normalize_for_distance(pa_raw, pb_raw)

        euc = euclidean_matrix(pa, pb, labels)
        cos = cosine_distance_matrix(pa, pb, labels)

        save_heatmap(
            euc,
            labels,
            title=f"{representation}: standardized Euclidean distance, {args.session_a} rows -> {args.session_b} cols",
            path=out_dir / f"{representation}_euclidean_distance_matrix.png",
        )

        save_heatmap(
            cos,
            labels,
            title=f"{representation}: cosine distance, {args.session_a} rows -> {args.session_b} cols",
            path=out_dir / f"{representation}_cosine_distance_matrix.png",
        )

        correct_nn = 0
        self_rank_values = []
        self_to_best_ratio = []

        rows = []

        for i, zone in enumerate(labels):
            order = np.argsort(euc[i])
            nearest_zone = labels[int(order[0])]
            self_idx = labels.index(zone)
            self_distance = float(euc[i, self_idx])
            nearest_distance = float(euc[i, order[0]])
            self_rank = int(np.where(order == self_idx)[0][0]) + 1

            if nearest_zone == zone:
                correct_nn += 1

            ratio = self_distance / max(nearest_distance, EPS)

            rows.append(
                {
                    "representation": representation,
                    "zone_A": zone,
                    "nearest_zone_B": nearest_zone,
                    "nearest_distance": nearest_distance,
                    "same_zone_distance": self_distance,
                    "same_zone_rank": self_rank,
                    "same_zone_over_best_distance": ratio,
                    "second_nearest_zone_B": labels[int(order[1])] if len(order) > 1 else "",
                    "second_nearest_distance": float(euc[i, order[1]]) if len(order) > 1 else None,
                }
            )

            ranking_rows.append(rows[-1])
            self_rank_values.append(self_rank)
            self_to_best_ratio.append(ratio)

        write_csv(
            out_dir / f"{representation}_zone_nearest_matches.csv",
            rows,
        )

        diag = np.diag(euc)
        offdiag = euc[~np.eye(len(labels), dtype=bool)]

        rep_summary = {
            "same_zone_nearest_neighbor_accuracy": float(correct_nn / len(labels)),
            "same_zone_nearest_neighbor_count": int(correct_nn),
            "n_zones": len(labels),
            "median_same_zone_rank": float(np.median(self_rank_values)),
            "mean_same_zone_rank": float(np.mean(self_rank_values)),
            "median_same_zone_distance": float(np.median(diag)),
            "median_between_zone_distance": float(np.median(offdiag)),
            "median_same_zone_over_best_distance": float(np.median(self_to_best_ratio)),
            "zone_matches": rows,
        }

        summary["representations"][representation] = rep_summary

    write_csv(
        out_dir / "all_zone_nearest_matches.csv",
        ranking_rows,
    )

    matched_rows = []
    matched_rows.extend(
        matched_orientation_trial_distances(
            trials_a, trials_b, "absolute"
        )
    )
    matched_rows.extend(
        matched_orientation_trial_distances(
            trials_a, trials_b, "centered"
        )
    )
    write_csv(
        out_dir / "matched_trial_orientation_distances.csv",
        matched_rows,
    )

    # RSSI shift per zone.
    rssi_rows = []
    for zone in labels:
        a_rows = [r for r in trials_a.values() if r["zone"] == zone]
        b_rows = [r for r in trials_b.values() if r["zone"] == zone]

        for link in ("L", "R"):
            a_vals = np.asarray([r[f"{link}_rssi_median"] for r in a_rows], dtype=float)
            b_vals = np.asarray([r[f"{link}_rssi_median"] for r in b_rows], dtype=float)
            rssi_rows.append(
                {
                    "zone": zone,
                    "link": link,
                    f"{args.session_a}_median_rssi": float(np.median(a_vals)),
                    f"{args.session_b}_median_rssi": float(np.median(b_vals)),
                    "median_shift_B_minus_A_db": float(np.median(b_vals) - np.median(a_vals)),
                }
            )

    write_csv(
        out_dir / "per_zone_rssi_shift.csv",
        rssi_rows,
    )

    summary["rssi_per_zone"] = rssi_rows

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("ZONE-CONDITIONED SHIFT SUMMARY")
    print("=" * 78)

    for representation in ("absolute", "centered"):
        s = summary["representations"][representation]
        print(
            f"{representation:8s} | "
            f"same-zone nearest={s['same_zone_nearest_neighbor_count']}/{s['n_zones']} "
            f"({100*s['same_zone_nearest_neighbor_accuracy']:.1f}%) | "
            f"median self-rank={s['median_same_zone_rank']:.1f} | "
            f"median same-zone dist={s['median_same_zone_distance']:.3f} | "
            f"median between-zone dist={s['median_between_zone_distance']:.3f}"
        )

        print("  nearest S2 match for each S1 zone:")
        for row in s["zone_matches"]:
            marker = "✓" if row["zone_A"] == row["nearest_zone_B"] else "✗"
            print(
                f"    {row['zone_A']:5s} -> "
                f"{row['nearest_zone_B']:5s} {marker} | "
                f"self-rank={row['same_zone_rank']} | "
                f"self/best={row['same_zone_over_best_distance']:.2f}"
            )

    print(f"\nSaved results: {out_dir}")
    print("Key files:")
    print("  summary.json")
    print("  absolute_euclidean_distance_matrix.png")
    print("  centered_euclidean_distance_matrix.png")
    print("  absolute_zone_nearest_matches.csv")
    print("  centered_zone_nearest_matches.csv")
    print("  matched_trial_orientation_distances.csv")
    print("  per_zone_rssi_shift.csv")


if __name__ == "__main__":
    main()
