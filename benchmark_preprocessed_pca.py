from __future__ import annotations

"""
Benchmark the paper-guided PCA-preprocessed TriCSI data.

This script compares:
    amplitude_pca10
    hampel_pca10
    savgol_pca10

using the exact same scientific split logic:
1) S1 5-fold whole-trial CV
2) S1 -> S2 cross-session
3) S2 -> S1 cross-session

Input
-----
data/preprocessed_paper/fit_S1/<variant>/<session>/*.csv

Each packet already contains:
    pc_01 ... pc_10

Window representation
---------------------
For every 1-second window and each link separately, summarize each PC with:
    median
    standard deviation
    IQR
    mean absolute adjacent-frame change

For 10 PCs x 4 stats x 2 links = 80 features.

We intentionally DO NOT rescale individual PCA components before the RF.
The PCA was fit using S1 only, and the paper reported that post-PCA scaling
reduced accuracy in its evaluated setups.

This is a compact "PCA + RF" baseline intended to answer:
    Did paper-guided preprocessing improve robustness compared with the
    original engineered-feature RF?

Run:
    python benchmark_preprocessed_pca.py

Output:
    results/preprocessed_pca_benchmark/
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from project_config import CLASS_ORDER, RESULTS_DIR


PC_RE = re.compile(r"^pc_(\d+)$", re.I)


def canonical_link(value: str) -> str | None:
    x = str(value).strip().lower()
    if x in {"left", "l", "tx_left", "tx-left"} or "left" in x:
        return "left"
    if x in {"right", "r", "tx_right", "tx-right"} or "right" in x:
        return "right"
    return None


def find_field(fieldnames: list[str], wanted: str) -> str:
    lookup = {x.lower(): x for x in fieldnames}
    if wanted.lower() not in lookup:
        raise ValueError(f"Missing required column {wanted!r}")
    return lookup[wanted.lower()]


def pc_columns(fieldnames: list[str], n_components: int) -> list[str]:
    found = []
    for name in fieldnames:
        m = PC_RE.match(name.strip())
        if m:
            found.append((int(m.group(1)), name))
    found.sort()
    cols = [name for _, name in found]
    if len(cols) < n_components:
        raise ValueError(
            f"Expected at least {n_components} PCA columns, found {len(cols)}"
        )
    return cols[:n_components]


def read_trial(path: Path, n_components: int, fallback_rate_hz: float) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header: {path}")
        fields = list(reader.fieldnames)
        rows = list(reader)

    link_field = find_field(fields, "link")
    zone_field = find_field(fields, "zone")
    trial_field = find_field(fields, "trial")
    orientation_field = find_field(fields, "orientation")
    session_field = find_field(fields, "session")
    time_field = None

    # Try common timestamp names first.
    for candidate in (
        "real_timestamp",
        "local_timestamp",
        "timestamp",
        "time",
        "time_s",
        "timestamp_s",
        "relative_time_s",
        "steady_timestamp",
        "steady_time",
        "steady_time_s",
        "ap_timestamp",
    ):
        try:
            time_field = find_field(fields, candidate)
            break
        except ValueError:
            pass

    # Then accept any remaining column whose name contains "time".
    if time_field is None:
        time_like = [
            name for name in fields
            if "time" in name.lower()
        ]
        if time_like:
            time_field = time_like[0]

    pcs = pc_columns(fields, n_components)

    by_link = {"left": [], "right": []}
    synthetic_counts = {"left": 0, "right": 0}

    for row in rows:
        link = canonical_link(row.get(link_field, ""))
        if link is None:
            continue

        vec = np.asarray(
            [float(row[c]) for c in pcs],
            dtype=np.float64,
        )

        if time_field is not None:
            try:
                t = float(row[time_field])
            except (TypeError, ValueError):
                t = synthetic_counts[link] / fallback_rate_hz
        else:
            # The preprocessed CSVs preserve packet order. When no explicit
            # timestamp column exists, reconstruct time from the known stable
            # packet rate (~33.3 Hz in this experiment). This is preferable to
            # failing and is appropriate because the link/rate quality checks
            # already showed essentially uniform packet cadence.
            t = synthetic_counts[link] / fallback_rate_hz

        synthetic_counts[link] += 1
        by_link[link].append((t, vec))

    for link in ("left", "right"):
        if not by_link[link]:
            raise ValueError(f"{path.name}: no {link} rows")
        by_link[link].sort(key=lambda x: x[0])

    # Normalize timestamps to seconds if they look like microseconds.
    def arrays(items):
        t = np.asarray([x[0] for x in items], dtype=np.float64)
        x = np.stack([x[1] for x in items], axis=0)
        if len(t) >= 2:
            median_dt = np.median(np.diff(t))
            if median_dt > 10:
                t = t / 1e6
        t = t - t[0]
        return t, x

    lt, lx = arrays(by_link["left"])
    rt, rx = arrays(by_link["right"])

    first = rows[0]
    return {
        "zone": str(first[zone_field]),
        "trial": int(str(first[trial_field]).strip()),
        "orientation": str(first[orientation_field]),
        "session": str(first[session_field]),
        "left_time": lt,
        "left_pc": lx,
        "right_time": rt,
        "right_pc": rx,
        "source_file": str(path),
    }


def summarize_pc_window(x: np.ndarray, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for j in range(x.shape[1]):
        v = x[:, j]
        q25, q75 = np.percentile(v, [25, 75])
        if len(v) >= 2:
            delta = float(np.mean(np.abs(np.diff(v))))
        else:
            delta = 0.0
        k = j + 1
        out[f"{prefix}_pc{k:02d}_median"] = float(np.median(v))
        out[f"{prefix}_pc{k:02d}_std"] = float(np.std(v))
        out[f"{prefix}_pc{k:02d}_iqr"] = float(q75 - q25)
        out[f"{prefix}_pc{k:02d}_delta"] = delta
    return out


def extract_windows(
    trial: dict[str, Any],
    *,
    window_s: float,
    hop_s: float,
    min_packets: int,
) -> list[dict[str, Any]]:
    lt = trial["left_time"]
    rt = trial["right_time"]
    lx = trial["left_pc"]
    rx = trial["right_pc"]

    common_start = max(float(lt[0]), float(rt[0]))
    common_end = min(float(lt[-1]), float(rt[-1]))

    rows = []
    start = common_start
    widx = 0

    while start + window_s <= common_end + 1e-9:
        end = start + window_s
        lm = (lt >= start) & (lt < end)
        rm = (rt >= start) & (rt < end)

        if int(np.sum(lm)) >= min_packets and int(np.sum(rm)) >= min_packets:
            feat = {}
            feat.update(summarize_pc_window(lx[lm], "L"))
            feat.update(summarize_pc_window(rx[rm], "R"))

            # Add a few cross-link relative features using corresponding PC medians.
            for j in range(lx.shape[1]):
                k = j + 1
                lmed = feat[f"L_pc{k:02d}_median"]
                rmed = feat[f"R_pc{k:02d}_median"]
                feat[f"X_pc{k:02d}_median_diff"] = lmed - rmed

            rows.append(
                {
                    "features": feat,
                    "zone": trial["zone"],
                    "trial": trial["trial"],
                    "orientation": trial["orientation"],
                    "session": trial["session"],
                    "source_file": trial["source_file"],
                    "trial_group": (
                        f"{trial['session']}|{trial['zone']}|"
                        f"{trial['trial']}|{trial['orientation']}"
                    ),
                    "window_index": widx,
                }
            )

        start += hop_s
        widx += 1

    return rows


def gather_variant(
    root: Path,
    variant: str,
    sessions: list[str],
    *,
    n_components: int,
    window_s: float,
    hop_s: float,
    min_packets: int,
    fallback_rate_hz: float,
) -> list[dict[str, Any]]:
    rows = []
    for session in sessions:
        folder = root / variant / session
        paths = sorted(folder.glob("*.csv"))
        if not paths:
            raise SystemExit(f"No CSVs found: {folder}")
        print(f"\n{variant} / {session}: {len(paths)} trials")
        for i, path in enumerate(paths, 1):
            print(f"  [{i:02d}/{len(paths):02d}] {path.name}")
            trial = read_trial(path, n_components, fallback_rate_hz)
            rows.extend(
                extract_windows(
                    trial,
                    window_s=window_s,
                    hop_s=hop_s,
                    min_packets=min_packets,
                )
            )
    return rows


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def matrix(rows: list[dict[str, Any]]):
    names = sorted(rows[0]["features"])
    X = np.asarray(
        [[r["features"][f] for f in names] for r in rows],
        dtype=np.float64,
    )
    y = np.asarray([r["zone"] for r in rows], dtype=object)
    return X, y, names


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def trial_mean_probability_accuracy(
    rows: list[dict[str, Any]],
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
) -> float:
    groups = np.asarray([r["trial_group"] for r in rows], dtype=object)
    true = []
    pred = []

    for group in sorted(set(groups)):
        idx = np.where(groups == group)[0]
        true.append(str(y_true[idx[0]]))
        mean_prob = probs[idx].mean(axis=0)
        pred.append(str(classes[int(np.argmax(mean_prob))]))

    return float(accuracy_score(true, pred))


def evaluate_cross(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, float]:
    Xtr, ytr, names = matrix(train_rows)
    Xte, yte, names2 = matrix(test_rows)
    if names != names2:
        raise RuntimeError("Feature names differ between train/test.")

    model = make_rf(seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    probs = model.predict_proba(Xte)

    out = metrics(yte, pred)
    out["trial_mean_probability_accuracy"] = (
        trial_mean_probability_accuracy(
            test_rows, yte, probs, model.classes_
        )
    )
    return out


def evaluate_5fold_s1(
    rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    fold_rows = []

    for fold in range(1, 6):
        train = [r for r in rows if r["trial"] != fold]
        test = [r for r in rows if r["trial"] == fold]
        result = evaluate_cross(train, test, seed + fold)
        fold_rows.append({"fold": fold, **result})

    acc = np.asarray([r["accuracy"] for r in fold_rows], dtype=float)
    f1 = np.asarray([r["macro_f1"] for r in fold_rows], dtype=float)

    return {
        "folds": fold_rows,
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=1)),
        "trial_mean_probability_accuracy_mean": float(
            np.mean(
                [r["trial_mean_probability_accuracy"] for r in fold_rows]
            )
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "preprocessed_paper" / "fit_S1",
    )
    p.add_argument("--components", type=int, default=10)
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--hop", type=float, default=0.25)
    p.add_argument("--min-packets", type=int, default=12)
    p.add_argument(
        "--fallback-rate-hz",
        type=float,
        default=33.333,
        help=(
            "Used only when a preprocessed CSV has no explicit timestamp "
            "column. Default matches the measured ~33.3 packet/s link rate."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(
        f"Timestamp fallback : {args.fallback_rate_hz:.3f} Hz "
        "(used only if no timestamp column exists)"
    )

    variants = [
        "amplitude_pca10",
        "hampel_pca10",
        "savgol_pca10",
    ]

    out_dir = RESULTS_DIR / "preprocessed_pca_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    full_summary = {}

    for variant in variants:
        print("\n" + "=" * 78)
        print(f"BENCHMARK: {variant}")
        print("=" * 78)

        all_rows = gather_variant(
            args.root,
            variant,
            ["S1", "S2"],
            n_components=args.components,
            window_s=args.window,
            hop_s=args.hop,
            min_packets=args.min_packets,
            fallback_rate_hz=args.fallback_rate_hz,
        )

        s1 = [r for r in all_rows if r["session"] == "S1"]
        s2 = [r for r in all_rows if r["session"] == "S2"]

        print(f"\nWindows: S1={len(s1)}, S2={len(s2)}")

        cv = evaluate_5fold_s1(s1, args.seed)
        s1s2 = evaluate_cross(s1, s2, args.seed)
        s2s1 = evaluate_cross(s2, s1, args.seed)

        row = {
            "variant": variant,
            "n_features": len(s1[0]["features"]),
            "s1_5fold_accuracy_mean": cv["accuracy_mean"],
            "s1_5fold_accuracy_std": cv["accuracy_std"],
            "s1_5fold_macro_f1_mean": cv["macro_f1_mean"],
            "s1_5fold_trial_mean_probability_accuracy_mean": (
                cv["trial_mean_probability_accuracy_mean"]
            ),
            "s1_to_s2_accuracy": s1s2["accuracy"],
            "s1_to_s2_macro_f1": s1s2["macro_f1"],
            "s1_to_s2_trial_mean_probability_accuracy": (
                s1s2["trial_mean_probability_accuracy"]
            ),
            "s2_to_s1_accuracy": s2s1["accuracy"],
            "s2_to_s1_macro_f1": s2s1["macro_f1"],
            "s2_to_s1_trial_mean_probability_accuracy": (
                s2s1["trial_mean_probability_accuracy"]
            ),
            "bidirectional_cross_session_mean_accuracy": float(
                (s1s2["accuracy"] + s2s1["accuracy"]) / 2.0
            ),
        }
        summary_rows.append(row)

        full_summary[variant] = {
            "s1_5fold": cv,
            "s1_to_s2": s1s2,
            "s2_to_s1": s2s1,
        }

        print("\nRESULT")
        print(
            f"  S1 5-fold    : "
            f"{cv['accuracy_mean']:.3f} ± {cv['accuracy_std']:.3f}"
        )
        print(
            f"  S1 -> S2     : {s1s2['accuracy']:.3f} "
            f"(trial mean-prob {s1s2['trial_mean_probability_accuracy']:.3f})"
        )
        print(
            f"  S2 -> S1     : {s2s1['accuracy']:.3f} "
            f"(trial mean-prob {s2s1['trial_mean_probability_accuracy']:.3f})"
        )

    summary_rows.sort(
        key=lambda r: r["bidirectional_cross_session_mean_accuracy"],
        reverse=True,
    )

    for rank, row in enumerate(summary_rows, 1):
        row["rank"] = rank

    fields = [
        "rank",
        "variant",
        "n_features",
        "s1_5fold_accuracy_mean",
        "s1_5fold_accuracy_std",
        "s1_5fold_macro_f1_mean",
        "s1_5fold_trial_mean_probability_accuracy_mean",
        "s1_to_s2_accuracy",
        "s1_to_s2_macro_f1",
        "s1_to_s2_trial_mean_probability_accuracy",
        "s2_to_s1_accuracy",
        "s2_to_s1_macro_f1",
        "s2_to_s1_trial_mean_probability_accuracy",
        "bidirectional_cross_session_mean_accuracy",
    ]

    with (out_dir / "benchmark_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "note": (
                    "PCA models were fit using the fit_S1 export. "
                    "S2 did not participate in PCA fitting."
                ),
                "ranked_results": summary_rows,
                "details": full_summary,
            },
            f,
            indent=2,
        )

    names = [r["variant"] for r in summary_rows][::-1]
    means = [
        r["bidirectional_cross_session_mean_accuracy"]
        for r in summary_rows
    ][::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    pos = np.arange(len(names))
    ax.barh(pos, means)
    ax.set_yticks(pos, labels=names)
    ax.set_xlabel("Mean cross-session window accuracy")
    ax.set_title("Paper-guided PCA preprocessing benchmark")
    for y, value in zip(pos, means):
        ax.text(value + 0.005, y, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_session_accuracy.png", dpi=180)
    plt.close(fig)

    print("\n" + "=" * 78)
    print("PAPER-PREPROCESSED PCA BENCHMARK RANKING")
    print("=" * 78)
    for row in summary_rows:
        print(
            f"{row['rank']}. {row['variant']:18s} | "
            f"S1 CV={row['s1_5fold_accuracy_mean']:.3f} | "
            f"S1->S2={row['s1_to_s2_accuracy']:.3f} | "
            f"S2->S1={row['s2_to_s1_accuracy']:.3f} | "
            f"cross mean={row['bidirectional_cross_session_mean_accuracy']:.3f}"
        )

    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
