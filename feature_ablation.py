from __future__ import annotations

"""
TriCSI engineered-feature ablation study.

Goal
----
Find which feature families are helping or hurting cross-session robustness
before moving to CNN / Transformer models.

Default development experiment:
    S1 -> S2
    S2 -> S1

The script extracts features ONCE, then evaluates several feature subsets with
the exact same Random Forest settings used by the existing TriCSI baseline.

Important:
- S1 and S2 are now DEVELOPMENT sessions because we are tuning based on them.
- Reserve a future S3 as the untouched final test session.

Run:
    python feature_ablation.py

Optional:
    python feature_ablation.py --session-a S1 --session-b S2
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from features import iter_trial_windows
from project_config import (
    CLASS_ORDER,
    HOP_SECONDS,
    MIN_PACKETS_PER_LINK,
    RAW_DATA_DIR,
    RESULTS_DIR,
    WINDOW_SECONDS,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def gather_trial_paths(session_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in session_dir.rglob("*.csv")
        if not p.name.startswith("schedule_")
    )


def gather_windows(
    paths: list[Path],
    *,
    window_seconds: float,
    hop_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for i, path in enumerate(paths, 1):
        print(f"[{i:03d}/{len(paths):03d}] features: {path.name}")
        rows.extend(
            iter_trial_windows(
                path,
                window_seconds=window_seconds,
                hop_seconds=hop_seconds,
                min_packets_per_link=MIN_PACKETS_PER_LINK,
            )
        )

    return rows


def matrix_from_rows(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        [
            [row["features"][name] for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.asarray([row["zone"] for row in rows], dtype=object)
    return X, y


# ---------------------------------------------------------------------------
# Feature grouping
# ---------------------------------------------------------------------------

def lower(name: str) -> str:
    return name.lower()


def has_any(name: str, tokens: tuple[str, ...]) -> bool:
    n = lower(name)
    return any(token in n for token in tokens)


def build_feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Build several deliberately broad feature families.

    The naming rules are heuristic because features.py may use slightly
    different prefixes/suffixes. The script prints and saves the exact members
    so the experiment is auditable.
    """

    rssi_tokens = (
        "rssi",
    )

    absolute_tokens = (
        "raw",
        "absolute",
        "abs_",
        "_abs",
        "level",
        "median_log",
        "med_log",
        "logamp",
        "log_amp",
    )

    shape_tokens = (
        "shape",
        "spectral",
        "subcarrier",
        "carrier",
    )

    dynamic_tokens = (
        "change",
        "delta",
        "temporal",
        "variance",
        "_var",
        "var_",
        "_std",
        "std_",
        "iqr",
        "p90",
        "adjacent",
        "frame",
        "motion",
        "euclid",
    )

    relative_tokens = (
        "cross",
        "left_right",
        "lr_",
        "_lr",
        "diff",
        "difference",
        "ratio",
        "relative",
        "minus",
    )

    rssi = [
        f for f in feature_names
        if has_any(f, rssi_tokens)
    ]

    absolute = [
        f for f in feature_names
        if has_any(f, absolute_tokens)
        and not has_any(f, shape_tokens + dynamic_tokens + relative_tokens)
    ]

    absolute_rssi = sorted(set(absolute + rssi))

    shape = [
        f for f in feature_names
        if has_any(f, shape_tokens)
        and not has_any(f, rssi_tokens)
    ]

    dynamic = [
        f for f in feature_names
        if has_any(f, dynamic_tokens)
        and not has_any(f, rssi_tokens)
    ]

    relative = [
        f for f in feature_names
        if has_any(f, relative_tokens)
    ]

    # Shape + dynamic + relative features, while intentionally removing the
    # most obviously session-sensitive absolute/RSSI features.
    shape_dynamic = [
        f for f in feature_names
        if (
            f in shape
            or f in dynamic
            or f in relative
        )
        and f not in absolute_rssi
    ]

    no_rssi = [
        f for f in feature_names
        if f not in rssi
    ]

    no_absolute_rssi = [
        f for f in feature_names
        if f not in absolute_rssi
    ]

    # Purely relative-ish subset. Include shape differences / dynamic changes
    # because these encode change/contrast rather than absolute level.
    relative_dynamic = sorted(
        set(relative + dynamic)
        - set(absolute_rssi)
    )

    groups = {
        "ALL": list(feature_names),
        "NO_RSSI": no_rssi,
        "NO_ABSOLUTE_RSSI": no_absolute_rssi,
        "SHAPE_DYNAMIC": shape_dynamic,
        "RELATIVE_DYNAMIC": relative_dynamic,
        "DYNAMIC_ONLY": dynamic,
        "SHAPE_ONLY": shape,
        "ABSOLUTE_RSSI_ONLY": absolute_rssi,
        "RSSI_ONLY": rssi,
    }

    # Remove duplicates while preserving original feature order.
    order = {name: i for i, name in enumerate(feature_names)}
    cleaned: dict[str, list[str]] = {}

    for group_name, members in groups.items():
        unique = sorted(set(members), key=lambda x: order[x])
        cleaned[group_name] = unique

    return cleaned


# ---------------------------------------------------------------------------
# Model + evaluation
# ---------------------------------------------------------------------------

def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def evaluate_direction(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_indices: list[int],
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    model = make_rf(seed)

    model.fit(
        X_train[:, selected_indices],
        y_train,
    )

    pred = model.predict(
        X_test[:, selected_indices]
    )

    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, pred)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

    return result, pred


def save_confusion(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    row_sum = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm.astype(np.float64),
        row_sum,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sum != 0,
    )

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    image = ax.imshow(normalized)

    ax.set_title(title)
    ax.set_xlabel("Predicted zone")
    ax.set_ylabel("True zone")
    ax.set_xticks(
        range(len(labels)),
        labels=labels,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(
        range(len(labels)),
        labels=labels,
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{normalized[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plt.close('all')


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-a",
        default="S1",
    )
    parser.add_argument(
        "--session-b",
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
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=2,
        help=(
            "Skip automatically generated groups with fewer than "
            "this many features."
        ),
    )
    args = parser.parse_args()

    session_a_dir = args.raw_dir / args.session_a
    session_b_dir = args.raw_dir / args.session_b

    if not session_a_dir.exists():
        raise SystemExit(
            f"Missing session folder: {session_a_dir}"
        )

    if not session_b_dir.exists():
        raise SystemExit(
            f"Missing session folder: {session_b_dir}"
        )

    paths_a = gather_trial_paths(session_a_dir)
    paths_b = gather_trial_paths(session_b_dir)

    print("\n" + "=" * 78)
    print("TriCSI CROSS-SESSION FEATURE ABLATION — headless plotting fix")
    print("=" * 78)
    print(
        f"Session A: {args.session_a} "
        f"({len(paths_a)} trials)"
    )
    print(
        f"Session B: {args.session_b} "
        f"({len(paths_b)} trials)"
    )
    print(
        "Each feature subset is tested in BOTH directions:"
    )
    print(
        f"  {args.session_a} -> {args.session_b}"
    )
    print(
        f"  {args.session_b} -> {args.session_a}"
    )
    print("=" * 78)

    print(
        f"\nExtracting {args.session_a} features..."
    )
    rows_a = gather_windows(
        paths_a,
        window_seconds=args.window,
        hop_seconds=args.hop,
    )

    print(
        f"\nExtracting {args.session_b} features..."
    )
    rows_b = gather_windows(
        paths_b,
        window_seconds=args.window,
        hop_seconds=args.hop,
    )

    if not rows_a or not rows_b:
        raise SystemExit(
            "No usable windows were extracted."
        )

    feature_names = sorted(
        rows_a[0]["features"]
    )

    feature_names_b = sorted(
        rows_b[0]["features"]
    )

    if feature_names != feature_names_b:
        raise SystemExit(
            "S1/S2 feature names do not match."
        )

    X_a, y_a = matrix_from_rows(
        rows_a,
        feature_names,
    )

    X_b, y_b = matrix_from_rows(
        rows_b,
        feature_names,
    )

    print("\nDATA")
    print(
        f"  {args.session_a}: "
        f"{len(X_a)} windows"
    )
    print(
        f"  {args.session_b}: "
        f"{len(X_b)} windows"
    )
    print(
        f"  Total engineered features: "
        f"{len(feature_names)}"
    )

    groups = build_feature_groups(
        feature_names
    )

    out_dir = (
        RESULTS_DIR
        / f"feature_ablation_"
          f"{args.session_a.lower()}_"
          f"{args.session_b.lower()}"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save exact feature names + membership.
    membership_rows: list[dict[str, Any]] = []

    for feature in feature_names:
        row: dict[str, Any] = {
            "feature": feature,
        }

        for group_name, members in groups.items():
            row[group_name] = int(
                feature in members
            )

        membership_rows.append(row)

    write_csv(
        out_dir / "feature_group_membership.csv",
        membership_rows,
    )

    print("\nFEATURE GROUPS")
    for group_name, members in groups.items():
        print(
            f"\n{group_name}: "
            f"{len(members)} features"
        )

        if members:
            for feature in members[:12]:
                print(f"  - {feature}")

            if len(members) > 12:
                print(
                    f"  ... +{len(members) - 12} more"
                )
        else:
            print(
                "  (no features matched)"
            )

    feature_to_index = {
        name: i
        for i, name in enumerate(feature_names)
    }

    labels = [
        label
        for label in CLASS_ORDER
        if (
            label in set(y_a)
            and label in set(y_b)
        )
    ]

    result_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}

    for group_name, members in groups.items():
        if len(members) < args.min_features:
            print(
                f"\nSKIP {group_name}: "
                f"only {len(members)} feature(s)"
            )
            continue

        selected_indices = [
            feature_to_index[name]
            for name in members
        ]

        print("\n" + "-" * 78)
        print(
            f"{group_name}: "
            f"{len(members)} features"
        )
        print("-" * 78)

        # A -> B
        result_ab, pred_ab = evaluate_direction(
            X_train=X_a,
            y_train=y_a,
            X_test=X_b,
            y_test=y_b,
            selected_indices=selected_indices,
            seed=args.seed,
        )

        # B -> A
        result_ba, pred_ba = evaluate_direction(
            X_train=X_b,
            y_train=y_b,
            X_test=X_a,
            y_test=y_a,
            selected_indices=selected_indices,
            seed=args.seed,
        )

        mean_accuracy = (
            result_ab["accuracy"]
            + result_ba["accuracy"]
        ) / 2.0

        mean_macro_f1 = (
            result_ab["macro_f1"]
            + result_ba["macro_f1"]
        ) / 2.0

        worst_direction_accuracy = min(
            result_ab["accuracy"],
            result_ba["accuracy"],
        )

        print(
            f"{args.session_a}->{args.session_b}: "
            f"acc={result_ab['accuracy']:.3f} "
            f"macroF1={result_ab['macro_f1']:.3f}"
        )
        print(
            f"{args.session_b}->{args.session_a}: "
            f"acc={result_ba['accuracy']:.3f} "
            f"macroF1={result_ba['macro_f1']:.3f}"
        )
        print(
            f"Bidirectional mean: "
            f"acc={mean_accuracy:.3f} "
            f"macroF1={mean_macro_f1:.3f}"
        )

        result_rows.append(
            {
                "group": group_name,
                "n_features": len(members),
                f"{args.session_a}_to_{args.session_b}_accuracy":
                    result_ab["accuracy"],
                f"{args.session_a}_to_{args.session_b}_balanced_accuracy":
                    result_ab["balanced_accuracy"],
                f"{args.session_a}_to_{args.session_b}_macro_f1":
                    result_ab["macro_f1"],
                f"{args.session_b}_to_{args.session_a}_accuracy":
                    result_ba["accuracy"],
                f"{args.session_b}_to_{args.session_a}_balanced_accuracy":
                    result_ba["balanced_accuracy"],
                f"{args.session_b}_to_{args.session_a}_macro_f1":
                    result_ba["macro_f1"],
                "mean_bidirectional_accuracy":
                    mean_accuracy,
                "mean_bidirectional_macro_f1":
                    mean_macro_f1,
                "worst_direction_accuracy":
                    worst_direction_accuracy,
            }
        )

        detailed[group_name] = {
            "features": members,
            f"{args.session_a}_to_{args.session_b}":
                result_ab,
            f"{args.session_b}_to_{args.session_a}":
                result_ba,
            "mean_bidirectional_accuracy":
                mean_accuracy,
            "mean_bidirectional_macro_f1":
                mean_macro_f1,
            "worst_direction_accuracy":
                worst_direction_accuracy,
        }

        safe_group = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            group_name.lower(),
        )

        save_confusion(
            y_true=y_b,
            y_pred=pred_ab,
            labels=labels,
            path=(
                out_dir
                / f"{safe_group}_"
                  f"{args.session_a.lower()}_to_"
                  f"{args.session_b.lower()}_confusion.png"
            ),
            title=(
                f"{group_name}: "
                f"{args.session_a}->{args.session_b}"
            ),
        )

        save_confusion(
            y_true=y_a,
            y_pred=pred_ba,
            labels=labels,
            path=(
                out_dir
                / f"{safe_group}_"
                  f"{args.session_b.lower()}_to_"
                  f"{args.session_a.lower()}_confusion.png"
            ),
            title=(
                f"{group_name}: "
                f"{args.session_b}->{args.session_a}"
            ),
        )

    if not result_rows:
        raise SystemExit(
            "No feature groups contained enough features."
        )

    # Rank by bidirectional mean accuracy.
    result_rows.sort(
        key=lambda row:
            row["mean_bidirectional_accuracy"],
        reverse=True,
    )

    for rank, row in enumerate(
        result_rows,
        1,
    ):
        row["rank"] = rank

    # Reorder CSV columns.
    csv_fields = [
        "rank",
        "group",
        "n_features",
        f"{args.session_a}_to_{args.session_b}_accuracy",
        f"{args.session_a}_to_{args.session_b}_balanced_accuracy",
        f"{args.session_a}_to_{args.session_b}_macro_f1",
        f"{args.session_b}_to_{args.session_a}_accuracy",
        f"{args.session_b}_to_{args.session_a}_balanced_accuracy",
        f"{args.session_b}_to_{args.session_a}_macro_f1",
        "mean_bidirectional_accuracy",
        "mean_bidirectional_macro_f1",
        "worst_direction_accuracy",
    ]

    write_csv(
        out_dir / "feature_ablation_results.csv",
        result_rows,
        fieldnames=csv_fields,
    )

    # Bar chart.
    names = [
        row["group"]
        for row in result_rows
    ][::-1]

    means = [
        row["mean_bidirectional_accuracy"]
        for row in result_rows
    ][::-1]

    fig, ax = plt.subplots(
        figsize=(10, 6.5)
    )
    positions = np.arange(
        len(names)
    )
    ax.barh(
        positions,
        means,
    )
    ax.set_yticks(
        positions,
        labels=names,
    )
    ax.set_xlabel(
        "Mean cross-session accuracy "
        f"({args.session_a}->{args.session_b} and "
        f"{args.session_b}->{args.session_a})"
    )
    ax.set_title(
        "TriCSI engineered-feature ablation"
    )
    ax.set_xlim(
        0.0,
        max(1.0, max(means) * 1.08),
    )

    for y_pos, value in zip(
        positions,
        means,
    ):
        ax.text(
            value + 0.01,
            y_pos,
            f"{value:.3f}",
            va="center",
        )

    fig.tight_layout()
    fig.savefig(
        out_dir / "feature_ablation_accuracy.png",
        dpi=180,
    )
    plt.close(fig)
    plt.close('all')

    baseline = next(
        (
            row
            for row in result_rows
            if row["group"] == "ALL"
        ),
        None,
    )

    best = result_rows[0]

    summary = {
        "session_a": args.session_a,
        "session_b": args.session_b,
        "development_note": (
            "S1 and S2 are being used for feature/model development. "
            "Do not use a future S3 for tuning if S3 is intended as the "
            "final untouched test session."
        ),
        "window_seconds": args.window,
        "hop_seconds": args.hop,
        "total_features": len(feature_names),
        "baseline_all_features": baseline,
        "best_group": best,
        "ranked_results": result_rows,
        "feature_groups": detailed,
    }

    with (
        out_dir / "feature_ablation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print("\n" + "=" * 78)
    print("FEATURE ABLATION RANKING")
    print("=" * 78)

    for row in result_rows:
        print(
            f"{row['rank']:2d}. "
            f"{row['group']:22s} "
            f"features={row['n_features']:3d}  "
            f"{args.session_a}->{args.session_b}="
            f"{row[f'{args.session_a}_to_{args.session_b}_accuracy']:.3f}  "
            f"{args.session_b}->{args.session_a}="
            f"{row[f'{args.session_b}_to_{args.session_a}_accuracy']:.3f}  "
            f"mean={row['mean_bidirectional_accuracy']:.3f}"
        )

    if baseline is not None:
        improvement = (
            best["mean_bidirectional_accuracy"]
            - baseline["mean_bidirectional_accuracy"]
        )

        print("\nBEST VS ALL-FEATURE BASELINE")
        print(
            f"  Best group : {best['group']}"
        )
        print(
            f"  Best mean  : "
            f"{best['mean_bidirectional_accuracy']:.3f}"
        )
        print(
            f"  ALL mean   : "
            f"{baseline['mean_bidirectional_accuracy']:.3f}"
        )
        print(
            f"  Difference : "
            f"{improvement:+.3f}"
        )

    print("\nSAVED")
    print(
        f"  {out_dir / 'feature_ablation_results.csv'}"
    )
    print(
        f"  {out_dir / 'feature_group_membership.csv'}"
    )
    print(
        f"  {out_dir / 'feature_ablation_accuracy.png'}"
    )
    print(
        f"  {out_dir / 'feature_ablation_summary.json'}"
    )
    print(
        "  Plus normalized confusion matrices for every tested group/direction."
    )


if __name__ == "__main__":
    main()
