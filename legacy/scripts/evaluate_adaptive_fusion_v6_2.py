from __future__ import annotations

"""
TriCSI V6.2 — OFFLINE ADAPTIVE-FUSION EVALUATOR

Uses the clean labeled-dwell CSV already collected by V6.1.
NO new CSI collection is required.

It compares:
  LEFT
  RIGHT
  CURRENT
  EQUAL
  GEO                 (existing fixed class-weighted geometric fusion)
  ADAPT_ARITH          (confidence-adaptive arithmetic fusion)
  ADAPT_GEO            (confidence-adaptive geometric/log-opinion fusion)
  DOMINANT_OR_GEO      (trust a clearly stronger link, otherwise use ADAPT_GEO)

It also prints ORACLE upper bounds. ORACLE is NOT deployable; it simply answers:
"Was the correct class present as the top prediction in at least one candidate?"
That tells us whether fusion still has meaningful headroom.

No hyperparameter sweep is performed. The adaptive rules are fixed.
"""

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from project_config import MODEL_DIR, RESULTS_DIR


EPS = 1e-6


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, 0.0)
    s = float(np.sum(p))
    if s <= 0:
        return np.full_like(p, 1.0 / len(p))
    return p / s


def entropy01(p: np.ndarray) -> float:
    p = normalize(p)
    h = -float(np.sum(p * np.log(np.maximum(p, EPS))))
    return h / math.log(len(p))


def margin(p: np.ndarray) -> float:
    p = np.sort(normalize(p))
    return float(p[-1] - p[-2]) if len(p) >= 2 else 1.0


def confidence(p: np.ndarray) -> float:
    """
    Generic per-link confidence in [0,1].

    Uses BOTH distribution sharpness and top1 separation:
        confidence = 0.5 * (1 - normalized entropy)
                   + 0.5 * top1-top2 margin

    A small floor keeps a diffuse link from disappearing completely.
    """
    c = 0.5 * (1.0 - entropy01(p)) + 0.5 * margin(p)
    return float(np.clip(c, 0.05, 1.0))


def top_zone(p: np.ndarray, classes: list[str]) -> str:
    return classes[int(np.argmax(p))]


def adaptive_arithmetic(
    p_left: np.ndarray,
    p_right: np.ndarray,
    class_weights: np.ndarray,
) -> np.ndarray:
    """
    Existing class reliability × current-window confidence.
    """
    c_left = confidence(p_left)
    c_right = confidence(p_right)

    w_left = class_weights[0] * c_left
    w_right = class_weights[1] * c_right

    score = (
        w_left * p_left
        + w_right * p_right
    )
    return normalize(score)


def adaptive_geometric(
    p_left: np.ndarray,
    p_right: np.ndarray,
    class_weights: np.ndarray,
) -> np.ndarray:
    """
    Confidence-adaptive logarithmic opinion pool.

    Each link's existing class-specific OOF reliability is multiplied by the
    current window's generic confidence before combining log probabilities.

    This is the main V6.2 candidate.
    """
    c_left = confidence(p_left)
    c_right = confidence(p_right)

    # Do not let confidence completely erase one link.
    # 0.35 + 0.65*c maps c∈[0,1] to [0.35,1].
    g_left = 0.35 + 0.65 * c_left
    g_right = 0.35 + 0.65 * c_right

    w_left = class_weights[0] * g_left
    w_right = class_weights[1] * g_right

    log_score = (
        w_left * np.log(np.maximum(p_left, EPS))
        + w_right * np.log(np.maximum(p_right, EPS))
    )

    log_score -= np.max(log_score)
    return normalize(np.exp(log_score))


def dominant_or_geo(
    p_left: np.ndarray,
    p_right: np.ndarray,
    p_geo_adaptive: np.ndarray,
) -> np.ndarray:
    """
    Generic disagreement gate.

    If one link is substantially more confident than the other, preserve that
    information instead of letting the weaker link overturn it. Otherwise use
    the adaptive geometric fusion.

    This is NOT zone-specific.
    """
    c_left = confidence(p_left)
    c_right = confidence(p_right)

    diff = c_right - c_left

    # Fixed conservative threshold; no sweep.
    if diff >= 0.18:
        return normalize(0.75 * p_right + 0.25 * p_geo_adaptive)

    if diff <= -0.18:
        return normalize(0.75 * p_left + 0.25 * p_geo_adaptive)

    return p_geo_adaptive


def find_latest_csv() -> Path:
    root = RESULTS_DIR / "live_fusion_diagnostic"
    paths = sorted(
        root.glob("fusion_diag_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not paths:
        raise SystemExit(
            f"No diagnostic CSV found under:\n{root}\n"
            "Run live_link_fusion_diagnostic_v6_1.py first."
        )
    return paths[-1]


def load_prob(row, prefix: str, classes: list[str]) -> np.ndarray:
    return normalize(
        np.asarray(
            [float(row[f"{prefix}_{z}"]) for z in classes],
            dtype=np.float64,
        )
    )


def accuracy(df: pd.DataFrame, col: str) -> float:
    return float(np.mean(df[col] == df["true_zone"]))


def mean_ptrue(df: pd.DataFrame, prob_col: str, classes: list[str]) -> float:
    vals = []
    for _, row in df.iterrows():
        p = row[prob_col]
        i = classes.index(str(row["true_zone"]))
        vals.append(float(p[i]))
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="V6.1 fusion_diag CSV. Default: newest one.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "link_ensemble_rf_s4_s5_s6.joblib",
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv is not None else find_latest_csv()

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    bundle = joblib.load(args.model)
    classes = [str(x) for x in bundle["occupied_classes"]]
    class_weights = np.asarray(bundle["weights"], dtype=np.float64)

    df = pd.read_csv(csv_path)

    required = {"true_zone"}
    for prefix in ("left", "right", "current", "equal", "geo"):
        for z in classes:
            required.add(f"{prefix}_{z}")

    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            "Diagnostic CSV is missing expected columns:\n"
            + "\n".join(missing[:20])
        )

    records = []

    for _, row in df.iterrows():
        p_left = load_prob(row, "left", classes)
        p_right = load_prob(row, "right", classes)
        p_current = load_prob(row, "current", classes)
        p_equal = load_prob(row, "equal", classes)
        p_geo = load_prob(row, "geo", classes)

        p_adapt_arith = adaptive_arithmetic(
            p_left, p_right, class_weights
        )
        p_adapt_geo = adaptive_geometric(
            p_left, p_right, class_weights
        )
        p_dom = dominant_or_geo(
            p_left, p_right, p_adapt_geo
        )

        rec = {
            "true_zone": str(row["true_zone"]),
            "left_top": top_zone(p_left, classes),
            "right_top": top_zone(p_right, classes),
            "current_top": top_zone(p_current, classes),
            "equal_top": top_zone(p_equal, classes),
            "geo_top": top_zone(p_geo, classes),
            "adapt_arith_top": top_zone(p_adapt_arith, classes),
            "adapt_geo_top": top_zone(p_adapt_geo, classes),
            "dominant_or_geo_top": top_zone(p_dom, classes),
            "p_left": p_left,
            "p_right": p_right,
            "p_current": p_current,
            "p_equal": p_equal,
            "p_geo": p_geo,
            "p_adapt_arith": p_adapt_arith,
            "p_adapt_geo": p_adapt_geo,
            "p_dominant_or_geo": p_dom,
            "left_conf": confidence(p_left),
            "right_conf": confidence(p_right),
        }

        # Non-deployable headroom indicators.
        rec["oracle_lr"] = (
            rec["left_top"] == rec["true_zone"]
            or rec["right_top"] == rec["true_zone"]
        )
        rec["oracle_existing"] = (
            rec["current_top"] == rec["true_zone"]
            or rec["geo_top"] == rec["true_zone"]
            or rec["right_top"] == rec["true_zone"]
            or rec["left_top"] == rec["true_zone"]
        )

        records.append(rec)

    out = pd.DataFrame(records)

    methods = [
        ("LEFT", "left_top", "p_left"),
        ("RIGHT", "right_top", "p_right"),
        ("CURRENT", "current_top", "p_current"),
        ("EQUAL", "equal_top", "p_equal"),
        ("GEO", "geo_top", "p_geo"),
        ("ADAPT_ARITH", "adapt_arith_top", "p_adapt_arith"),
        ("ADAPT_GEO", "adapt_geo_top", "p_adapt_geo"),
        ("DOM_OR_GEO", "dominant_or_geo_top", "p_dominant_or_geo"),
    ]

    print("\n" + "=" * 86)
    print("TriCSI V6.2 — ADAPTIVE FUSION EVALUATION")
    print("=" * 86)
    print(f"CSV   : {csv_path}")
    print(f"Rows  : {len(out)}")
    print(f"Model : {args.model}")
    print("=" * 86)

    print("\nOVERALL")
    for label, pred_col, prob_col in methods:
        print(
            f"{label:12s} "
            f"top1={accuracy(out, pred_col):.3f} "
            f"mean P(true)={mean_ptrue(out, prob_col, classes):.3f}"
        )

    print(
        f"\nORACLE L/R top1      = {float(np.mean(out['oracle_lr'])):.3f}"
    )
    print(
        f"ORACLE existing top1 = {float(np.mean(out['oracle_existing'])):.3f}"
    )
    print(
        "(ORACLE is only an upper-bound diagnostic; it cannot be deployed.)"
    )

    print("\nPER-ZONE TOP1")
    header = (
        "zone   LEFT RIGHT CURRENT GEO  "
        "A_ARITH A_GEO DOM_GEO ORACLE"
    )
    print(header)

    for zone in classes:
        z = out[out["true_zone"] == zone]
        if len(z) == 0:
            continue

        print(
            f"{zone:4s} "
            f"{accuracy(z, 'left_top'):5.3f} "
            f"{accuracy(z, 'right_top'):5.3f} "
            f"{accuracy(z, 'current_top'):7.3f} "
            f"{accuracy(z, 'geo_top'):5.3f} "
            f"{accuracy(z, 'adapt_arith_top'):7.3f} "
            f"{accuracy(z, 'adapt_geo_top'):5.3f} "
            f"{accuracy(z, 'dominant_or_geo_top'):7.3f} "
            f"{float(np.mean(z['oracle_existing'])):6.3f}"
        )

    print("\nMEAN LINK CONFIDENCE BY TRUE ZONE")
    print("zone   LEFT_conf RIGHT_conf")
    for zone in classes:
        z = out[out["true_zone"] == zone]
        if len(z) == 0:
            continue
        print(
            f"{zone:4s} "
            f"{float(z['left_conf'].mean()):9.3f} "
            f"{float(z['right_conf'].mean()):10.3f}"
        )

    # Save compact per-row predictions for inspection.
    save_cols = [
        "true_zone",
        "left_top",
        "right_top",
        "current_top",
        "geo_top",
        "adapt_arith_top",
        "adapt_geo_top",
        "dominant_or_geo_top",
        "left_conf",
        "right_conf",
        "oracle_lr",
        "oracle_existing",
    ]

    out_path = csv_path.with_name(
        csv_path.stem + "_adaptive_eval.csv"
    )
    out[save_cols].to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")

    print("\nDECISION RULE")
    print(
        "If ADAPT_GEO or DOM_OR_GEO clearly beats both RIGHT and GEO overall "
        "without collapsing Z1/Z2/Z6/Z8/Z9, use it for the next live localizer."
    )
    print(
        "If neither adaptive method beats ~0.50 materially, stop fusion tuning: "
        "the remaining failures are primarily representation/model failures."
    )


if __name__ == "__main__":
    main()
