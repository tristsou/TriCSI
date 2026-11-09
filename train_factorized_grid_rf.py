from __future__ import annotations

"""
TriCSI FACTORIZED 3x3 GRID RANDOM FOREST

One focused experiment using EXISTING standardized development data only.

Idea
----
Instead of learning 9 unrelated occupied-zone classes:

    Z1 Z2 Z3
    Z4 Z5 Z6
    Z7 Z8 Z9

learn two easier 3-class problems:

    COLUMN: LEFT / CENTER / RIGHT
    ROW:    TOP / MIDDLE / BOTTOM

Then reconstruct zone probabilities by:

    P(zone) = P(row) * P(column)

Examples:
    Z1 = TOP    * LEFT
    Z4 = MIDDLE * LEFT
    Z7 = BOTTOM * LEFT
    Z9 = BOTTOM * RIGHT

EMPTY remains a separate binary RF gate.

Evaluation
----------
Leave-one-session-out on S4/S5/S6:

    S5+S6 -> S4
    S4+S6 -> S5
    S4+S5 -> S6

The script also evaluates the standard 9-class RF on the EXACT SAME splits
as a control.

No new trials.
No hyperparameter sweep.
No S7.
No tracker tuning.

At the end, it trains one final factorized model on S4+S5+S6 and saves:

    models/factorized_grid_rf_s4_s5_s6.joblib

Run:
    python train_factorized_grid_rf.py
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from project_config import (
    CLASS_ORDER,
    RAW_DATA_DIR,
    RESULTS_DIR,
    MODEL_DIR,
    WINDOW_SECONDS,
    HOP_SECONDS,
)

from train_crossval import (
    gather_windows,
    make_rf,
    trial_number_from_row,
)
from train_csi_cnn import trial_paths_for_session


SESSIONS = ("S4", "S5", "S6")

OCCUPIED_CLASSES = tuple(
    z for z in CLASS_ORDER
    if z != "EMPTY"
)

ROW_CLASSES = ("TOP", "MIDDLE", "BOTTOM")
COL_CLASSES = ("LEFT", "CENTER", "RIGHT")

ZONE_TO_ROW = {
    "Z1": "TOP",
    "Z2": "TOP",
    "Z3": "TOP",
    "Z4": "MIDDLE",
    "Z5": "MIDDLE",
    "Z6": "MIDDLE",
    "Z7": "BOTTOM",
    "Z8": "BOTTOM",
    "Z9": "BOTTOM",
}

ZONE_TO_COL = {
    "Z1": "LEFT",
    "Z2": "CENTER",
    "Z3": "RIGHT",
    "Z4": "LEFT",
    "Z5": "CENTER",
    "Z6": "RIGHT",
    "Z7": "LEFT",
    "Z8": "CENTER",
    "Z9": "RIGHT",
}

ROW_COL_TO_ZONE = {
    (ZONE_TO_ROW[z], ZONE_TO_COL[z]): z
    for z in OCCUPIED_CLASSES
}

# Same existing tape/grid centers, meters.
ZONE_CENTERS = {
    "Z1": (0.4064, 1.5240),
    "Z2": (1.1176, 1.5240),
    "Z3": (1.8288, 1.5240),
    "Z4": (0.4064, 0.9144),
    "Z5": (1.1176, 0.9144),
    "Z6": (1.8288, 0.9144),
    "Z7": (0.4064, 0.3048),
    "Z8": (1.1176, 0.3048),
    "Z9": (1.8288, 0.3048),
}

GRID_POS = {
    "Z1": (0, 0),
    "Z2": (1, 0),
    "Z3": (2, 0),
    "Z4": (0, 1),
    "Z5": (1, 1),
    "Z6": (2, 1),
    "Z7": (0, 2),
    "Z8": (1, 2),
    "Z9": (2, 2),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_rows(
    raw_dir: Path,
    session: str,
    window: float,
    hop: float,
):
    paths = trial_paths_for_session(
        raw_dir,
        session,
    )

    if not paths:
        raise SystemExit(
            f"No trial files found for {session}"
        )

    rows = gather_windows(
        paths,
        window_seconds=window,
        hop_seconds=hop,
    )

    for r in rows:
        r["_session"] = session

    print(
        f"{session}: "
        f"{len(paths)} trials, "
        f"{len(rows)} windows"
    )

    return rows


def matrix(
    rows,
    feature_names,
):
    X = np.asarray(
        [
            [
                float(r["features"][name])
                for name in feature_names
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        [str(r["zone"]) for r in rows],
        dtype=object,
    )

    return X, y


def trial_group(r):
    return (
        f"{r['_session']}|"
        f"{r['zone']}|"
        f"{trial_number_from_row(r)}|"
        f"{r['orientation']}"
    )


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def align_proba(
    model,
    X,
    ordered_classes,
):
    raw = model.predict_proba(X)

    lookup = {
        str(c): i
        for i, c in enumerate(model.classes_)
    }

    out = np.zeros(
        (len(X), len(ordered_classes)),
        dtype=np.float64,
    )

    for j, cls in enumerate(ordered_classes):
        if cls in lookup:
            out[:, j] = raw[:, lookup[cls]]

    row_sum = out.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        out,
        row_sum,
        out=np.full_like(
            out,
            1.0 / len(ordered_classes),
        ),
        where=row_sum > 0,
    )


def combine_row_col_probabilities(
    p_row,
    p_col,
):
    """
    Convert:
        P(TOP/MIDDLE/BOTTOM)
        P(LEFT/CENTER/RIGHT)

    into:
        P(Z1)...P(Z9)

    using the product rule:
        P(zone) ∝ P(row) * P(column)

    Then normalize over the 9 zones.
    """

    row_idx = {
        name: i
        for i, name in enumerate(ROW_CLASSES)
    }

    col_idx = {
        name: i
        for i, name in enumerate(COL_CLASSES)
    }

    p_zone = np.zeros(
        (len(p_row), len(OCCUPIED_CLASSES)),
        dtype=np.float64,
    )

    for j, zone in enumerate(OCCUPIED_CLASSES):
        r = ZONE_TO_ROW[zone]
        c = ZONE_TO_COL[zone]

        p_zone[:, j] = (
            p_row[:, row_idx[r]]
            * p_col[:, col_idx[c]]
        )

    s = p_zone.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        p_zone,
        s,
        out=np.full_like(
            p_zone,
            1.0 / len(OCCUPIED_CLASSES),
        ),
        where=s > 0,
    )


def weighted_xy(
    zone_probs,
):
    centers = np.asarray(
        [
            ZONE_CENTERS[z]
            for z in OCCUPIED_CLASSES
        ],
        dtype=np.float64,
    )

    return zone_probs @ centers


def physical_error_m(
    true_zone,
    xy,
):
    if true_zone == "EMPTY":
        return None

    true_xy = np.asarray(
        ZONE_CENTERS[true_zone],
        dtype=np.float64,
    )

    return float(
        np.linalg.norm(
            true_xy
            - np.asarray(xy, dtype=np.float64)
        )
    )


def adjacent(
    true_zone,
    pred_zone,
):
    if true_zone == "EMPTY":
        return int(
            pred_zone == "EMPTY"
        )

    if pred_zone == "EMPTY":
        return 0

    tx, ty = GRID_POS[true_zone]
    px, py = GRID_POS[pred_zone]

    return int(
        max(
            abs(tx - px),
            abs(ty - py),
        )
        <= 1
    )


# ---------------------------------------------------------------------------
# Fit factorized model
# ---------------------------------------------------------------------------

def fit_factorized(
    train_rows,
    feature_names,
    seed,
):
    X, y = matrix(
        train_rows,
        feature_names,
    )

    # Stage 1:
    # EMPTY vs OCCUPIED
    y_occ = np.where(
        y == "EMPTY",
        "EMPTY",
        "OCCUPIED",
    )

    occupancy_model = make_rf(seed)
    occupancy_model.fit(
        X,
        y_occ,
    )

    # Stage 2:
    # occupied-only row / column models
    occ_mask = y != "EMPTY"

    X_occ = X[occ_mask]
    y_zone = y[occ_mask]

    y_row = np.asarray(
        [
            ZONE_TO_ROW[str(z)]
            for z in y_zone
        ],
        dtype=object,
    )

    y_col = np.asarray(
        [
            ZONE_TO_COL[str(z)]
            for z in y_zone
        ],
        dtype=object,
    )

    row_model = make_rf(
        seed + 1000
    )
    col_model = make_rf(
        seed + 2000
    )

    row_model.fit(
        X_occ,
        y_row,
    )

    col_model.fit(
        X_occ,
        y_col,
    )

    return {
        "occupancy_model": occupancy_model,
        "row_model": row_model,
        "col_model": col_model,
    }


def predict_factorized(
    bundle,
    X,
    empty_threshold,
):
    # EMPTY gate
    occ_model = bundle["occupancy_model"]

    occ_raw = occ_model.predict_proba(X)

    occ_lookup = {
        str(c): i
        for i, c in enumerate(
            occ_model.classes_
        )
    }

    p_empty = occ_raw[
        :,
        occ_lookup["EMPTY"],
    ]

    # Row / column
    p_row = align_proba(
        bundle["row_model"],
        X,
        ROW_CLASSES,
    )

    p_col = align_proba(
        bundle["col_model"],
        X,
        COL_CLASSES,
    )

    p_zone = combine_row_col_probabilities(
        p_row,
        p_col,
    )

    hard_zone = np.asarray(
        [
            OCCUPIED_CLASSES[i]
            for i in np.argmax(
                p_zone,
                axis=1,
            )
        ],
        dtype=object,
    )

    pred = np.where(
        p_empty >= empty_threshold,
        "EMPTY",
        hard_zone,
    )

    return {
        "pred": pred,
        "p_empty": p_empty,
        "p_row": p_row,
        "p_col": p_col,
        "p_zone": p_zone,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_factorized(
    train_rows,
    test_rows,
    feature_names,
    seed,
    empty_threshold,
):
    bundle = fit_factorized(
        train_rows,
        feature_names,
        seed,
    )

    X_test, y_test = matrix(
        test_rows,
        feature_names,
    )

    pred = predict_factorized(
        bundle,
        X_test,
        empty_threshold,
    )

    # Window-level exact
    window_acc = float(
        accuracy_score(
            y_test,
            pred["pred"],
        )
    )

    window_f1 = float(
        f1_score(
            y_test,
            pred["pred"],
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )
    )

    groups = np.asarray(
        [
            trial_group(r)
            for r in test_rows
        ],
        dtype=object,
    )

    trial_rows = []

    y_trial = []
    p_trial = []

    occupied_errors = []

    # row / col diagnostics at trial level
    row_true = []
    row_pred = []

    col_true = []
    col_pred = []

    for group in sorted(set(groups)):
        idx = np.where(
            groups == group
        )[0]

        true_zone = str(
            y_test[idx[0]]
        )

        mean_empty = float(
            np.mean(
                pred["p_empty"][idx]
            )
        )

        mean_row = np.mean(
            pred["p_row"][idx],
            axis=0,
        )

        mean_col = np.mean(
            pred["p_col"][idx],
            axis=0,
        )

        mean_zone = (
            combine_row_col_probabilities(
                mean_row[None, :],
                mean_col[None, :],
            )[0]
        )

        occupied_zone = (
            OCCUPIED_CLASSES[
                int(
                    np.argmax(
                        mean_zone
                    )
                )
            ]
        )

        final_zone = (
            "EMPTY"
            if mean_empty >= empty_threshold
            else occupied_zone
        )

        xy = weighted_xy(
            mean_zone[None, :]
        )[0]

        err = physical_error_m(
            true_zone,
            xy,
        )

        if err is not None:
            occupied_errors.append(err)

        sample = test_rows[
            int(idx[0])
        ]

        predicted_row = (
            ROW_CLASSES[
                int(
                    np.argmax(
                        mean_row
                    )
                )
            ]
        )

        predicted_col = (
            COL_CLASSES[
                int(
                    np.argmax(
                        mean_col
                    )
                )
            ]
        )

        if true_zone != "EMPTY":
            row_true.append(
                ZONE_TO_ROW[true_zone]
            )
            row_pred.append(
                predicted_row
            )

            col_true.append(
                ZONE_TO_COL[true_zone]
            )
            col_pred.append(
                predicted_col
            )

        trial_rows.append(
            {
                "trial_group": group,
                "true_zone": true_zone,
                "pred_zone": final_zone,
                "trial": int(
                    trial_number_from_row(
                        sample
                    )
                ),
                "orientation": str(
                    sample["orientation"]
                ),
                "p_empty": mean_empty,
                "pred_row": predicted_row,
                "pred_col": predicted_col,
                "x_m": float(xy[0]),
                "y_m": float(xy[1]),
                "physical_error_m": err,
                "physical_error_ft": (
                    None
                    if err is None
                    else err * 3.28084
                ),
                "correct": int(
                    final_zone == true_zone
                ),
                "exact_or_adjacent": adjacent(
                    true_zone,
                    final_zone,
                ),
            }
        )

        y_trial.append(
            true_zone
        )

        p_trial.append(
            final_zone
        )

    y_trial = np.asarray(
        y_trial,
        dtype=object,
    )

    p_trial = np.asarray(
        p_trial,
        dtype=object,
    )

    occupied_trial_rows = [
        r
        for r in trial_rows
        if r["true_zone"] != "EMPTY"
    ]

    errors = np.asarray(
        occupied_errors,
        dtype=np.float64,
    )

    # Per-zone exact trial accuracy
    per_zone = {}

    for zone in CLASS_ORDER:
        zrows = [
            r
            for r in trial_rows
            if r["true_zone"] == zone
        ]

        if not zrows:
            continue

        per_zone[zone] = float(
            np.mean(
                [
                    r["correct"]
                    for r in zrows
                ]
            )
        )

    metrics = {
        "window_accuracy": window_acc,
        "window_macro_f1": window_f1,
        "trial_accuracy": float(
            accuracy_score(
                y_trial,
                p_trial,
            )
        ),
        "trial_macro_f1": float(
            f1_score(
                y_trial,
                p_trial,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "row_trial_accuracy_occupied": float(
            accuracy_score(
                row_true,
                row_pred,
            )
        ),
        "column_trial_accuracy_occupied": float(
            accuracy_score(
                col_true,
                col_pred,
            )
        ),
        "adjacent_accuracy_occupied": float(
            np.mean(
                [
                    r["exact_or_adjacent"]
                    for r in occupied_trial_rows
                ]
            )
        ),
        "median_xy_error_m": float(
            np.median(errors)
        ),
        "p90_xy_error_m": float(
            np.percentile(
                errors,
                90,
            )
        ),
        "per_zone_trial_accuracy": per_zone,
    }

    return (
        metrics,
        trial_rows,
        bundle,
    )


# ---------------------------------------------------------------------------
# Plain 9-class RF control
# ---------------------------------------------------------------------------

def evaluate_plain_rf(
    train_rows,
    test_rows,
    feature_names,
    seed,
):
    X_train, y_train = matrix(
        train_rows,
        feature_names,
    )

    X_test, y_test = matrix(
        test_rows,
        feature_names,
    )

    model = make_rf(seed)

    model.fit(
        X_train,
        y_train,
    )

    p = align_proba(
        model,
        X_test,
        CLASS_ORDER,
    )

    groups = np.asarray(
        [
            trial_group(r)
            for r in test_rows
        ],
        dtype=object,
    )

    true = []
    pred = []

    per_zone_correct = defaultdict(list)

    for group in sorted(set(groups)):
        idx = np.where(
            groups == group
        )[0]

        mean_prob = np.mean(
            p[idx],
            axis=0,
        )

        true_zone = str(
            y_test[idx[0]]
        )

        pred_zone = (
            CLASS_ORDER[
                int(
                    np.argmax(
                        mean_prob
                    )
                )
            ]
        )

        true.append(
            true_zone
        )

        pred.append(
            pred_zone
        )

        per_zone_correct[
            true_zone
        ].append(
            int(
                pred_zone == true_zone
            )
        )

    return {
        "trial_accuracy": float(
            accuracy_score(
                true,
                pred,
            )
        ),
        "trial_macro_f1": float(
            f1_score(
                true,
                pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "per_zone_trial_accuracy": {
            zone: float(
                np.mean(values)
            )
            for zone, values
            in per_zone_correct.items()
        },
    }


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def write_csv(
    path,
    rows,
):
    if not rows:
        return

    keys = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore",
        )

        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

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
        "--empty-threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TriCSI FACTORIZED 3x3 GRID RF"
    )

    print(
        "=" * 78
    )

    print(
        "One experiment only:"
    )

    print(
        "  9 occupied zones"
        " -> ROW(3) + COLUMN(3)"
    )

    print(
        "Development data: S4/S5/S6 only"
    )

    print(
        "New trials: NO"
    )

    print(
        "Hyperparameter sweep: NO"
    )

    print(
        "=" * 78
    )

    rows_by_session = {
        session: load_rows(
            args.raw_dir,
            session,
            args.window,
            args.hop,
        )
        for session in SESSIONS
    }

    feature_names = sorted(
        rows_by_session[
            "S4"
        ][0]["features"]
    )

    for session in (
        "S5",
        "S6",
    ):
        other = sorted(
            rows_by_session[
                session
            ][0]["features"]
        )

        if other != feature_names:
            raise RuntimeError(
                f"{session} feature set "
                "does not match S4."
            )

    out_dir = (
        RESULTS_DIR
        / "factorized_grid_rf_s4_s6"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    all_trial_rows = []
    per_zone_rows = []

    for outer_i, test_session in enumerate(
        SESSIONS
    ):
        train_sessions = [
            s
            for s in SESSIONS
            if s != test_session
        ]

        train_rows = []

        for s in train_sessions:
            train_rows.extend(
                rows_by_session[s]
            )

        test_rows = (
            rows_by_session[
                test_session
            ]
        )

        print(
            "\n"
            + "=" * 78
        )

        print(
            f"{'+'.join(train_sessions)}"
            f" -> {test_session}"
        )

        print(
            "=" * 78
        )

        seed = (
            args.seed
            + outer_i * 10000
        )

        factorized_metrics, trial_rows, _ = (
            evaluate_factorized(
                train_rows,
                test_rows,
                feature_names,
                seed,
                args.empty_threshold,
            )
        )

        plain_metrics = (
            evaluate_plain_rf(
                train_rows,
                test_rows,
                feature_names,
                seed + 9000,
            )
        )

        print(
            f"Plain 9-class RF trial : "
            f"{plain_metrics['trial_accuracy']:.3f}"
        )

        print(
            f"Factorized trial       : "
            f"{factorized_metrics['trial_accuracy']:.3f}"
        )

        print(
            f"Factorized row acc     : "
            f"{factorized_metrics['row_trial_accuracy_occupied']:.3f}"
        )

        print(
            f"Factorized column acc  : "
            f"{factorized_metrics['column_trial_accuracy_occupied']:.3f}"
        )

        print(
            f"Adjacent occupied      : "
            f"{factorized_metrics['adjacent_accuracy_occupied']:.3f}"
        )

        print(
            f"Median XY error        : "
            f"{factorized_metrics['median_xy_error_m']:.3f} m"
        )

        print(
            f"P90 XY error           : "
            f"{factorized_metrics['p90_xy_error_m']:.3f} m"
        )

        print(
            "\nLEFT COLUMN:"
        )

        for zone in (
            "Z1",
            "Z4",
            "Z7",
        ):
            plain_z = (
                plain_metrics[
                    "per_zone_trial_accuracy"
                ].get(
                    zone,
                    float("nan"),
                )
            )

            fact_z = (
                factorized_metrics[
                    "per_zone_trial_accuracy"
                ].get(
                    zone,
                    float("nan"),
                )
            )

            print(
                f"  {zone}: "
                f"plain={plain_z:.3f} "
                f"factorized={fact_z:.3f}"
            )

        summary_rows.append(
            {
                "train_sessions": "+".join(
                    train_sessions
                ),
                "test_session": test_session,
                "plain_trial_accuracy": (
                    plain_metrics[
                        "trial_accuracy"
                    ]
                ),
                "factorized_trial_accuracy": (
                    factorized_metrics[
                        "trial_accuracy"
                    ]
                ),
                "factorized_trial_macro_f1": (
                    factorized_metrics[
                        "trial_macro_f1"
                    ]
                ),
                "row_trial_accuracy_occupied": (
                    factorized_metrics[
                        "row_trial_accuracy_occupied"
                    ]
                ),
                "column_trial_accuracy_occupied": (
                    factorized_metrics[
                        "column_trial_accuracy_occupied"
                    ]
                ),
                "adjacent_accuracy_occupied": (
                    factorized_metrics[
                        "adjacent_accuracy_occupied"
                    ]
                ),
                "median_xy_error_m": (
                    factorized_metrics[
                        "median_xy_error_m"
                    ]
                ),
                "p90_xy_error_m": (
                    factorized_metrics[
                        "p90_xy_error_m"
                    ]
                ),
            }
        )

        for row in trial_rows:
            all_trial_rows.append(
                {
                    "test_session": test_session,
                    **row,
                }
            )

        for zone in CLASS_ORDER:
            per_zone_rows.append(
                {
                    "test_session": test_session,
                    "zone": zone,
                    "plain_trial_accuracy": (
                        plain_metrics[
                            "per_zone_trial_accuracy"
                        ].get(
                            zone,
                            None,
                        )
                    ),
                    "factorized_trial_accuracy": (
                        factorized_metrics[
                            "per_zone_trial_accuracy"
                        ].get(
                            zone,
                            None,
                        )
                    ),
                }
            )

    plain = np.asarray(
        [
            r[
                "plain_trial_accuracy"
            ]
            for r in summary_rows
        ],
        dtype=np.float64,
    )

    fact = np.asarray(
        [
            r[
                "factorized_trial_accuracy"
            ]
            for r in summary_rows
        ],
        dtype=np.float64,
    )

    row_acc = np.asarray(
        [
            r[
                "row_trial_accuracy_occupied"
            ]
            for r in summary_rows
        ],
        dtype=np.float64,
    )

    col_acc = np.asarray(
        [
            r[
                "column_trial_accuracy_occupied"
            ]
            for r in summary_rows
        ],
        dtype=np.float64,
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FACTORIZED GRID RF SUMMARY"
    )

    print(
        "=" * 78
    )

    for r in summary_rows:
        print(
            f"{r['train_sessions']:5s}"
            f" -> {r['test_session']} | "
            f"plain={r['plain_trial_accuracy']:.3f} | "
            f"factorized={r['factorized_trial_accuracy']:.3f} | "
            f"row={r['row_trial_accuracy_occupied']:.3f} | "
            f"col={r['column_trial_accuracy_occupied']:.3f} | "
            f"adj={r['adjacent_accuracy_occupied']:.3f}"
        )

    print(
        f"\nMean plain exact      : "
        f"{plain.mean():.3f}"
    )

    print(
        f"Mean factorized exact : "
        f"{fact.mean():.3f}"
    )

    print(
        f"Delta                 : "
        f"{fact.mean() - plain.mean():+.3f}"
    )

    print(
        f"Mean row accuracy     : "
        f"{row_acc.mean():.3f}"
    )

    print(
        f"Mean column accuracy  : "
        f"{col_acc.mean():.3f}"
    )

    # ---------------------------------------------------------------
    # Final all-development-data factorized model
    # ---------------------------------------------------------------

    print(
        "\nTraining final "
        "S4+S5+S6 factorized model..."
    )

    final_rows = (
        rows_by_session["S4"]
        + rows_by_session["S5"]
        + rows_by_session["S6"]
    )

    final_bundle = fit_factorized(
        final_rows,
        feature_names,
        args.seed + 77777,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "factorized_grid_rf_s4_s5_s6.joblib"
    )

    joblib.dump(
        {
            **final_bundle,
            "feature_names": feature_names,
            "occupied_classes": list(
                OCCUPIED_CLASSES
            ),
            "row_classes": list(
                ROW_CLASSES
            ),
            "col_classes": list(
                COL_CLASSES
            ),
            "zone_to_row": ZONE_TO_ROW,
            "zone_to_col": ZONE_TO_COL,
            "zone_centers": ZONE_CENTERS,
            "empty_threshold": (
                args.empty_threshold
            ),
            "train_sessions": list(
                SESSIONS
            ),
            "method": (
                "factorized_3x3_grid_rf"
            ),
        },
        model_path,
    )

    write_csv(
        out_dir
        / "loso_metrics.csv",
        summary_rows,
    )

    write_csv(
        out_dir
        / "trial_predictions.csv",
        all_trial_rows,
    )

    write_csv(
        out_dir
        / "per_zone_comparison.csv",
        per_zone_rows,
    )

    summary_json = {
        "mean_plain_trial_accuracy": float(
            plain.mean()
        ),
        "mean_factorized_trial_accuracy": float(
            fact.mean()
        ),
        "delta_factorized_minus_plain": float(
            fact.mean()
            - plain.mean()
        ),
        "mean_row_accuracy_occupied": float(
            row_acc.mean()
        ),
        "mean_column_accuracy_occupied": float(
            col_acc.mean()
        ),
        "splits": summary_rows,
        "final_model": str(
            model_path
        ),
        "decision_rule": (
            "Compare factorized exact accuracy "
            "and Z1/Z4/Z7 per-zone accuracy "
            "against the same-split plain RF. "
            "Do not collect new data based on "
            "this experiment."
        ),
    }

    with (
        out_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_json,
            f,
            indent=2,
        )

    print(
        f"\nSaved results: "
        f"{out_dir}"
    )

    print(
        f"Saved final model: "
        f"{model_path}"
    )

    print(
        "\nSTOP after this result. "
        "Do not collect another session "
        "for this experiment."
    )


if __name__ == "__main__":
    main()
