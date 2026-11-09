from __future__ import annotations

"""
TriCSI-Anchor fixed ablation study on DEVELOPMENT sessions S4-S7 only.

This script does NOT use S8, S9, or S10 to choose a method.

It compares three fixed variants:

FULL
    Empty-room background subtraction
    + cross-session AND cross-orientation spatial metric loss.

NO_BACKGROUND
    Same neural architecture/loss, but no empty-room background subtraction.

NO_ORIENTATION_INVARIANCE
    Empty-room background subtraction
    + spatial metric loss that aligns sessions
    + removes the extra orientation-invariance weighting.

Protocol:
    Leave one entire session out among S4/S5/S6/S7.
    Train on the other 3.
    Use only EMPTY trial01 of the held-out session for background calibration.
    Score all occupied trials and EMPTY trials02+.
    Use SOURCE prototypes only (zero labeled target-zone calibration).

This is an ablation, not a hyperparameter sweep.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from project_config import (
    CLASS_ORDER,
    RAW_DATA_DIR,
    RESULTS_DIR,
)

import train_tricsi_anchor_final as anchor


SESSIONS = ("S4", "S5", "S6", "S7")
LABELS = list(CLASS_ORDER)
OCCUPIED = [
    z
    for z in LABELS
    if z != "EMPTY"
]


def cross_session_only_spatial_loss(
    emb,
    labels,
    sessions,
    orientations,
):
    """
    Same geometry-aware target as the full loss,
    but only cross-session same-zone pairs get extra weight.
    Orientation does not receive special weighting.
    """
    n = emb.shape[0]

    if n < 2:
        return emb.new_tensor(
            0.0
        )

    sim = emb @ emb.T
    sim01 = (
        sim + 1.0
    ) * 0.5

    d = torch.from_numpy(
        anchor.ZONE_DIST
    ).to(
        emb.device
    )

    pair_d = d[
        labels[:, None],
        labels[None, :],
    ]

    target = torch.exp(
        -0.5
        * (
            pair_d / 0.35
        )
        ** 2
    )

    eye = torch.eye(
        n,
        dtype=torch.bool,
        device=emb.device,
    )

    valid = ~eye

    same_zone = (
        labels[:, None]
        == labels[None, :]
    )

    cross_session = (
        sessions[:, None]
        != sessions[None, :]
    )

    weights = torch.ones_like(
        target
    )

    weights = torch.where(
        same_zone
        & cross_session,
        torch.full_like(
            weights,
            3.0,
        ),
        weights,
    )

    err = (
        sim01 - target
    ) ** 2

    return (
        torch.sum(
            err[valid]
            * weights[valid]
        )
        / torch.sum(
            weights[valid]
        )
    )


def train_variant(
    source_raw,
    *,
    variant,
    epochs,
    batch_size,
    lr,
    seed,
):
    anchor.set_seed(
        seed
    )

    norm = anchor.compute_norm(
        source_raw
    )

    model = anchor.TriCSIAdaptNet(
        n_classes=len(
            LABELS
        ),
        embed_dim=64,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    loader = anchor.make_loader(
        source_raw,
        norm,
        batch_size,
        augment=True,
        shuffle=True,
    )

    model.train()

    for epoch in range(
        1,
        epochs + 1,
    ):
        total = 0.0
        n_total = 0

        for (
            x,
            rssi,
            y,
            occ,
            xy,
            session,
            orientation,
        ) in loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            out = model(
                x,
                rssi,
                link_dropout=True,
            )

            ce = F.cross_entropy(
                out[
                    "zone_logits"
                ],
                y,
            )

            occ_loss = (
                F.binary_cross_entropy_with_logits(
                    out[
                        "occ_logit"
                    ],
                    occ,
                )
            )

            occupied_mask = (
                occ > 0.5
            )

            if torch.any(
                occupied_mask
            ):
                xy_loss = (
                    F.smooth_l1_loss(
                        out[
                            "xy"
                        ][
                            occupied_mask
                        ],
                        xy[
                            occupied_mask
                        ],
                    )
                )
            else:
                xy_loss = ce.new_tensor(
                    0.0
                )

            if (
                variant
                == "NO_ORIENTATION_INVARIANCE"
            ):
                spatial = (
                    cross_session_only_spatial_loss(
                        out[
                            "embedding"
                        ],
                        y,
                        session,
                        orientation,
                    )
                )
            else:
                spatial = (
                    anchor.spatial_similarity_loss(
                        out[
                            "embedding"
                        ],
                        y,
                        session,
                        orientation,
                    )
                )

            loss = (
                ce
                + 0.25
                * occ_loss
                + 0.40
                * xy_loss
                + 0.55
                * spatial
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            total += float(
                loss.item()
            ) * len(y)

            n_total += len(y)

        if (
            epoch == 1
            or epoch % 20 == 0
            or epoch == epochs
        ):
            print(
                f"  epoch {epoch:03d}/{epochs}: "
                f"loss={total/max(n_total,1):.4f}"
            )

    return (
        model.eval(),
        norm,
    )


def prepare_source(
    raw,
    variant,
):
    if (
        variant
        == "NO_BACKGROUND"
    ):
        return raw

    return anchor.prepare_source_session(
        raw
    )


def prepare_target(
    raw,
    variant,
):
    if (
        variant
        == "NO_BACKGROUND"
    ):
        return raw

    return anchor.prepare_target_session(
        raw
    )


def trial_group(
    raw,
    i,
):
    return (
        f"{raw.session[i]}|"
        f"{raw.y_text[i]}|"
        f"{int(raw.trial_number[i])}|"
        f"{raw.orientation[i]}"
    )


def evaluate_zero_shot(
    model,
    norm,
    source,
    target,
):
    source_embed = anchor.embed_raw(
        model,
        source,
        norm,
    )

    target_embed = anchor.embed_raw(
        model,
        target,
        norm,
    )

    source_occ = (
        source.y_text
        != "EMPTY"
    )

    prototypes = anchor.make_prototypes(
        source_embed[
            "embedding"
        ][
            source_occ
        ],
        source.y_text[
            source_occ
        ],
    )

    # Keep held-out EMPTY trial01 out of scoring because FULL uses it
    # as unlabeled session background calibration.
    test_mask = (
        (
            target.y_text
            != "EMPTY"
        )
        |
        (
            (
                target.y_text
                == "EMPTY"
            )
            & (
                target.trial_number
                >= 2
            )
        )
    )

    idx = np.where(
        test_mask
    )[0]

    probs = anchor.prototype_probabilities(
        target_embed[
            "embedding"
        ][
            idx
        ],
        prototypes,
    )

    occ_prob = target_embed[
        "occ_prob"
    ][
        idx
    ]

    groups = np.asarray(
        [
            trial_group(
                target,
                int(i),
            )
            for i in idx
        ],
        dtype=object,
    )

    y_true = []
    y_pred = []
    adjacent_hits = []

    for group in sorted(
        set(
            groups
        )
    ):
        local = np.where(
            groups == group
        )[0]

        global_i = idx[
            local[0]
        ]

        true_zone = str(
            target.y_text[
                global_i
            ]
        )

        mean_occ = float(
            np.mean(
                occ_prob[
                    local
                ]
            )
        )

        mean_prob = np.mean(
            probs[
                local
            ],
            axis=0,
        )

        if mean_occ < 0.5:
            pred = "EMPTY"
        else:
            pred = OCCUPIED[
                int(
                    np.argmax(
                        mean_prob
                    )
                )
            ]

        y_true.append(
            true_zone
        )
        y_pred.append(
            pred
        )

        if (
            true_zone
            != "EMPTY"
        ):
            adjacent_hits.append(
                int(
                    pred
                    != "EMPTY"
                    and anchor.adjacent(
                        true_zone,
                        pred,
                    )
                )
            )

    y_true = np.asarray(
        y_true,
        dtype=object,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=object,
    )

    return {
        "n_trials": int(
            len(
                y_true
            )
        ),
        "exact_accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "occupied_exact_or_adjacent": float(
            np.mean(
                adjacent_hits
            )
        )
        if adjacent_hits
        else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=70,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22082026,
    )

    args = parser.parse_args()

    variants = (
        "FULL",
        "NO_BACKGROUND",
        "NO_ORIENTATION_INVARIANCE",
    )

    print("\n" + "=" * 78)
    print("TriCSI-Anchor DEVELOPMENT ABLATION")
    print("=" * 78)
    print("Sessions: S4/S5/S6/S7 only.")
    print("S8/S9/S10 are NOT used.")
    print("This is a fixed ablation, not a tuning sweep.")
    print("=" * 78)

    raw_map = {
        s: anchor.load_session(
            args.raw_dir,
            s,
        )
        for s in SESSIONS
    }

    rows = []

    for variant_i, variant in enumerate(
        variants
    ):
        print("\n" + "#" * 78)
        print(f"VARIANT: {variant}")
        print("#" * 78)

        for fold_i, test_session in enumerate(
            SESSIONS
        ):
            train_sessions = [
                s
                for s in SESSIONS
                if s != test_session
            ]

            print(
                f"\n{variant}: "
                f"{'+'.join(train_sessions)} -> {test_session}"
            )

            source = anchor.combine(
                [
                    prepare_source(
                        raw_map[s],
                        variant,
                    )
                    for s in train_sessions
                ]
            )

            target = prepare_target(
                raw_map[
                    test_session
                ],
                variant,
            )

            model, norm = train_variant(
                source,
                variant=variant,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=(
                    args.seed
                    + variant_i
                    * 10000
                    + fold_i
                    * 1000
                ),
            )

            metrics = evaluate_zero_shot(
                model,
                norm,
                source,
                target,
            )

            row = {
                "variant": variant,
                "train_sessions": "+".join(
                    train_sessions
                ),
                "test_session": test_session,
                **metrics,
            }

            rows.append(
                row
            )

            print(
                f"  exact={metrics['exact_accuracy']:.3f} | "
                f"macroF1={metrics['macro_f1']:.3f} | "
                f"adj={metrics['occupied_exact_or_adjacent']:.3f}"
            )

    print("\n" + "=" * 78)
    print("ABLATION SUMMARY")
    print("=" * 78)

    summary = {}

    for variant in variants:
        vrows = [
            r
            for r in rows
            if r[
                "variant"
            ]
            == variant
        ]

        exact = [
            r[
                "exact_accuracy"
            ]
            for r in vrows
        ]

        macro = [
            r[
                "macro_f1"
            ]
            for r in vrows
        ]

        adjacent = [
            r[
                "occupied_exact_or_adjacent"
            ]
            for r in vrows
        ]

        summary[
            variant
        ] = {
            "mean_exact_accuracy": float(
                np.mean(
                    exact
                )
            ),
            "mean_macro_f1": float(
                np.mean(
                    macro
                )
            ),
            "mean_adjacent": float(
                np.mean(
                    adjacent
                )
            ),
        }

        print(
            f"{variant:27s} "
            f"exact={summary[variant]['mean_exact_accuracy']:.3f} | "
            f"macroF1={summary[variant]['mean_macro_f1']:.3f} | "
            f"adj={summary[variant]['mean_adjacent']:.3f}"
        )

    out_dir = (
        RESULTS_DIR
        / "tricsi_anchor_ablation"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        out_dir
        / "ablation_folds.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    with (
        out_dir
        / "ablation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"\nSaved ablation results under:\n{out_dir}"
    )


if __name__ == "__main__":
    main()
