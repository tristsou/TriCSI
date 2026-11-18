# Reproduce a result

## Choose the claim first

| Goal | Correct action |
| --- | --- |
| Re-evaluate the original 95% / 52.5% model | Load the original saved checkpoint and rebuild source prototypes; do not retrain a replacement |
| Re-evaluate the original 84% three-link model | Load the original frozen `.joblib` with its original features and link mapping |
| Reconstruct historical training | Use the original trainer with recorded settings; script defaults alone do not prove historical identity |
| Compare representation or augmentation changes | Train matched conditions, report all seeds/sessions, then freeze before fresh-session testing |

Install [dependencies](SETUP.md), obtain [assets](DATA.md), verify their manifest,
and run `python -m tricsi doctor`. All examples run from the checkout root.
Recipes are plain JSON with a command and argument list. Use `--dry-run` to
review. If an output already exists, copy/edit a recipe to a new output path.

## 1. Original frozen two-link result

```powershell
python -m tricsi run configs/frozen_two_link.json
```

Requires raw S4–S9 and the exact `models/tricsi_anchor_s4_s5_s6_s7.pt`. Sources
S4–S7 construct prototypes with the frozen encoder; target information is Empty
trial 1 only. S8 occupied trials 1–2 and both sessions' Empty trial 1 are excluded
from scoring. Expect 40 scored trials per session with the known recordings.
Original reported values are 38/40 and 21/40; compare predictions and hashes, not
only rounded accuracy. The original environment/checkpoint identity is still
pending publication, so these values have not been independently reproduced here.

The supported adapter writes under `results/reproduced_two_link/` and refuses an
existing evaluation directory. For another run:

```powershell
python -m tricsi run configs/frozen_two_link.json --results-root results/reproduced_two_link_v2
```

## 2. Script-default reconstruction

Prepare caches only if absent:

```powershell
python -m tricsi run configs/prepare_two_link.json
python -m tricsi run configs/historical_s4_s7.json
```

An existing cache is not silently replaced. Add only a missing S7 cache with
`python -m tricsi prepare --raw-dir data/raw --sessions S7 --cache-dir outputs/session_cache_v1`.
The historical recipe uses 70 epochs, seed 26082026, batch 128 and LR 0.001 from
the original script defaults. All source Empty trials establish the background;
Empty trial 1 remains eligible for source training. The original shuffled loader
drops incomplete final batches. Only the final epoch is scored.

The owner's reported reconstruction is 35/40 S8 and 22/40 S9. Actual historical
training settings are unverified; this is not an exact-weights reproduction of
the original model and never overwrites that checkpoint.

## 3. Matched shape comparison

```powershell
python -m tricsi run configs/background_shape_s4_s7.json
```

Uses S4–S7 sources, S8/S9 development, seeds 2026/2027/2028, 150 epochs and
evaluation every five epochs. Both representations use identical training data,
architecture, losses, existing augmentation and update budget. Unlike the
historical trainer, source Empty trial 1 is reserved for calibration. The shape
channels are the only intervention between the two comparison arms.
One checkpoint per arm/seed is selected jointly by worst-session accuracy, mean
accuracy, then macro-F1. Every arm completes the same maximum epoch budget.

Inspect `comparison_metrics.csv`, `comparison_predictions.csv`,
`learning_curves.csv`, `run_config.json` and the checkpoint index. Report both
sessions and all seeds. The negative result is in the [ledger](../reports/README.md).
These selected scores are development findings, not independent test evidence.

## 4. Three-link historical models

Set the actual recorded MAC mapping as explained in [data](DATA.md).

```powershell
python -m tricsi run configs/frozen_three_link.json --link-map metadata/local_link_map.json
```

The original bundle must include its fitted subset models, feature configuration,
training sessions/data hashes and primary subset. It evaluates historical 3TX_S3
without fitting. A parser/version/mapping mismatch is a reproducibility problem
to inspect, not a reason to retrain silently.

To rerun development separately:

```powershell
python -m tricsi run configs/three_link_development.json --link-map metadata/local_link_map.json
```

This uses script defaults (400 trees, seed 2026, 64 packets, seven windows).
The historical freeze selected per-subset seeds in the code; inspect the actual
bundle instead of assuming that one CLI seed identifies every historical model.
Do not compare a refitted model against the original frozen claim as if unchanged.

## 5. Augmentation and new sessions

See the complete [session-robustness guide](../experiments/session_robustness/README.md).
The supported CLI exposes `develop`, `freeze` and `evaluate` with the same options:

```powershell
python -m tricsi develop --help
python -m tricsi freeze --help
python -m tricsi evaluate --help
```

Use all compatible source data. Fit the generator and normalization on source
data only; compare with existing and generic augmentation under matched budgets.
New final evaluation sessions must be disjoint from training and development.
The current freezer handles augmentation development runs, not the separate
shape-comparison output format. Do not substitute one format for the other.

## Reproducibility receipt

For every reported result retain the exact code/data/checkpoint hashes, Python
and package versions, grid/device mapping, source/development/evaluation session
lists, seed, epoch/selection rule, calibration and exclusions, whole-trial
predictions and completion receipt. Preserve the original artifact even when
retraining is possible. Cross-platform retraining may not produce identical
floating-point weights; reference predictions identify the claimed fitted model.
