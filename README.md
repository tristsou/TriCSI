# TriCSI

**Low-cost, device-free indoor localization with ESP32-S3 Wi-Fi channel state
information (CSI).** TriCSI classifies a stationary person into nine zones and
distinguishes occupied trials from an empty room. It contains a two-link neural
model, a three-link feature/classifier study, and cross-session experiments.

## Start here

1. Read the [overview](docs/OVERVIEW.md) for the scientific question and workflow.
2. Follow [setup](docs/SETUP.md); run `python -m tricsi doctor --code-only`.
3. Check [data availability](docs/DATA.md). **The owner's data/model bundle has
   been received and verified, but its GitHub release upload is pending. A
   code-only clone still cannot run the empirical evaluations.**
4. Follow [reproduction](docs/REPRODUCE.md) for exact commands and split rules.
5. Read [contributing](CONTRIBUTING.md) before changing a model or protocol.

## Reported measurements

| Experiment | Sources | Evaluation | Accuracy | Evidence here |
| --- | --- | --- | --- | --- |
| Original frozen two-link model | S4–S7; target Empty calibration | Historical S8 / S9 | 95% / 52.5% | Re-evaluated uploaded checkpoint; all 80 trial labels match |
| Original frozen three-link study | 3TX_S1/S2 | Historical 3TX_S3 | Three links 84%; outer pair 70% | Re-evaluated uploaded models; all 100 compared trial labels match |
| Script-default reconstruction | S4–S7; original trainer, 70 epochs | S8 / S9 | 87.5% / 55% | Owner-supplied terminal report |
| Original shape, three seeds | S4–S7; checkpoints selected on S8/S9 | Development S8 / S9 | Mean 79.17% / 65.83% | Parsed owner-supplied log |
| Empty-relative shape, three seeds | Same data and matched budget | Development S8 / S9 | Mean 72.50% / 54.17% | Parsed owner-supplied log |

The original fitted models were re-evaluated without retraining. This confirms
saved-model reproduction, not new independent recording sessions. Probability
values may differ slightly across platforms while predicted labels match.
The shape change **did not improve accuracy**. New checkpoint-selected S8/S9
scores are development findings, not independent test evidence. See the
[results ledger](reports/README.md) for provenance and limitations. Synthetic
software tests check implementation behavior; they do not establish RF accuracy.

## One command interface

Run from the checkout root with the project's Python environment:

```powershell
python -m tricsi --help
python -m tricsi inventory --raw-dir data/raw --output results/inventory_new
python -m tricsi run configs/background_shape_s4_s7.json --dry-run
```

[Recipes](configs/README.md) record complete argument lists. `--dry-run` prints
the recipe without training. Original `python -m experiments...` commands remain
valid. Use new output directories for reruns.

## Repository map

| Location | Purpose |
| --- | --- |
| [tricsi/](tricsi/) | Supported CLI, asset checks, bundle creation and verification |
| [configs/](configs/README.md) | Fixed source lists, seeds and scoring exclusions |
| [experiments/](experiments/) | Neural training, generator, shape comparison and diagnostics |
| [tricsi_3tx_modeling_bundle/](tricsi_3tx_modeling_bundle/README_3TX_MODELING.md) | Three-link extraction, subset training and frozen evaluation |
| [docs/](docs/README.md) | Setup, architecture, data dictionary and reproduction |
| [metadata/](metadata/README.md) | Session roles, asset status, engine hashes and migration map |
| [reports/](reports/README.md) | Small result summaries with evidence labels |
| [workflows/](workflows/) | Descriptive wrappers for preserved entry points |
| [legacy/](legacy/README.md), [archive/](archive/README.md) | Historical utilities and recovered packages |
| [tests/](tests/) | Interface, recipe, archive integrity and preservation checks |

Nine compatibility/engine Python files remain at the root because historical
scripts and saved audit records refer to their paths. See
[architecture](docs/ARCHITECTURE.md); new development belongs in modules above.

## Data release status

`TriCSI-reproduction-v1.zip` has 2,018 manifest files and is 246,014,376 bytes.
Every file hash passed verification. See the [full manifest](metadata/reproduction_bundle_manifest.json),
[owner environment](metadata/owner_environment.json) and
[reproduction receipts](reports/verified/). The ZIP is not yet attached to a
GitHub release; [assets.json](metadata/assets.json) deliberately retains a null
download URL until upload succeeds.

The owner can package the original working project, including raw measurements,
exact local code, caches, models, run settings, predictions and environment:

```powershell
python -m tricsi bundle --project-root C:\ESP32-CSI-Tool\tricsi_localization --output C:\Users\ch_ts\TriCSI-reproduction-v1.zip
python -m tricsi verify-bundle C:\Users\ch_ts\TriCSI-reproduction-v1.zip
```

Run from this reorganized checkout using the Python environment that runs TriCSI.
The tool reads the original project, writes outside it, preserves source/data
bytes, skips virtual environments and redundant ZIPs, and verifies included file
hashes. It does not publish anything. Follow [publishing](docs/PUBLISHING.md) to
upload the asset and update the [catalog](metadata/assets.json).

## Scope and availability

This is not a validated room simulator or a solution proven robust across rooms.
Full ESP32 firmware/build settings remain missing. A current Windows environment
lock was received; its identity as the environment for every historical training
run is not proven. Three-link source uses placeholder transmitter MACs; use the
documented runtime link map for real recordings. The owner has not selected a
code or dataset license; see [availability](docs/AVAILABILITY.md). Implementation
and documentation include AI-assisted work requiring researcher review.
