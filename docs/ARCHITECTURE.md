# Code architecture

| Responsibility | Module |
| --- | --- |
| User-facing command routing | `tricsi/__main__.py` |
| Snapshot and integrity checks | `tricsi/bundle.py` |
| Raw two-link parser | `csi_io.py` and `train_csi_cnn.py` |
| Historical network, losses, normalization and prototypes | `train_tricsi_anchor_final.py` |
| Historical saved-model scoring | `evaluate_frozen_anchor_confirmatory.py` |
| Current cache/calibration contracts | `experiments/session_robustness/data.py` |
| Current matched neural training | `experiments/session_robustness/model.py` |
| Empirical signal perturbations | `experiments/session_robustness/generator.py` |
| Train/select/freeze/evaluate receipts | `experiments/session_robustness/study.py` |
| Shape representation and embedding diagnostics | `experiments/session_diagnostics/core.py` |
| Controlled shape comparison and audit | `experiments/session_diagnostics/__main__.py` |
| Original-trainer reconstruction | `experiments/session_diagnostics/historical.py` |
| Three-link features and models | `tricsi_3tx_modeling_bundle/` |

The root additionally retains `features.py`, `train_crossval.py`,
`run_tricsi_anchor_ablations.py` and `collect_three_link_trials.py` for historical
import/reference compatibility. These nine root Python files are deliberately
not renamed. Protected engine hashes are in
[engine_hashes.json](../metadata/engine_hashes.json).

## Data flow

The two-link audit creates non-pickle NPZ window caches. Source calibration yields
real training windows and separate Empty references. The generator fits those
sources; the network fits source normalization and features. Development scores
select one checkpoint jointly across sessions. Frozen evaluation checks the
receipts and never fits on occupied target recordings.

## Where to improve the project

Add a candidate generator/representation in the experiment modules, a declarative
recipe in `configs`, and a result record in `reports` after validation. Keep a
matched baseline, all seeds and all session outcomes. Never improve a score by
changing test labels, excluding difficult valid trials or fitting target labels.

New code should receive explicit paths, avoid hardcoded workstation directories,
fail on incompatible formats, and preserve originals. Update the code hashes
only for an intentional new version; do not rewrite old run receipts to suppress
a mismatch. Hash-verified old runs must use their recorded engine bytes.

## Migration

[migration.json](../metadata/migration.json) records every moved old path, new
path and unchanged SHA-256. Historical kits and utilities moved into `legacy`;
`workflows/_run_source.py` resolves their original names through this map. Legacy
batch files and old embedded README paths are archival, not supported commands.
`SOURCE_MANIFEST.tsv` records import provenance at the original source-export
stage; it is not a live hash manifest. Use migration/engine/run manifests for
current checks. Original Git history also preserves all earlier layouts.
