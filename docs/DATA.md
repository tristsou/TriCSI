# Data dictionary and session roles

## Availability and paths

The owner supplied `TriCSI-reproduction-v1.zip`: 2,018 files, 246,014,376 bytes,
with all per-file checksums verified. Raw data, models, caches and outputs are in
the [published release bundle](https://github.com/tristsou/TriCSI/releases/tag/reproducibility-v1). See
[asset status](../metadata/assets.json) and the
[complete manifest](../metadata/reproduction_bundle_manifest.json).
The original two-link and primary/outer three-link fitted models were re-evaluated
without fitting; all 180 compared trial labels match the stored predictions.

| Path | Contents |
| --- | --- |
| `data/raw/S4` … `data/raw/S9` | Two-link whole-trial CSVs |
| `data_3tx/raw/3TX_S1` … `3TX_S3` | Three-link CSVs and collection manifests |
| `models/tricsi_anchor_s4_s5_s6_s7.pt` | Original two-link checkpoint, not a retrained substitute |
| `results/3tx_dev_comparison/frozen_3tx_model.joblib` | Original three-link frozen bundle |
| `outputs/session_cache_v1/*.npz` and `*.json` | Prepared windows and raw-source/code hashes |
| `results/<run>/` | Settings, predictions, curves, checkpoints and completion receipt |

## Two-link CSV fields

| Field | Meaning |
| --- | --- |
| `session` | Session identifier; directory and row metadata must agree |
| `zone` | `EMPTY`, `Z1` … `Z9`; physical meanings must be documented |
| `trial` | Integer trial number within a zone |
| `orientation` | `N`, `E`, `S`, `W`, or collector Empty value, typically `NA` |
| `link` | `left` or `right` |
| `device_time_s` | Device timestamp used in two-link extraction |
| `rssi` | Measured signal strength |
| `iq_000` … `iq_127` | 128 signed values: 64 complex LLTF samples as imag/real pairs |

Preserve the full original header and other collector columns. A `len` metadata
field does not replace inspection of the IQ payload. The audit rejects malformed
rows, missing fields, inconsistent identities and invalid signed IQ ranges.

Two-link windows have shape `[N,4,32,64]`, plus four RSSI statistics. CSI channels
0/2 are log amplitudes and 1/3 packet-centered shape. Windows are 1 s with a
0.25 s hop and at least 12 packets per link. Cache metadata retains labels,
orientations, filenames, trial groups and window indices.

## Three-link differences

The historical parser groups packets by `mac`, requires RSSI and the 128 IQ
fields, and uses host/local timestamps where available. It extracts seven
64-packet windows per trial. Link names are `back_left`, `middle_back`,
`back_right`. The expected protocol is 50 trials: five per occupied zone plus
five Empty trials. Session manifests may provide grid coordinates.

Public source uses placeholder MACs. Copy
[link_map.example.json](../metadata/link_map.example.json) to
`metadata/local_link_map.json`, enter the actual recorded addresses and pass
`--link-map metadata/local_link_map.json` to a three-link command or recipe.
The adapter changes the runtime mapping without rewriting source/data bytes.

## Session rules

| Sessions | Role / scoring |
| --- | --- |
| S1–S3 | Earlier feasibility/transfer work; do not automatically mix into later sources |
| S4–S7 | Original sources; check S7's physical zone compatibility |
| S8 | 59 trial files; exclude occupied trials 1–2 and Empty trial 1; score 40 |
| S9 | 41 trial files; keep all occupied trials, exclude Empty trial 1; score 40 |
| 3TX_S1/S2 | Original three-link development/training |
| 3TX_S3 | Originally independent frozen evaluation; not a fresh test for later tuning |
| Other 3TX sessions | Present in the local backup log; no role/unseen status assigned here |

The historical trainer uses all source Empty trials for its background and keeps
Empty trial 1 in source training. The current matched experiment reserves source
Empty trial 1 and uses it for background calibration. Both use dedicated target
Empty calibration and source-only occupied prototypes.

`project_config.py` contains numerical grid coordinates; old comments are not a
survey record. Confirm grid, antenna/device positions and environmental changes
from collection notes. A CSV label cannot prove where the participant stood.
