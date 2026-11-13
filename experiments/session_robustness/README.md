# TriCSI session-robustness experiment

## The question

Does adding source-informed CSI variations improve real cross-session
localization, especially the weakest session, beyond the augmentation already
in TriCSI-Anchor?

**Every condition uses ALL eligible real source trials. There is no training-data
reduction or real-data-budget sweep in this implementation.** The dedicated
Empty calibration recording is separated from training/scoring in every arm.

This is the first implementation stage: an empirical, source-anchored signal
generator. It is **not** a ray tracer, a digital replica of the room, a physical
human scattering model, or evidence that session robustness has been solved.
No real recordings or experimental results are included in this repository.
Implementation and documentation were prepared with AI assistance and require
student review and experimental validation.

## What stays matched

- The existing two-link TriCSI-Anchor encoders, fusion, occupancy head and losses.
- All eligible real training windows/trials; no random trial subsampling.
- Real-source-only normalization and source-only location prototypes.
- The same model initialization seed, minibatch order, epochs and update counts.
- Network and link dropout settings in every condition.
- The same dedicated Empty trial 1 for per-session background calibration.
- The same evaluation trials and whole-trial aggregation across conditions.

This is a new controlled experiment, not an exact replay of the published frozen
checkpoint: the loader retains the final incomplete batch and the source-session
list is explicit. Do not compare its newly trained score directly with the old
95% as if only augmentation had changed. Compare the matched arms below.

| Method | Input perturbations |
| --- | --- |
| `none` | No input perturbations; network/link dropout is retained |
| `existing` | Original time shift, subcarrier masking and residual noise |
| `generic` | Existing augmentation + independent spectral offsets with matched learned marginal variance, link gains and correlated temporal noise |
| `session` | Existing augmentation + low-rank joint spectral/link offsets, link gains and correlated temporal noise |

The primary practical comparison is `session` versus `existing`. `session`
versus `generic` probes the value of learned joint spectral/link structure.
This first control does not separately isolate session persistence or temporal
correlation; additional ablations are future work.

The generator summarizes each source trial first, then forms differences for
matching zone/orientation conditions across source sessions. A rank-at-most-4
basis approximates those differences. Source Empty recordings estimate temporal
noise. This does not establish that the estimated differences are exclusively
nuisance or that the transformation always preserves zone labels.

One generated style is shared across classes in each training epoch. Half of
trial views receive the extra generated transformation, selected independently
of the class label; other views stay real apart from ordinary augmentation.
The style changes each epoch. Additional views are generated on the fly, so no
millions-of-files export is needed. Matched update counts keep longer training
from explaining the result.

Changes to calibrated absolute CSI channels are also propagated to their
packet-centered channels. RSSI is deliberately held unchanged: this is a
CSI-only intervention, not a fully consistent synthetic hardware packet.
The historical extractor's 64-bin representation is preserved; no new physical
subcarrier interpretation, phase calibration or packet-rate simulation is claimed.

## Before running

Use the existing Windows virtual environment that runs TriCSI-Anchor. The code
expects the original repository modules in the project root. If dependencies
are missing:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\experiments\session_robustness\requirements.txt
```

PyTorch must also be installed in that environment. Use the CPU installation
instructions at https://pytorch.org/get-started/locally/ if needed. The workflow
was software-tested here with Python 3.12 and CPU PyTorch; that does not identify
the Python version used for any historical paper figure.

Raw recordings must be the existing two-link per-trial CSV format: `session`,
`zone`, `trial`, `orientation`, `link`, `device_time_s`, `rssi`, and `iq_000`
through `iq_127`. Other formats are reported rather than guessed or repaired.
`left` and `right` link labels in saved CSVs are used, so a new MAC mapping is
not needed to read those files. Do not collect new data just to run the audit.

**Check zone meanings and source-session compatibility before choosing them.**
The examples use S4-S6 as candidate sources, exclude S7, and treat S8/S9 as
historical development. These are examples, not a finding that those source
sessions have identical geometry. If coordinates or labels changed, document
that separately; do not silently merge incompatible grids. This workflow reports
classification, not continuous positioning error in meters.

## 1. First action: inventory the recordings you already have

From `C:\ESP32-CSI-Tool\tricsi_localization`:

```powershell
.\.venv\Scripts\python.exe -m experiments.session_robustness inventory --raw-dir .\data\raw --output .\results\session_inventory_v1
```

This is read-only with respect to recordings. It writes `inventory.csv` and
`inventory.json`, showing sessions, metadata, supported layouts, row counts,
left/right packets and malformed rows. Share the inventory to choose the first
real run. If your recordings are elsewhere, change `--raw-dir`.

All output directories must be new. To rerun, use a new suffix; existing models,
caches and results are not overwritten.

## 2. Prepare caches without fitting

After reviewing the inventory and source compatibility:

```powershell
.\.venv\Scripts\python.exe -m experiments.session_robustness prepare --raw-dir .\data\raw --sessions S4 S5 S6 S8 S9 --cache-dir .\outputs\session_cache_v1
```

The original extractor resamples 1-second windows to `[4,32,64]` with a
0.25-second hop and at least 12 packets/link/window. Caches include trial groups,
session IDs and file/decoder hashes. The preflight rejects wrong CSV widths,
out-of-range signed IQ, mixed trial identities, absent links, duplicate trials,
or recordings that produce no windows. It does not silently repair historical
parser problems. Audit and resolve those first.

## 3. First development run: check the pipeline on historical data

```powershell
.\.venv\Scripts\python.exe -m experiments.session_robustness develop --cache-dir .\outputs\session_cache_v1 --train S4 S5 S6 --dev S8 S9 --exclude-dev-occupied-trials 1 2 --epochs 10 --seeds 2026 --strengths 1 --output .\results\session_robustness_pilot_v1
```

The explicit occupied-trial exclusion preserves the historical S8/S9 scoring
partition **only if** occupied trials 1 and 2 really were their calibration
recordings. Review the inventory first. These excluded occupied trials are not
used for generator fitting, background estimation or target prototypes.

S8 and S9 have already been inspected. For this extension they are development
benchmarks, not new untouched tests. A new score on them cannot retroactively
establish a new independent test result.

The 10-epoch run is a pipeline check, not an epoch recommendation or final
comparison. After checking the outputs, run a planned matched comparison:

```powershell
.\.venv\Scripts\python.exe -m experiments.session_robustness develop --cache-dir .\outputs\session_cache_v1 --train S4 S5 S6 --dev S8 S9 --exclude-dev-occupied-trials 1 2 --epochs 70 --seeds 2026 2027 2028 --strengths 0.5 1 1.5 --output .\results\session_robustness_development_v1
```

No routine uses occupied evaluation data for fitting. Generator parameters and
normalization are fitted only on source sessions. Development labels select
strength, which is why these sessions are not confirmatory tests. Choose the
epoch budget before this comparison or treat further changes as development.

Outputs include `generator.json`, `run_config.json`, per-trial development
predictions, per-session metrics, checkpoints and a summary. Rank settings by
the mean across seeds of each seed's worst-session accuracy, followed by mean
session accuracy and macro-F1; exact ties prefer lower strength, with stable
method order. The summary averages each session equally.

## 4. Freeze before new real evaluation

```powershell
.\.venv\Scripts\python.exe -m experiments.session_robustness freeze --development-run .\results\session_robustness_development_v1 --output .\models\session_robustness_frozen_v1
```

This copies the best development strength for each condition and all its seed
checkpoints. It does not refit. Code and checkpoint hashes are verified.

## 5. Small fresh evaluation only after freezing

Use the current room as a new documented starting configuration. Photographs
and sensor/grid measurements suffice; reconstructing every historical object
position is not required. Keep a dedicated Empty trial 1 and separate scored
Empty recordings. Use distinguishable new session IDs, for example `R1` and
`R2`, on separate days. This module does not control hardware collection.

After collecting with a compatible collector:

```powershell
.\.venv\Scripts\python.exe -m experiments.session_robustness prepare --raw-dir .\data\raw --sessions R1 R2 --cache-dir .\outputs\session_cache_v1
.\.venv\Scripts\python.exe -m experiments.session_robustness evaluate --bundle .\models\session_robustness_frozen_v1 --cache-dir .\outputs\session_cache_v1 --sessions R1 R2 --output .\results\session_robustness_test_v1
```

Do not use the historical occupied-trial exclusion for fresh sessions unless
their protocol actually reserves those trials. Calibration is **strictly Empty
trial 1**, with no fallback to test Empty recordings. Evaluation performs no
gradient update, generator fitting, target occupied calibration or target-wide
normalization. It rejects session overlap with source/development and changed
model/code hashes. File hashes do not prove that a renamed or edited recording
is truly new: keep the collection record and roles honest.

Report all technically valid test sessions. Changed room conditions are part
of the reported scope, not a reason to remove a low score. A model should not
be called session-robust based on one good session or synthetic evaluation.

Outputs include per-session and worst-session accuracy, macro-F1, occupied
accuracy, Empty recall, orientation summaries, confusion matrices, paired
corrected/harmed counts, exploratory exact McNemar tests and real-evaluation
PNG/SVG figures. Accuracy is based on whole trials. Seeds and overlapping
windows do not increase the count of independent recording days; the within-day
McNemar p-values do not establish broad cross-room generalization. Repeated
comparisons are exploratory, not multiple confirmatory discoveries.

## Software verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\experiments\session_robustness\tests -v
```

Tests use artificial fixtures to check shapes, deterministic generation,
non-mutation, calibration separation, leakage rejection, matched update counts,
and the full neural train/freeze/evaluate path. Their printed accuracies are
**software-test outputs, not TriCSI research results**. Temporary files are
removed after tests.

## Next simulation stage

After a real development run, inspect whether the empirical generator helps,
is too weak, or distorts location cues. A separately identified multipath model
can then be compared against the same `existing` and `generic` controls. It
must use measured/substantiated frequency ordering, hardware assumptions and
source-only parameter fitting. Geometry-based placement optimization requires
additional real validation; it is not implemented by the empirical generator.

## Relevant foundations

- Serbetci, Lee, Burghal & Molisch, *Simple and Effective Augmentation Methods
  for CSI Based Indoor Localization*: https://arxiv.org/abs/2211.10790
- Hou & Wu (2024), *RFBoost*: https://doi.org/10.1145/3659620
- Li et al., *AF-DCGAN*: https://arxiv.org/abs/1804.05347
- Cai, Korany, Karanam & Mostofi (2020), *Teaching RF to Sense without RF
  Training Measurements*: https://doi.org/10.1145/3432224
- Rao et al. (2024), *MFFALoc*: https://doi.org/10.1109/JIOT.2023.3339797

These are related work, not claims that this implementation reproduces those
methods. Augmentation, synthetic RF training and adaptive localization are
established; the new experimental question concerns this generator and its
incremental benefit under matched real cross-session evaluation.
