# TriCSI three-transmitter comparison and frozen test

This pipeline compares every one-, two-, and three-transmitter subset on
`3TX_S1` and `3TX_S2`, freezes a development-selected primary model plus the
predeclared hardware-ablation models, and later tests those exact models once
on untouched `3TX_S3`.

## Files

- `tricsi_3tx_modeling.py`: shared parsing, features, and metrics
- `compare_and_freeze_3tx.py`: S1/S2 development comparison and model-set freezing
- `evaluate_frozen_3tx.py`: final S3 primary test and predeclared ablations, without retraining

## Why this split is important

The unit of evaluation is a whole trial, and the test set is an entirely
different session. Fixed windows from a trial are never divided between train
and test. This prevents nearly adjacent CSI packets from leaking across the
split and producing an unrealistically high score.

Each link gets exactly seven 64-packet windows per trial. Trial predictions are
made by averaging the seven predicted-probability vectors. The amplitude
features retain frequency-selective CSI shape, within-window variation, common
gain, and RSSI. The final position error includes a room-diagonal penalty if an
occupied trial is incorrectly predicted as `EMPTY`.

## 1. Put these files in the project folder

Place all three Python files beside `collect_three_link_trials.py` in:

```text
C:\ESP32-CSI-Tool\tricsi_localization
```

The expected data folders are:

```text
data_3tx\raw\3TX_S1
data_3tx\raw\3TX_S2
```

## 2. Install dependencies

Activation is not required. In PowerShell:

```powershell
Set-Location C:\ESP32-CSI-Tool\tricsi_localization
.\.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn matplotlib joblib
```

## 3. Compare S1 and S2 and freeze the winner

```powershell
.\.venv\Scripts\python.exe .\compare_and_freeze_3tx.py
```

The script tests both directions: S1 training to S2 test, and S2 training to S1
test. It compares:

- each single transmitter;
- all three two-transmitter pairs;
- all three transmitters together.

`auto` designates the subset with the highest mean trial accuracy, then the
lowest mean position error, then fewer links if still tied, as the primary
model. All seven subset models are frozen before S3 so the one/two/three-link
ablation can also be reported without post-test fitting. To preselect all three
links as the primary model, add `--freeze-subset 3tx_all`.

Main outputs:

```text
results\3tx_dev_comparison\subset_summary.csv
results\3tx_dev_comparison\fold_metrics.csv
results\3tx_dev_comparison\trial_predictions.csv
results\3tx_dev_comparison\paired_outer_vs_3tx.csv
results\3tx_dev_comparison\development_report.json
results\3tx_dev_comparison\subset_comparison.png
results\3tx_dev_comparison\frozen_3tx_model.joblib
```

The paired comparison specifically measures the effect of adding the weak
middle path to the original outer pair. The 95% bootstrap confidence interval
is more informative than a raw accuracy difference alone.

## 4. Preserve the development decision

Before collecting S3, copy the entire `results\3tx_dev_comparison` folder into
your research backup. Do not choose a different feature setting, transmitter
subset, or model after seeing S3.

## 5. Collect untouched 3TX_S3

Keep the finalized hardware geometry unchanged and use a new random seed:

```powershell
.\.venv\Scripts\python.exe .\collect_three_link_trials.py `
  --session 3TX_S3 `
  --seed 311
```

## 6. Run the final test once

```powershell
.\.venv\Scripts\python.exe .\evaluate_frozen_3tx.py
```

This loads `frozen_3tx_model.joblib`, checks that S3 was not a training session,
and writes the primary trial-level result plus the predeclared one/two/three-link
ablation under `results\3tx_s3_final`. It does not retrain, tune, change the
primary subset, or learn fusion weights from S3.
