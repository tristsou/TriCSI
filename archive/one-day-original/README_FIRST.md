# TriCSI One-Day Code Kit: S9 audit + frozen three-link CSI/RSSI comparisons

**What this is:** A standalone, reproducible research pipeline that reads a *common ESP32 CSI Toolkit CSV format*, constructs an approximation of the engineered three-link features in your manuscript, trains six matched Extra Trees variants on the two **development sessions only**, freezes them, and evaluates complete later sessions on the **same physical trials**. It also provides a read-only QC audit for S8/S9.

**Critical limitation:** The code that originally achieved **70% versus 84% on 3TX_S3**, your hardware MAC mappings, and raw CSI files were not uploaded. This is an **independent reimplementation**, not an exact reproduction of the existing project. It **cannot claim to be your original frozen model**. The defaults for the 52 subcarrier positions and 7 window starts are plausible reconstructions from the paper, **not verified against your original extractor**. There may also be different CSV headers, conditioning, packet filtering or feature order. Match the original feature extraction code before comparing new accuracy values to 70%/84% or claiming an identical model.

## Contents

- `00_project_inventory.py`: read-only local finder for collector, inference code and saved models.
- `00_discover_and_audit.py`: inspect raw files, MAC counts, channel, RSSI, CSI length, packet counts and timestamps; works for old S8/S9 too.
- `01_build_features.py`: capture 7 windows × 64 packets per link, compute 52 CSI shape × 3 summaries plus 3 gain and 3 RSSI summaries per link (162/link, 486 total).
- `02_train_freeze.py`: fit **six** Extra Trees combinations × 3 seeds (2026–2028) on 3TX_S1+3TX_S2 only; record model hashes. Never overwrite.
- `03_evaluate_frozen.py`: load saved models, verify checksum, average 7 window probabilities to whole-trial predictions on exactly held-out session(s).
- `04_analyze_and_plot.py`: matching two-/three-link comparisons; per-session accuracy, corrected and harmed trials; CSI vs RSSI effect; figures and paired bootstraps.
- `RUN_WINDOWS.ps1`: step-by-step PowerShell instructions, no destructive automatic actions.
- `config.example.json`: exact transmitter MACs and raw data directory must be edited.
- `smoke_test.py`: validates parser, feature dimensions, train/evaluate and reporting on **SYNTHETIC TEST DATA ONLY**.

## Before doing anything

1. Confirm your planned new human-participant collection is covered by appropriate approval; if not, do not collect new participant data until it is authorized.
2. Keep all old raw trial CSVs and original S8/S9 and 3TX_S3 model/prediction outputs unchanged. **Do not rename, overwrite or delete the S9 failure.**
3. In your Windows project: `C:\ESP32-CSI-Tool\tricsi_localization`, extract this folder, preserving the name `TriCSI_One_Day_Code`.
4. Run the read-only inventory and QC commands first. The latter prints the actual MAC strings.
5. Create a copy of `config.example.json` named `config.json`. Change `data_root` to your folder that contains subfolders `3TX_S1`, `3TX_S2`, `3TX_S3` (and later S4/S5). Replace all FILL placeholders with verified MACs for BACK_LEFT/MIDDLE_BACK/BACK_RIGHT. If MACs changed by session, add `macs_by_session`, e.g. `"3TX_S1":{"BACK_LEFT":"...",...}`.
6. Compare `populated_indices`, `packet_csi_offset`, packet filtering, and 7-window selection against your **original** three-link feature extractor. This is important: if the historical S3 score is not reproduced, your new implementation is only a separate method, not a replication of your old frozen models.
7. Verify all three MACs appear on each trial (roughly 33.6 packets/sec/link was measured in the paper). If MAC labels cannot be matched with physical module identifiers, stop; RSSI is not a reliable basis for guessing physical assignment.

## Windows PowerShell (copy/paste; paths shown are examples)

```powershell
cd C:\ESP32-CSI-Tool\tricsi_localization
$K = '.\TriCSI_One_Day_Code'
$Py = '.\.venv\Scripts\python.exe'
& $Py -m pip install -r "$K\requirements.txt"
Copy-Item "$K\config.example.json" "$K\config.json"   # then edit config.json
& $Py "$K\00_project_inventory.py" --root . --output "$K\project_inventory.txt"
& $Py "$K\00_discover_and_audit.py" --root .\data --sessions 3TX_S1 3TX_S2 3TX_S3 --output "$K\three_link_qc.csv"
```

**S9 forensic check** (replace `--root` with directory containing `S8` and `S9`):

```powershell
& $Py "$K\00_discover_and_audit.py" --root .\data --sessions S8 S9 --output "$K\S8_S9_qc.csv"
```

Audit is about raw-file defects and configuration shifts, NOT inferring that S9 was "bugged" simply from poor accuracy. Missing/short recordings, source-MAC swaps, CSI length/channel/mode changes, and malformed EMPTY recordings are possible things to look for. Saved model inference reproducibility must still be tested with your original evaluation code.

**Build development features + freeze BEFORE collecting new S4/S5:**

```powershell
& $Py "$K\01_build_features.py" --config "$K\config.json" --sessions 3TX_S1 3TX_S2 --output "$K\dev_features.npz"
& $Py "$K\02_train_freeze.py" --features "$K\dev_features.npz" --out "$K\frozen_models"
```

**Reconstruct S3 as a pipeline parity check (do NOT adjust parameters on S3):**

```powershell
& $Py "$K\01_build_features.py" --config "$K\config.json" --sessions 3TX_S3 --output "$K\S3_features.npz"
& $Py "$K\03_evaluate_frozen.py" --models "$K\frozen_models" --features "$K\S3_features.npz" --sessions 3TX_S3 --out "$K\S3_reconstructed_predictions.csv"
& $Py "$K\04_analyze_and_plot.py" --predictions "$K\S3_reconstructed_predictions.csv" --out "$K\S3_parity_check"
```

Original paper's full-feature S3 accuracies were 70% for the outer pair and 84% for all three. This **newly fitted** pipeline may not reproduce them. If it does not, you may still report a **separate** controlled experiment using this implementation, but cannot treat its models as your original frozen S3 models. Do not tune to match S3 then count S3 as prospective validation.

**Next, if approved, collect 3TX_S4 and S5 using your ORIGINAL working collection code:**

- Each: 5 EMPTY, 5 trials per zone Z1–Z9, one trial per recorded orientation N, E, S, W, N, 50 trials total. Use 12-second initial EMPTY, 8-second ordinary trials, identical setup and settling periods.
- Keep all three transmitters in place and record packets from all three MACs simultaneously for every trial.
- Treat S4 and S5 as distinct same-day recording sessions: start/stop collector and log any board reboot. Do not introduce a new chair or move furniture intentionally. No photos are needed. A written setup log suffices.
- Do not change preprocessing, feature indices, tuning, seeds or models after seeing new-session performance.

**Evaluate and analyze new sessions:**

```powershell
& $Py "$K\01_build_features.py" --config "$K\config.json" --sessions 3TX_S4 3TX_S5 --output "$K\new_features.npz"
& $Py "$K\03_evaluate_frozen.py" --models "$K\frozen_models" --features "$K\new_features.npz" --sessions 3TX_S4 3TX_S5 --out "$K\S4_S5_predictions.csv"
& $Py "$K\04_analyze_and_plot.py" --predictions "$K\S4_S5_predictions.csv" --out "$K\new_results"
```

Outputs: `new_results/per_session.csv`, `new_results/feature_mechanism_contrast.csv`, `new_results/paired_trial_results.csv`, `new_results/paired_trial_bootstrap.csv`, `new_results/fig1_full_accuracy.png`, `new_results/fig2_feature_link_gains.png`, `new_results/SUMMARY.md`.

## Important validity rules

- The six model variants are `2tx_outer` / `3tx_all` crossed with `full` / `csi` / `rssi` features. "CSI-only" retains CSI shape and packet gain summaries, not RSSI. Every link has 162 source features before feature selection.
- Model seed **2026** is your primary seed; 2027 and 2028 are computational sensitivity checks, not independent recording sessions. Report the original S3 result separately from newly fitted model checks.
- `03_evaluate_frozen.py` does not call `.fit()`; `02_train_freeze.py` refuses non-development sessions and existing output directories. If you change the config, it refuses evaluation against mismatched model configuration fingerprint.
- The held-out S4/S5 results are valid for new models only if all model decisions were fixed before evaluating S4/S5. Do not train on any S3/S4/S5 data.
- Never hide S9. If a defect is confirmed, document original result, objective defect and corrected reanalysis, plus independent validation. S9 is not replaceable merely because its score is low.
- No continuous tracking, gait analysis, fall detection, generalization across people/rooms, or centimeter-level positioning is established by these tests.
- If source files do not parse correctly, **do not modify the science blindly**. Share one representative raw trial CSV and your original `3TX_S3` evaluator + feature extractor. We can make an exact adapter then.
- STS requires independent authorship of the student's scientific report. Check and comply with current STS rules concerning permitted coding support, disclosure, and data provenance.

## Exact original model integration (strongly preferred)

If your existing frozen Extra Trees `.joblib` files **and** original feature extractor remain available, reuse the original inference script instead of substituting the models from this kit. The following read-only commands locate likely artifacts:

```powershell
Get-ChildItem -Recurse -File -Include *.joblib,*.pkl,*.pickle | Select-Object FullName
Get-ChildItem -Recurse -Filter *.py | Select-String -Pattern '3TX_S3|ExtraTreesClassifier|predict_proba' | Select-Object -First 35 Path,LineNumber,Line
```

Do not present this approximate reconstruction as identical to the previous 70/84 pipeline until input feature order, bin indices, window positions and actual saved model artifacts have been verified.
