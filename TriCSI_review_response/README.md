# TriCSI analysis tools after journal review

These tools address specific evidence gaps. They do not promise publication, invent results, or change the already frozen model. Every new Python file in this folder was generated with ChatGPT assistance. The shared tricsi_3tx_modeling.py is copied unchanged from the original AI-assisted modeling bundle. Preserve the prompt history and verify all output yourself.

## Installation

Extract the zip so this folder is directly under:

    C:\ESP32-CSI-Tool\tricsi_localization\TriCSI_review_response

Run commands from the main tricsi_localization folder. Keep your existing scripts, raw data, and results in place. The toolkit does not contain a replacement frozen model.

```powershell
Set-Location C:\ESP32-CSI-Tool\tricsi_localization
.\.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn matplotlib joblib
```

Use your existing working PyTorch installation for neural component runs. Avoid changing scikit-learn versions immediately before loading an existing joblib model; use the environment that originally evaluated it.

## 1 Simple development baselines

```powershell
.\.venv\Scripts\python.exe .\TriCSI_review_response\run_development_baselines.py
```

This uses only raw 3TX_S1 and 3TX_S2. It tests the outer pair and all three links with four conditions: full-feature Extra Trees, RSSI-only Extra Trees, CSI-without-RSSI Extra Trees (CSI shape plus gain), and full-feature Random Forest. Both session-transfer directions use three preset seeds and whole-trial probability aggregation. These are new development controls, not published-method reproductions or independent confirmation on S3. They do not overwrite or refit the frozen bundle.

## 2 Matched-seed neural component checks

```powershell
.\.venv\Scripts\python.exe .\TriCSI_review_response\run_matched_anchor_ablations.py --project-root .
```

Requires your original run_tricsi_anchor_ablations.py, train_tricsi_anchor_final.py, train_csi_cnn.py, project_config.py, their dependencies, and raw S4-S7. The default is 36 training runs: 3 variants x 4 held-out DEVELOPMENT sessions x 3 seeds, 70 epochs each. It can take considerable time. Do not use S8/S9 to choose epochs, seeds, or variants.

Each variant receives the same seed for a given session and repetition. Training still follows the original code. This reduces the original seed confound but is not proof that every stochastic operation is identical. Seeds are repeated algorithm runs, not new people or days. Results save to a NEW folder results/anchor_matched_seeds. If that folder already exists, use --output with a new folder; never delete old results to make an apparent first run.

## 3 Record a replication commitment before collecting

Confirm and log whether the original three-link setup remains unchanged: room, board identities and MAC mapping, positions/heights, antenna directions, power, firmware, receiver settings, grid, and furniture. If it has changed, document the changes and describe later measurements as a changed-setup test rather than an exact replication. Do not quietly adjust the arrangement after inspecting model accuracy.

Suggested manageable fixed series: three additional 50-trial sessions, 3TX_S4, 3TX_S5, and 3TX_S6, on separate days. Three is a practical starting series, not a statistical guarantee. Record intended dates and the inclusion/exclusion rules before collection; if these session names already exist, choose genuinely new names and change all commands consistently.

Record the local original frozen model hash BEFORE collection. If you already overwrote the original model, stop and identify the authentic saved model rather than treating a newly fitted model as the original.

```powershell
$FrozenModel = '.\results\3tx_dev_comparison\frozen_3tx_model.joblib'
$FrozenHash = (Get-FileHash $FrozenModel -Algorithm SHA256).Hash.ToLower()
$FrozenHash
$FrozenHash | Set-Content .\results\replication_model_sha256.txt
```

Copy that hash, session names, intended dates, fixed model/feature settings, and primary comparison (3tx_all minus 2tx_outer) into your dated research notebook. Retain every scheduled completed session. Base technical exclusions on recorded acquisition failures, not accuracy. Collect the whole fixed series before opening its prediction results if practical.

Use the original collector with the same protocol; run each command on its scheduled day:

```powershell
.\.venv\Scripts\python.exe .\collect_three_link_trials.py --session 3TX_S4 --seed 312
.\.venv\Scripts\python.exe .\collect_three_link_trials.py --session 3TX_S5 --seed 313
.\.venv\Scripts\python.exe .\collect_three_link_trials.py --session 3TX_S6 --seed 314
```

Do NOT rerun compare_and_freeze_3tx.py as part of this replication. The purpose is to test the saved model rather than adapt it to the new recordings. Keep any model development as a separate study with separate future test data.

## 4 Evaluate the original frozen model on the new sessions

```powershell
$FrozenHash = (Get-Content .\results\replication_model_sha256.txt).Trim()
foreach ($SessionName in @('3TX_S4','3TX_S5','3TX_S6')) {
    .\.venv\Scripts\python.exe .\TriCSI_review_response\run_frozen_replication.py --model .\results\3tx_dev_comparison\frozen_3tx_model.joblib --expected-model-sha256 $FrozenHash --session ".\data_3tx\raw\$SessionName" --output ".\results\frozen_replication\$SessionName"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $SessionName" }
}
```

The evaluator checks model SHA256, training-name/content overlap, and grid coordinates. It cannot verify that physical geometry truly remained unchanged. It does not fit, tune, or select a model. Each output folder must be new. It evaluates all original subsets for transparency; 3tx_all versus the outer pair remains the primary comparison. Any trial bootstrap interval is conditional on that session, not an estimate of generalization across days.

## 5 Summarize each session separately

```powershell
.\.venv\Scripts\python.exe .\TriCSI_review_response\summarize_replication.py --predictions .\results\3tx_s3_final\s3_trial_predictions.csv .\results\frozen_replication\3TX_S4\trial_predictions.csv .\results\frozen_replication\3TX_S5\trial_predictions.csv .\results\frozen_replication\3TX_S6\trial_predictions.csv --output .\results\replication_summary
```

This creates a per-session CSV and Python graph, with S3 identified as the initial test. It reports the new sessions descriptively and does not manufacture an across-session confidence interval from a small number of days. Every input must have identical matched 50-trial predictions for the two subsets. Verify model and acquisition logs before including any file.

## Validation completed here

- New reporting reproduced the saved S3 results: 70% outer pair, 84% all links, seven improved trials and none harmed.
- Feature-selection checks use the actual prefixed feature names from your module.
- Command-line/import and synthetic integration checks are recorded in VALIDATION.md.
- Raw trial CSVs are absent from the uploaded archive, so no real-data training, neural ablation runs, or new-session evaluations were completed here.

Read PAPER_REVISION_GUIDE.md for manuscript-specific feedback. A direct, faithful comparison against relevant published methods remains additional work; these simple controls should not be described as that comparison.
