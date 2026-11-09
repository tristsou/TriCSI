# TriCSI six model study

This package orchestrates your ORIGINAL three-link extractor, Extra Trees classifier, collector, and whole-trial probability aggregation. It does not rebuild CSI parsing or change your original frozen model. All new code was generated with ChatGPT assistance; preserve the prompt history and verify the outputs yourself.

## Read this first

- Run in your EXISTING working virtual environment. Do not upgrade packages or rerun compare_and_freeze_3tx.py to force historical agreement.
- The original archived model records scikit-learn 1.9.0. Loading across scikit-learn versions is not supported; this toolkit stops on a version warning. Its real-data reproduction must run on your PC with the original compatible environment.
- Actual raw trial CSVs were not uploaded here. Saved S3 predictions were checked, but actual historical raw-data reproduction and new-session results remain to be produced locally.
- The code uses your actual module at tricsi_3tx_modeling_bundle/tricsi_3tx_modeling.py. The reference_sources folder is an exact inspected snapshot for comparison, not an automatic replacement.
- Both new sessions should be covered by the appropriate research approval. This toolkit does not assess approval status.
- Keep your current room, furniture, grid, antennas, and boards in place. No room photographs or disturbances are requested. Record any known changes since S1-S3; code cannot certify physical geometry. If the arrangement has changed, describe it accurately rather than claiming an exact unchanged-setup replication.

## Why there are six new models plus two historical models

The original frozen outer-pair model used seed 2029; the original all-three model used seed 2032. Their 70% and 84% S3 results remain historical results. They are preserved as H1 and H2 and evaluated unchanged on the two new sessions.

All SIX new comparison models use seed 2026, the original 400-tree Extra Trees settings, S1+S2 only, identical extracted windows, and identical trial aggregation. The full-feature models are newly frozen matched-seed controls, NOT replacements for H1/H2. No new M1/M2 S3 result is claimed or expected to be 70/84.

| ID | Links | Features per link | Meaning |
| --- | --- | ---: | --- |
| M1 | Outer pair | 162 | CSI shape + packet gain + RSSI |
| M2 | All three | 162 | CSI shape + packet gain + RSSI |
| M3 | Outer pair | 159 | CSI shape + packet gain, no RSSI |
| M4 | All three | 159 | CSI shape + packet gain, no RSSI |
| M5 | Outer pair | 3 | RSSI summaries only |
| M6 | All three | 3 | RSSI summaries only |
| H1 | Outer pair | 162 | Original historical frozen model, unchanged |
| H2 | All three | 162 | Original historical frozen model, unchanged |

M3/M4 are the same feature view as your recent CSI-without-RSSI ablation. They do NOT isolate spectral shape alone because they retain gain. The six conditions test predictive feature contributions, not a uniquely identified physical mechanism. Changing feature dimension also changes the number of candidate split features under max_features="sqrt"; that is retained from your methodology and must be acknowledged.

Primary historical replication contrast: H2 minus H1. Matched-seed link contrasts: M2-M1, M4-M3, M6-M5. Feature interaction: (M4-M3)-(M6-M5). M2-M6 and M1-M5 compare full features with RSSI alone. These are committed before the two future outcomes; no seed/model/subset is selected using them.

## 0 Extract once into the project root

Open PowerShell:

```powershell
Set-Location C:\ESP32-CSI-Tool\tricsi_localization
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\TriCSI_six_model_study.zip" -DestinationPath .
Test-Path .\TriCSI_six_model_study\tricsi_day.py
```

The last command must print True. If you already extracted the ZIP, just place its inner TriCSI_six_model_study folder directly under the project root and run Test-Path. Do not extract a second nested folder. No virtual-environment activation is needed.

The package uses numpy, pandas, scikit-learn, scipy, matplotlib, and joblib already present in your working modeling environment. The original collector requires pyserial. This experiment does not require PyTorch.

## 1 Inspect the actual local code and saved models

```powershell
.\.venv\Scripts\python.exe .\TriCSI_six_model_study\tricsi_day.py inspect
```

Expected checkpoint: INSPECTION PASS. Output:

    results/six_model_inspection_20261030/environment.json
    results/six_model_inspection_20261030/inspection.json

The inspection verifies the original extractor's Python syntax tree against the inspected snapshot (whitespace and line-ending changes are tolerated), checks the current collector's MAC/grid/protocol, lists actual model seeds and parameters, and compares the recent review-response extractor when present. Importing the collector here does not open a serial port.

If a code-version guard stops, share the inspection/error and the actual changed file; do not overwrite it with a reconstructed parser. If an environment warning stops, print:

```powershell
.\.venv\Scripts\python.exe -c "import sys, sklearn, numpy, pandas, joblib; print(sys.version); print('sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'joblib', joblib.__version__)"
```

## 2 Reproduce S3 without fitting anything

```powershell
.\.venv\Scripts\python.exe .\TriCSI_six_model_study\tricsi_day.py reproduce
```

Expected checkpoint: HISTORICAL REPRODUCTION PASS: H1=70%, H2=84%.

Outputs under results/s3_historical_reproduction_20261030:

- historical_metrics.csv
- historical_trial_predictions.csv
- reproduction.json (raw/model hashes, environment, original paired bootstrap)
- saved_prediction_comparison.csv, when your original s3_trial_predictions.csv is present
- raw_audit/audit.json, row_issues.csv, trial_link_counts.csv

It compares the actual 50 trial predictions with your saved historical predictions when available, not merely the two accuracy totals. A mismatch stops with saved diagnostics; it does not change parameters, labels, or exclusions until the scores match.

Equivalent supported ORIGINAL evaluator command, if you want an additional direct original-script check AFTER inspect passes:

```powershell
.\.venv\Scripts\python.exe .\tricsi_3tx_modeling_bundle\evaluate_frozen_3tx.py --model .\results\3tx_dev_comparison\frozen_3tx_model.joblib --session .\data_3tx\raw\3TX_S3 --output .\results\s3_original_script_recheck_20261030
```

That direct command uses the original evaluator's actual flags. It is optional; do not rerun its original output directory.

### S3 row 418 and len metadata

The original extractor does not use len. It reads the 128 saved IQ columns and RSSI. Declared original lengths 128, 256, and 384 are not compared with saved-column count.

The audit records an invalid len as an unused_metadata_warning. If that row's IQ/RSSI remain valid, the original pipeline retains it; dropping it solely because of an unused field would change historical behavior without a feature-based reason. If IQ or RSSI are nonnumeric/nonfinite, the original extractor already excludes that row and the audit logs it as excluded_by_original_extractor. Structural field-count damage stops with the file and line, rather than guessing how to repair it. No original file is edited or silently cleaned.

Consequently, a prior audit's "valid row" total that required every metadata field to be valid can differ from usable_feature_rows here. This is a documented definition difference, not a manufactured replacement result. The uploaded prompt's specific row cannot be verified without the raw S3 file on your PC.

## 3 Freeze before collecting S4/S5

```powershell
.\.venv\Scripts\python.exe .\TriCSI_six_model_study\tricsi_day.py freeze
```

Expected checkpoint: FREEZE PASS.

Run directory:

    results/3tx_six_model_20261030_v1/

It contains six separate models models/M1.joblib through models/M6.joblib; a byte-identical COPY of the historical bundle; model_definitions.csv; manifest.json; manifest_sha256.txt; and training audit files. The manifest stores source/model/data hashes, seed, selected feature names, classifier parameters, class/grid definitions, exact versions, and planned future session IDs/order seeds. Your original historical model is never written to.

The training data must match the raw CSV hashes stored in the original historical bundle. No S3/S4/S5 labels enter fitting. No target normalization is fitted. Freeze refuses if S4 or S5 already contains trials; after observing results, never rename an old session to masquerade as new. If those names are already used, choose genuinely future IDs with --future-sessions and use them consistently in subsequent commands.

Record the manifest hash and your intended collection times in your dated notebook before collecting. The file is an auditable local commitment, not an externally authenticated timestamp.

## 4 Collect two sessions with the unchanged collector

```powershell
.\.venv\Scripts\python.exe .\TriCSI_six_model_study\tricsi_day.py collect --session 3TX_S4
.\.venv\Scripts\python.exe .\TriCSI_six_model_study\tricsi_day.py collect --session 3TX_S5
```

Each wrapper invokes your ORIGINAL collect_three_link_trials.py with its supported options: session name, collection seed 312/313, five trials/class, 8-second ordinary recordings, 12-second first EMPTY, 5-second movement countdown, and a longer preflight minimum of 100 packets/link. The first EMPTY is followed by 49 randomized trials. Actual raw data go to the original data_3tx/raw/<SESSION> folders.

The receiver default from the inspected collector is COM5 at 921600 baud. If your current receiver uses a different port, add --port COMx to each wrapper command and log that setting. Port naming does not by itself identify the physical receiver. Keep the same hardware and geometry if claiming replication.

Take an ordinary break between sessions; record actual start times and elapsed separation. Same-day sessions are short-term cross-session replication, not independent-day robustness. No environmental disturbances, furniture movement, or room photos are required. Retain both sessions regardless of their eventual accuracy.

Expected checkpoint after EACH session: COLLECTION AUDIT PASS. The acquisition criterion is 50 distinct protocol trials, all three known MACs, valid labels, and at least 64 valid saved IQ/RSSI packets per link per trial, sufficient for the unchanged extractor. Other metadata/timing warnings are logged and require review; do not invent post-score exclusion thresholds.

If interrupted, resume with the SAME command plus --resume. It requires the original receipt and identical settings. The original collector skips every existing filename, including a partial trial. Partial files are not overwritten automatically. If a partial file fails the acquisition audit, preserve it and the interruption record and arrange a genuinely new session; do not delete data based on model accuracy.

If you need a separate manual preflight, use a separate name so the original script does not overwrite a historical session manifest:

```powershell
.\.venv\Scripts\python.exe .\collect_three_link_trials.py --session 3TX_PREFLIGHT_20261030 --preflight-only --preflight-seconds 30 --preflight-min-packets 100
```

Do not run default preflight-only into 3TX_S1. The unchanged original collector writes its manifest/schedule before the preflight-only exit.

## 5 Evaluate both sessions together

Only after both collection audits pass:

```powershell
.\.venv\Scripts\python.exe .\TriCSI_six_model_study\tricsi_day.py evaluate
```

Expected checkpoint: EVALUATION PASS. Per-session outputs:

    results/3tx_six_model_20261030_v1/evaluations/3TX_S4/metrics.csv
    results/3tx_six_model_20261030_v1/evaluations/3TX_S4/trial_predictions.csv
    results/3tx_six_model_20261030_v1/evaluations/3TX_S4/evaluation.json

The analogous files are written for S5, along with M1-M6 confusion-matrix CSVs. Each session has 400 prediction rows: eight fixed models x 50 trials. Only the six new models form the matched-seed feature comparison; H1/H2 form the separate historical replication.

The evaluator refuses changed model/toolkit/manifest hashes, changed environments, wrong grids, training-data overlap, byte-duplicate trials across training/new sessions, pre-existing output folders, or sessions lacking prospective collection receipts. It validates both requested datasets before scoring. No model fitting, label-based normalization, threshold selection, or subset selection occurs.

Model verification does not prove the room's physical arrangement was unchanged. Do not change any scientific choices after opening S4/S5 outcomes. Preserve errors as results, including occupied trials predicted EMPTY.

## 6 Generate statistics and figures

```powershell
.\.venv\Scripts\python.exe .\TriCSI_six_model_study\tricsi_day.py report
```

Output folder:

    results/3tx_six_model_20261030_v1/report/

Key tables:

- metrics_by_session.csv: ten-class trial accuracy, occupied accuracy, EMPTY recall, original zone-based scores, missed-occupancy counts.
- paired_contrasts.csv: the seven planned accuracy contrasts, matched corrections/new errors, 95% trial bootstrap intervals.
- feature_link_interaction.csv: (M4-M3)-(M6-M5), in percentage points.
- paired_trial_effects.csv: trial-level contributions to the planned contrasts.
- paired_zone_error_score_reductions.csv: paired occupied-trial score reductions and intervals.
- descriptive_session_summary.csv: per-model means, ranges, and descriptive SD across the two sessions.
- all_trial_predictions.csv, per_class_results.csv, orientation_diagnostics.csv.

Figures are exported as 300-dpi PNG plus vector PDF and SVG:

- six_model_accuracy: six matched-seed conditions, shown separately for S4 and S5.
- paired_link_gains: historical and matched-seed link gains with intervals for each session.
- corrected_and_harmed_trials: corrections versus newly introduced errors.
- 3TX_S4_confusion_matrices and 3TX_S5_confusion_matrices: six-model confusion matrices.

No synthetic figures are included as scientific output in this ZIP. The figures you generate locally use your actual new predictions. Credit their code/software/AI assistance accurately if used in a paper.

### Statistical interpretation

Whole trials are resampled within actual class and session, keeping all model predictions paired. Adjacent windows are never counted as independent trials. Intervals are unadjusted marginal percentile intervals from 5,000 resamples, conditional on the recorded trials. Trial-level dependence or other acquisition effects can still limit their interpretation. Do not present them as family-wise-corrected significance tests.

The pooled-observed-sessions interval holds those two sessions fixed; it is NOT a confidence interval for future-session performance. Two same-room sessions do not support a convincing estimate of day-to-day, cross-person, or cross-home reliability. Report each session and the observed range, including any reversals.

The original position_error_m column is retained for compatibility. It is a zone-center classification score for occupied trials, including the frozen 1.873 m penalty for occupied-to-EMPTY errors; it is NOT continuously measured positioning accuracy. EMPTY trials remain in ten-class accuracy but not that distance-score average.

## Optional lower-priority S8/S9 inventory

```powershell
.\.venv\Scripts\python.exe .\TriCSI_six_model_study\audit_s8_s9_readonly.py
```

This reads actual project_config.py MAC definitions and saved IQ columns, counts trials/packets, flags formatting/mapping issues, identifies EMPTY trial 1, inventories the original checkpoint and source S4-S7 hashes, and records the inspected different S8/S9 scoring rules. It writes to a new folder and does not alter the old model or S9.

It does NOT rerun neural inference or establish that historical prototype/normalization values match. The original neural evaluator recomputes source prototypes from S4-S7 at evaluation time; identical encoder weights alone do not establish identical prototypes if those source inputs/code changed. The actual checkpoint and raw S8/S9 were not uploaded here, and PyTorch is absent from the verification environment. Investigate a cause only if actual records support it; low S9 accuracy is not grounds for exclusion.

## Failure and overwrite policy

All substantive result/model folders must be new. Stop if an instruction prints STOP or a traceback. Audits are saved before most data-integrity failures; share those rather than patching raw files. An unsuccessful training directory is marked by the absence of a completed manifest; do not treat it as frozen. If retrying a technical training failure before collection, use --run-dir with a genuinely new versioned directory for freeze and ALL later stages. Do not silently delete an earlier attempt.

If only report rendering fails, the immutable evaluation CSVs remain; rerun report with --output pointing to a new report folder. Never edit the model/manifest/code after freezing merely to bypass its guard.

Optional synthetic self-test (not necessary before your real workflow):

```powershell
.\.venv\Scripts\python.exe .\TriCSI_six_model_study\verify_toolkit.py
```

It uses temporary synthetic data and mocked serial collection and removes its temporary files. It does not evaluate your real S3, S4, or S5.

## Return the actual outputs for review

After all commands finish:

```powershell
Compress-Archive -Path .\results\3tx_six_model_20261030_v1\manifest.json, .\results\3tx_six_model_20261030_v1\manifest_sha256.txt, .\results\3tx_six_model_20261030_v1\model_definitions.csv, .\results\3tx_six_model_20261030_v1\evaluations, .\results\3tx_six_model_20261030_v1\report, .\results\3tx_six_model_20261030_v1\collection_3TX_S4.json, .\results\3tx_six_model_20261030_v1\collection_3TX_S5.json -DestinationPath .\TriCSI_S4_S5_results_for_review.zip
```

This ZIP includes outputs and provenance but not the large model copies or raw CSI. Keep all raw files/models locally. Do not claim the new experiment resolves clinical monitoring or methodological novelty; assess what the actual results support.
