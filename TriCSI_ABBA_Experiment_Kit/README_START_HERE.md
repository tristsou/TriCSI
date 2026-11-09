# TriCSI A–B–B–A Study: Start Here

This is an **experiment-planning and analysis kit** for a post-review investigation of environmental robustness. It is not an ESP32 data collector and it **does not contain your trained Extra Trees models**. Those live on your Windows computer. Do not claim that this kit ran the frozen models.

## The question and critical experiment

**Question:** Does adding a third Wi-Fi transmitter reduce sensitivity of stationary zone localization to a deliberate change in room furniture?

**Design:** Four *new* 50-trial sessions, all recorded using the same three ESP32-S3 transmitters and one receiver. The two-link and three-link predictions must use **the same raw trial recordings**.

| New session | Physical condition | Set up |
|:--|:--|:--|
| 3TX_S4 | A_normal | Normal/original layout, 50 trials |
| 3TX_S5 | B_changed | Move one chair to predefined, safely marked location, 50 trials |
| 3TX_S6 | B_changed | Leave the chair moved; restart collection as separate session, 50 trials |
| 3TX_S7 | A_normal | Restore chair exactly to original position, 50 trials |

A-B-B-A is a **same-day, within-room case study**, not proof of robustness in other homes or long-term monitoring.

## STOP/GO: do this BEFORE recording

1. **Approvals:** Confirm with the adult sponsor/mentor or relevant IRB that a deliberate furniture-layout change and any new recordings are covered by your approved human-participant research plan. If not, get required approval first. Do not proceed without it. This kit is not an IRB approval.
2. **Freeze:** Identify the exact Extra Trees models already fitted on **3TX_S1 + 3TX_S2**, and verified on **3TX_S3** (original two-link outer pair versus three-link all). Never fit on S3, S4, S5, S6, or S7; never tune on new results. Preserve the model files, code, configuration, and hashes. If the original model files do not exist, any reconstruction must be documented as a reconstructed model and independently checked against saved historical S3 predictions. Do not call it the original frozen model without verification.
3. **Lock the chair manipulation:** Choose **one** movable chair, measure and tape-mark A and B footprints (e.g., about 0.6 m apart, if feasible), photograph each empty-room arrangement, and write down exactly which way it faces. Keep the chair outside the marked standing zones and walking path. Do not move transmitter/receiver locations, external antennas, cabling, or the 8 ft × 6 ft grid. Choose the B position *before* seeing any new test predictions.
4. **Lock the metrics:** Read `01_PREREGISTRATION.md`, write the setup details and date, and save the file before collecting. The primary metric is the **difference in added-link gain between B and A**, not the largest observed accuracy.
5. **Verify setup:** Make sure the 3TX geometry matches the original development setup (BACK_LEFT, MIDDLE_BACK, BACK_RIGHT; one receiver), or document deviations and avoid interpreting changed geometry as a clean furniture effect. The source paper specifies 36-inch, vertical, fixed external antennas.

## Collect four sessions

Use the **same existing collection program** and raw CSI protocol from the 3TX_S1/S2/S3 study. Use `collection_order.csv` only if the collector supports externally specified order; otherwise retain the collector's own randomized order and save its actual order. Do **not** change the collector merely to match this kit.

- Per session: five EMPTY + 5 trials for each Z1–Z9 = 50 trials. Occupied trial orientations N, E, S, W, N. Initial EMPTY recording 12 seconds; other recordings 8 seconds. Allow the original movement/settling time.
- Do each new session as a new directory. Keep all raw `.csv` packet streams and metadata. Keep one participant, clothing, doors, lighting/occupants and antenna placements as consistent as practicable; log deviations rather than erasing them.
- Session S4: A original. S5: move chair once and record B. S6: leave B arrangement and start a second separately initiated session. S7: restore A, record again. Photograph chair locations and equipment at beginning and end. Do not have people deliberately fall or perform extra movements.
- Set output paths with explicit new session names to avoid overwriting any historical data.
- The original source had approximately 33.6 packets/second **per link**. Confirm data are present for all 3 transmitter MAC labels in each trial. Log packet counts and missing data; no silent deletions.
- If hardware is bumped, record time, session, trial and the details. Do not silently remove a troublesome session because the accuracy is poor. Declare obvious protocol-invalid trials using rules established before looking at model predictions, and keep a complete audit trail.

`session_log.csv` records session metadata; append short contemporaneous notes and retain raw files untouched.

## Export frozen-model predictions

**Do not train new models to score these sessions.** Run the same 3TX frozen evaluation script that originally evaluated 3TX_S3, pointing it at the new S4-S7 session directories. If your script can score one new session at a time, that's fine. For each raw trial, score the exact same recording using both (a) BACK_LEFT + BACK_RIGHT and (b) all three transmitters. Keep the same feature extraction: 7 windows × 64 packets, 52 populated subcarriers, 162 features/link, mean window probabilities per trial, identical fitted preprocessing and thresholds. Do not substitute the older two-link CNN from S4-S9; that is a different hardware/classifier experiment.

**Important:** This kit cannot invoke your models because their `.pkl`/`.joblib` files and the 3TX inference source code were not uploaded. It supplies an analysis script for their resulting prediction CSV, not an invented evaluation function.

To locate the real evaluator and model artifacts in Windows PowerShell:

```powershell
cd C:\ESP32-CSI-Tool\tricsi_localization
Get-ChildItem -Recurse -File -Include *.joblib,*.pkl,*.pickle | Select-Object FullName
Get-ChildItem -Recurse -Filter *.py | Select-String -Pattern '3TX_S3|ExtraTreesClassifier|predict_proba|joblib.load' | Select-Object -First 35 Path,LineNumber,Line
```

Once the right saved model file is identified, record a hash:

```powershell
Get-FileHash 'C:\PATH\TO\SAVED_MODEL.joblib' -Algorithm SHA256
```

**This is a search command, not an instruction to retrain.** If the model evaluation script is unclear, stop and share the file or its output so it can be connected correctly.

## Enter model predictions in one CSV

The included `prediction_template.csv` already lists **all 200 planned physical trials** (no predictions are filled). Add predicted class labels in `pred_2link` and `pred_3link`, using only `EMPTY` or `Z1` through `Z9`. Match predictions by **session + exact trial filename**; do not rely on row order.

CSV header:

```csv
session,condition,trial_id,true_label,orientation,pred_2link,pred_3link
```

The example below is format-only, **not an observed experimental result**:

```csv
3TX_S4,A_normal,Z1_trial01_N.csv,Z1,N,Z1,Z2
```

If your collector generates different file names, edit `trial_id` accordingly. For CSV import, ensure trial IDs are unique inside each session.

## Optional: automatically merge the two frozen-model export files

If the 2-link and 3-link evaluators can each export a CSV with `session,trial_id,true_label,predicted_label`, you **do not** need to type 400 predictions into the template. Combine the files by their physical trial ID with:

```powershell
.\.venv\Scripts\python.exe .\TriCSI_ABBA_Experiment_Kit\combine_model_predictions.py `
  --two .\outer_pair_predictions.csv `
  --three .\all_three_predictions.csv `
  --template .\TriCSI_ABBA_Experiment_Kit\prediction_template.csv `
  --output .\TriCSI_ABBA_Experiment_Kit\filled_predictions.csv
```

Then use `filled_predictions.csv` as the `--input` to `analyze_abba.py` instead of the blank `prediction_template.csv`. The combiner rejects mismatched trials, inconsistent ground-truth labels, and missing predictions. See `04_MODEL_EXPORT_GUIDE.md`.

## Run the analysis on your Windows machine

Unzip this folder under `C:\ESP32-CSI-Tool\tricsi_localization\`, or adjust paths as needed. Your existing `.venv` needs `pandas`, `numpy`, and `matplotlib` (usually already installed if you generated previous graphs).

```powershell
cd C:\ESP32-CSI-Tool\tricsi_localization
.\.venv\Scripts\python.exe .\TriCSI_ABBA_Experiment_Kit\analyze_abba.py `
  --input .\TriCSI_ABBA_Experiment_Kit\prediction_template.csv `
  --output .\TriCSI_ABBA_Experiment_Kit\results
```

Running it **before filling predictions** intentionally produces an error rather than fake results.

To regenerate the randomized trial-order file and prediction template (CAUTION: overwrites the blank template):

```powershell
.\.venv\Scripts\python.exe .\TriCSI_ABBA_Experiment_Kit\make_templates.py
```

Do not rerun `make_templates.py` after entering actual predictions unless you have backed up the filled file.

## What the analyzer will produce

- `results/SUMMARY.md`: complete per-session and condition summary, predeclared environmental-robustness contrast, limitations.
- `results/per_session_metrics.csv`: 2/3 link whole-trial, occupied and EMPTY accuracy, errors, corrections and harms.
- `results/condition_metrics.csv`: mean of A sessions and mean of B sessions.
- `results/trial_outcomes.csv`: full, auditable 200-row prediction outcomes.
- `results/per_class_metrics.csv`: per-zone and EMPTY accuracy by session.
- `results/paired_bootstrap_conditional.csv`: trial-resampled paired conditional intervals, **not** room-level generalization intervals.
- `results/figure_accuracy_by_session.png`: primary result chart.
- `results/figure_link_gain_by_session.png`: per-session gain/loss chart.
- `results/figure_condition_comparison.png`: normal versus changed comparison.

## The analysis decision rule

Let A be the **average of S4 and S7**, B be the **average of S5 and S6**, for each model.

- `2-link change = acc(2link,B) - acc(2link,A)`.
- `3-link change = acc(3link,B) - acc(3link,A)`.
- **Robustness contrast = 3-link change − 2-link change = (3-link gain in B) − (3-link gain in A)**.

A **positive robustness contrast** means three links were *relatively less affected* than the outer pair in this experiment, even if both performed badly. This is distinct from claiming the absolute 3-link accuracy is useful.

Treat results as a **same-day, single-person, single-room controlled case study**. Report all new sessions including unsuccessful results; don't present S3 as a new replication or as part of the new A condition, since it comes from an earlier collection.

## If human-study approval cannot be confirmed today

Do not collect new human participant data. Use the day to lock code/artifacts, analyze existing development and S3 predictions, perform already-approved non-human hardware checks, and obtain the necessary review/authorization for a new protocol. Do not backdate records.

## Research integrity / STS reminders

- Student must independently make decisions, understand scripts, audit results, and write STS submission text under current competition rules; document/disclose appropriate AI assistance.
- The 2027 STS application deadline is **November 5, 2026, 8 PM ET**, and the FAQ describes a **maximum 20-page original scientific paper**. Check the official 2027 Research Report Guidelines and do not submit the 29-page working draft without reducing it.
- This investigation does not demonstrate continuous tracking, gait assessment, falls, multi-person localization, or deployment in other homes.

Official references:
- https://www.societyforscience.org/regeneron-sts/official-rules/
- https://www.societyforscience.org/regeneron-sts/application-requirements/
- https://www.societyforscience.org/regeneron-sts/frequently-asked-questions/
