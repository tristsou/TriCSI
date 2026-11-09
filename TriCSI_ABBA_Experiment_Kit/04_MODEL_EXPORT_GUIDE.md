# Bridging your actual frozen 3TX model to the prediction CSV

The dataset in this chat does **not** contain the Windows project source code, frozen Extra Trees artifacts, or raw 3TX session data. A universal script that calls those models is therefore not possible without guessing APIs. This document tells you exactly what to export from your existing `3TX_S3` evaluator.

## Required model inputs

For each session S4-S7, two saved classifiers:

1. `2tx_outer`: only `BACK_LEFT` plus `BACK_RIGHT` paths (324 engineered features if using 162 per link).
2. `3tx_all`: `BACK_LEFT` + `MIDDLE_BACK` + `BACK_RIGHT` (486 features).

Use the original saved feature extractor and model objects, trained from `3TX_S1 + 3TX_S2` only; each physical trial's seven windows yield class probabilities that are averaged before argmax, exactly as done for S3. Verify class ordering from model's saved `classes_` attribute rather than assuming a hard-coded numeric label mapping.

## Output exactly one row per physical trial and model

Your existing evaluator likely already writes a row per trial to a CSV. At the point where its final predicted class is known, the only needed fields are:

```python
# Pseudocode for integration into your EXISTING real inference code.
# Replace variables with the actual names from your evaluator.
result = {
    'session': session_name,                  # '3TX_S4' ... '3TX_S7'
    'trial_id': trial_csv_filename,           # exact raw trial filename
    'true_label': ground_truth_label,         # 'EMPTY' or 'Z1'...'Z9'
    'subset': subset_name,                    # '2tx_outer' or '3tx_all'
    'predicted_label': model_predicted_label, # same ten labels
}
```

Pivot or join *by session and trial_id*, then insert the outputs as `pred_2link` and `pred_3link` in the provided `prediction_template.csv`. Do not join by row number: the two evaluators may use different file ordering.

## Strict checks

- Same 50 trial IDs for both models in each session.
- Exactly one record per model/trial combination.
- Predicted labels use `EMPTY`, `Z1` ... `Z9` only.
- Ground-truth labels must match the actual collected zone and original CSV filename.
- Do not run `fit`, `fit_transform`, or optimization on any data from S3-S7. `predict_proba` only, except preprocessing transformations with fixed training-fitted parameters, if applicable.
- No labels from S4-S7 in prototypes, thresholds, imputation, calibration parameters, or learned feature normalizers.
- The analyzer will error on blank predictions, missing trials, duplicates, or unexpected class counts rather than silently ignoring them.

If you provide your actual original 3TX evaluator Python script and model-file paths, the import/export glue can be made exact and automated without guessing.
