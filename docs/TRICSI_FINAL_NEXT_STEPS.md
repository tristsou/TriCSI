# TriCSI — Final next steps after S8

The model is now frozen.

Do **not** change:
- `tricsi_anchor_s4_s5_s6_s7.pt`
- neural architecture
- training epochs
- losses
- preprocessing
- CSI packet rate
- zone geometry
- antenna geometry
- classification thresholds

The goal now is confirmation and evidence, not more model development.

## Step 1 — Run the fixed ablation study

This uses only development sessions S4-S7.

```cmd
python run_tricsi_anchor_ablations.py
```

It compares:
1. FULL method;
2. no empty-room background subtraction;
3. no special body-orientation invariance.

S8/S9/S10 are never used for this ablation.

This tells you which parts of the final method matter.

## Step 2 — Collect S9

Ideally collect it on a different day/time from S8, while keeping the exact same physical geometry.

```cmd
python collect_confirmatory_session.py --session S9
```

The model gets only:
- 12 s EMPTY background calibration.

The 36 occupied trials are evaluation-only:
- Z1-Z9;
- N/E/S/W;
- one 8 s trial per orientation.

There are also four separate EMPTY evaluation trials.

## Step 3 — Evaluate S9 with the exact saved model

```cmd
python evaluate_frozen_anchor_confirmatory.py --sessions S8 S9
```

This loads the saved S4-S7 checkpoint. It does not train anything.

## Step 4 — Collect S10 on another day/time

```cmd
python collect_confirmatory_session.py --session S10
```

Do not modify the model after seeing S9.

## Step 5 — Final pooled frozen evaluation

```cmd
python evaluate_frozen_anchor_confirmatory.py --sessions S8 S9 S10
```

This gives:
- exact accuracy for every session;
- pooled exact accuracy;
- 95% Wilson interval;
- macro F1;
- exact-or-adjacent accuracy;
- weighted-coordinate errors;
- per-zone recall;
- LEFT/RIGHT learned gate values.

This becomes the main confirmatory result.

## Step 6 — Create final report figures

```cmd
python make_tricsi_final_figures.py
```

It produces:
- held-out session accuracy chart;
- pooled confusion matrix;
- per-zone recall chart;
- learned LEFT/RIGHT gate chart;
- summary metrics CSV.

## Scientific interpretation

S8 remains the first untouched final test.

S9 and S10 are **prospective confirmatory replications** using the exact same frozen model. They answer the key question:

> Does the 95% S8 result repeat across additional sessions, or was S8 unusually favorable?

Do not hide a weak S9/S10. The cross-session variation itself is scientifically important.

## After S10

Stop collecting data for the main project unless there is a genuine data-quality failure such as a dead transmitter or corrupted file.

At that point the research package should contain:
- S4-S7 development sessions;
- frozen final model;
- S8 first held-out final test;
- S9/S10 prospective confirmation;
- fixed ablation study;
- final figures/statistics;
- live inference demonstration.

That is enough for the main STS research report.
