# Results ledger

[reported_results.json](reported_results.json) preserves the initial owner-reported
and log-derived evidence. The owner subsequently supplied a verified full bundle.
[Reproduction receipts](verified/) now record rerunning the original fitted
two-link model and the outer/all-three link models without training. All 180
compared trial labels match. These are checks of the same historical recordings,
not fresh independent validation. The GitHub bundle attachment is pending.

The owner-uploaded measured comparison/reconstruction CSVs and the newly reproduced
trial predictions are tracked in `verified`. Checkpoint and bundle hashes in the
receipts link them to the complete manifest. The validation host used Linux/Python
3.12.14, CPU PyTorch 2.12.1 and scikit-learn 1.9.0. The owner's captured working
environment is Windows/Python 3.14.6. Tiny probability differences did not change
any compared label. These checks do not reproduce model training itself.

## S4–S7 shape comparison

Source data S4–S7; development S8/S9; 40 scored trials per session; 150 epochs;
evaluation every five epochs; existing augmentation in both arms. One checkpoint
is selected jointly per arm/seed by worst-session accuracy, then mean accuracy
and macro-F1. Each listed winner is unique on the first two criteria in the log.

| Representation | Seed | Selected epoch | S8 | S9 |
| --- | --- | --- | --- | --- |
| Original shape | 2026 | 90 | 82.5% | 62.5% |
| Original shape | 2027 | 80 | 72.5% | 70% |
| Original shape | 2028 | 110 | 82.5% | 65% |
| Empty-relative shape | 2026 | 105 | 77.5% | 50% |
| Empty-relative shape | 2027 | 130 | 72.5% | 55% |
| Empty-relative shape | 2028 | 70 | 67.5% | 57.5% |

Mean across seeds/sessions: original shape 72.50%, Empty-relative shape 63.33%.
The candidate lost 9.17 percentage points overall. This comparison does not
support the proposed shape change. It does not establish why it failed.
These seeds are repeated trainings, not independent new recording sessions.

## Preserve complete evidence

Each release should include real per-trial predictions, metrics, curves,
configuration, selected checkpoint index, model/data/code hashes and environment.
Track small reviewed tables here with explicit provenance; keep bulk artifacts
in the versioned data/model release. Do not promote a terminal-log-derived summary
to independently reproduced status just because software tests pass.
