# Results ledger

[reported_results.json](reported_results.json) separates owner-reported historical
scores from results parsed from a supplied training log. The original raw data,
checkpoint hashes and full metrics/prediction files have not yet been published.
No summary here is presented as an independently rerun scientific result.

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
