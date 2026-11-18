# Reproduction recipes

Each JSON file contains a supported CLI `command` and complete `arguments` list.
Run `python -m tricsi run configs/NAME.json --dry-run` first; remove `--dry-run`
to execute. Relative experiment paths resolve from `--project-root`, or this
checkout by default. Edit a copy to choose a fresh output directory for reruns.

| Recipe | Action |
| --- | --- |
| `prepare_two_link.json` | Audit/prepare S4–S9 once; rejects existing caches |
| `frozen_two_link.json` | Original checkpoint evaluation on S8/S9, no training |
| `historical_s4_s7.json` | Original-trainer reconstruction with script defaults |
| `background_shape_s4_s7.json` | Original vs Empty-relative shape, 150 epochs, three seeds |
| `three_link_development.json` | New three-link development fit, separate output |
| `frozen_three_link.json` | Original frozen bundle on historical 3TX_S3 |

The original three-link model path deliberately differs from the new development
output path: retraining a model does not reproduce the identity of a saved one.
See [reproduction](../docs/REPRODUCE.md) for prerequisites and expected outputs.
