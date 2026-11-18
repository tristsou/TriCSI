# Include S7 and compare with the historical pipeline

Use this after installing the diagnostics add-on. Extract this ZIP into
`C:\ESP32-CSI-Tool\tricsi_localization`, merging its `experiments` directory.
Keep all existing recordings, results, caches and saved models.

S7 must use compatible two-link recordings and the same physical meanings for
Z1-Z9. Different room conditions alone do not make a source session ineligible.
Check the collection notes for changed grid locations or swapped device/link
identities. Software can check metadata, but cannot establish where someone stood.
S7 has not been inspected by the author of this add-on; preparing it is the next
local check. Stop and inspect a failed audit rather than relabeling files.

## 1. Add only S7 to the existing cache

From PowerShell in the project folder:

```powershell
cd C:\ESP32-CSI-Tool\tricsi_localization

.\.venv\Scripts\python.exe -m experiments.session_robustness prepare `
  --raw-dir .\data\raw --sessions S7 `
  --cache-dir .\outputs\session_cache_v1
```

This preserves S4-S6/S8/S9 caches. If S7.npz and S7.json already exist from a
successful preparation, skip this command. An existing partial cache needs
inspection; do not blindly delete it. The audit checks CSV identities and that
each recording produces usable windows, not physical compatibility.

## 2. Reconstruct the original training pipeline

```powershell
.\.venv\Scripts\python.exe -m experiments.session_diagnostics.historical `
  --cache-dir .\outputs\session_cache_v1 `
  --epochs 70 --seed 26082026 --batch-size 128 --lr 0.001 `
  --output .\results\historical_s4_s7_reconstruction_v1
```

This calls the original source-preparation and training functions directly.
S4-S7 use all source Empty recordings for their background reference, and Empty
trial 1 is included in source training, matching the original script. The original
augmentation, shuffled loader and dropped final incomplete batch are retained.
Only the final epoch is scored; S8/S9 do not select this checkpoint. Source-only
prototypes and target Empty trial 1 reproduce the confirmatory prediction rule.
S8 excludes occupied trials 1-2; S9 keeps all occupied trials. Both exclude Empty
trial 1 from scoring. Expect 40 scored trials per session with the known files.

The command uses the original script's DEFAULT settings. The actual historical
run seed/epochs have not been verified. This is a reconstruction, not a claim
to recreate the exact 95% weights. If the original run record establishes different
settings, supply those explicitly and use a new output directory. The original
frozen checkpoint remains the reference for the reported 95%/52.5% result.
The script saves a separate checkpoint and never overwrites that original.

## 3. Test the current baseline and relative-shape representation with S7

```powershell
.\.venv\Scripts\python.exe -m experiments.session_diagnostics compare `
  --cache-dir .\outputs\session_cache_v1 `
  --train S4 S5 S6 S7 --dev S8 S9 `
  --exclude-dev-occupied-by-session "S8:1,2" `
  --epochs 150 --eval-every 5 `
  --seeds 2026 2027 2028 `
  --output .\results\background_shape_s4_s7_v1
```

This runs six fits: current legacy shape and Empty-relative shape, each with the
same three seeds. Both use all eligible S4-S7 trials, existing augmentation,
dedicated source Empty calibration, and identical training budgets. One checkpoint
per fit is chosen jointly across S8/S9 by weaker-session accuracy, then mean
accuracy and macro-F1. It changes only the representation between the two arms.
This does not run the empirical session generator; establish the baseline first.

Comparisons have different meanings:

| Comparison | What can be learned |
| --- | --- |
| Historical reconstruction vs reported frozen checkpoint | Whether script-default reconstruction reproduces the historical behavior; mismatch does not identify its cause |
| Historical reconstruction vs current legacy S4-S7 | Combined effect of calibration, sampler, seed, training duration and checkpoint selection; not a one-factor experiment |
| Current legacy S4-S6 vs S4-S7, same seed/settings | Effect of adding S7 under the current pipeline; number of updates per epoch changes with dataset size |
| Current legacy vs relative shape, both S4-S7 | Controlled representation comparison with matched updates and seeds |

Keep S4-S6 results, report all seeds and both sessions, and do not select which
source set to report solely from its highest score. S8/S9 have already informed
development, so none of these new scores are fresh independent test evidence.
No new collection or reduction of the available training data is required here.

## Share results

Upload `reconstruction_metrics.csv` and `run_config.json` from the reconstruction,
plus `comparison_metrics.csv` and `learning_curves.csv` from the S4-S7 comparison.
Keep predictions and checkpoints locally for follow-up diagnostics.

## Software validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\experiments\session_diagnostics\tests -v
```

Synthetic tests verify tensor-identical weights and normalization versus calling
the original trainer directly, historical source-trial inclusion, 40/40 scoring,
no cache mutation, invalid-configuration rejection and output overwrite rejection.
They do not establish real accuracy or physical compatibility of S7.
