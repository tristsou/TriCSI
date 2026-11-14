# Zone-confusion diagnostics and the background-shape experiment

This is an additive package. It does not edit the original collection, training,
calibration or evaluation scripts, or the `experiments/session_robustness` package.
Use the existing version-3 environment and caches. No new recordings are needed
to begin. S7 is not added to the source set. Existing models and raw data are never
overwritten. Every output directory must be new.

Implementation and documentation were prepared with AI assistance. Synthetic
software fixtures establish implementation behavior, not research accuracy.

## Install the add-on

Extract the ZIP into `C:\ESP32-CSI-Tool\tricsi_localization`. The result must be
`experiments\session_diagnostics\__main__.py` inside that project.

Dependencies are those already used by the session-robustness experiment:
NumPy, pandas, SciPy, scikit-learn, matplotlib and PyTorch. Keep the installed
versions used for your existing experiment. Do not update packages just to run
these commands. All inference uses CPU and a configurable thread count.

## 1. Diagnose the saved 150-epoch run before another training run

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m experiments.session_diagnostics audit `
  --development-run .\results\session_robustness_150epoch_selection_v1 `
  --cache-dir .\outputs\session_cache_v1 `
  --raw-dir .\data\raw `
  --output .\results\zone_confusion_diagnostic_v1
```

The directory supplied to `--development-run` must contain `run_config.json`,
`checkpoint_index.json`, `development_predictions.csv`, `completed.json` and
its `checkpoints` folder. Use the actual directory name if yours differs.
The audit reads the source/dev session lists and trial exclusions from that run;
it does not guess them or apply S8's exclusions to S9.

The default neural diagnostic is the `existing` augmentation control. To inspect
all four saved methods, add `--methods none existing generic session`. The
measured-signal diagnostic uses the original cache arrays in either case.
The old experiment code, cache files and selected checkpoints must match their
recorded hashes. An altered file stops the audit; it is not silently accepted.

The audit performs no neural training, generator fitting, target normalization,
occupied-target calibration, label correction or data exclusion based on accuracy.
It reproduces saved predictions before marking its receipt complete.
After completion, share `diagnostic_results.zip` from the output directory. This
contains the reports and plots, without your raw recordings, caches or weights.

### Files to inspect

- `raw_metadata_audit.csv`: CSV structure, row metadata, duplicate/missing trial
  identities and hashes compared with the cache's source manifest where available.
  This can identify file inconsistencies. It cannot prove where the participant
  stood or whether the physical grid was moved. Verify that from collection notes.
- `trial_metadata.csv`: session, label, orientation, source filename, scored status,
  window count and minimum packet counts. These windows overlap; they are not
  independent trials or recording days.
- `csi_raw_amplitude.png` and `csi_empty_relative.png`: each link's retained-bin
  profiles for source Zone 5/8 and each development session's Zone 5/8.
- `csi_legacy_shape.png` and `csi_relative_shape.png`: the shape channels before
  and after the proposed representation change. These are signal visualizations,
  not predictions from the old model with altered inputs.
- `signal_profiles.csv`: per-trial values behind those plots. Session plots use
  medians of whole-trial profiles, so long recordings do not dominate.
- `signal_distances.csv`: source-standardized feature distances to source Zone 5
  and Zone 8 references. Source variation alone sets the scale. Values are not
  meters, probabilities, or evidence of a physical cause.
- `existing_strength0_seed2026_trial_probabilities.csv`: exact whole-trial
  predictions, occupancy probability, all nine conditional zone probabilities,
  and cosine distances from trial embedding centroids to source zone prototypes.
  The filename changes with the selected method, strength and seed.
- `*_source_probabilities.csv` and `*_trial_embeddings.npz`: source/target trial
  summaries and source prototypes for further inspection. Source predictions
  are training-data diagnostics, not validation accuracy.
- `*_embeddings.png`: a 2D PCA view fitted only on occupied source trial centroids.
  Projection can hide separation. Use full-dimensional distances, too. Stars are
  the saved source prototypes. Predicted labels still use the original mean of
  per-window probabilities, not nearest centroid in this 2D plot.
- `diagnostic_receipt.json`: input hashes, exclusions, no-fitting declaration,
  completion status and whether saved predictions were reproduced.

Interpretation: If the measured background-relative Zone 5 profiles resemble
source Zone 8, the shift precedes the network, although its physical cause remains
unknown. If input profiles appear distinguishable but neural representations
confuse them, inspect the learned representation and source prototypes. Neither
visual observation alone establishes causation. Do not relabel Zone 5 as Zone 8
because a model predicts it.

## 2. One-factor training comparison

Only after inspecting the audit, compare two representations using the same
existing augmentation, architecture, real source trials and training schedule:

```powershell
.\.venv\Scripts\python.exe -m experiments.session_diagnostics compare `
  --cache-dir .\outputs\session_cache_v1 `
  --train S4 S5 S6 `
  --dev S8 S9 `
  --exclude-dev-occupied-by-session "S8:1,2" `
  --epochs 150 --eval-every 5 `
  --seeds 2026 2027 2028 `
  --output .\results\background_shape_comparison_v1
```

There are six fits: two representations x three seeds. Runtime will depend on
the PC. It need not finish before sharing the read-only audit. The maximum
training budget is identical in both arms. Every eligible real source trial is
used; there is no training-data reduction or new synthetic generator in this test.

For each link let A(t,k) denote its measured log-amplitude feature and B(k) the
median profile from dedicated Empty trial 1. Both arms retain R(t,k)=A(t,k)-B(k).

- `legacy`: shape = A(t,k) - median_k A(t,k), as in the existing experiment.
- `relative_shape`: shape = R(t,k) - median_k R(t,k).

The old shape channel retains a frequency-dependent background pattern even
though its common level has been removed. The new shape channel centers the
Empty-relative signal. RSSI preparation, dedicated Empty reference separation,
source normalization procedure, original loss, model/link dropout, minibatch order,
initialization seed, trial scoring and update budget are matched. Normalization
values are refitted on each representation's real source data only. No target
labels enter either fit. A separate model is trained from scratch for each arm;
new inputs are never substituted into the old frozen model.

Both arms run all 150 epochs. Within each arm and seed, one checkpoint is selected
jointly for S8 and S9: highest lower-session accuracy, then mean accuracy, then
macro-F1; an exact tie retains the earlier epoch. Development labels therefore
select checkpoints. These are development results, not fresh held-out evidence.
Inspect both sessions, all seeds, per-zone errors and occupied accuracy, rather
than reporting only the highest score.

Outputs: `comparison_metrics.csv`, `comparison_predictions.csv` (including all
nine conditional zone probabilities), `learning_curves.csv`,
`comparison_summary.csv`, fitted checkpoints, input/code hashes and a completion
receipt. The `legacy` arm should reproduce the previous existing-augmentation
condition when all settings, code, caches and software environment match.

A result should support the proposed representation only if the matched comparison
improves real cross-session performance across seeds without simply trading away
one session's accuracy. The experiment may also show that this representation
does not help; that is an informative result and should be preserved.

## Optional prediction-only summary

This needs no PyTorch, caches or checkpoint files:

```powershell
.\.venv\Scripts\python.exe -m experiments.session_diagnostics predictions `
  --predictions .\results\session_robustness_150epoch_selection_v1\development_predictions.csv `
  --output .\results\prediction_error_summary_v1
```

It exports errors by zone/orientation, common failures and an exploratory vote
of `existing`, `generic` and `session`. Three-way ties go to `existing`. It rejects
mismatched trial identities and multiple strengths per method rather than merging
them. The vote uses saved class labels, not probability averaging.

## Software checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\experiments\session_diagnostics\tests -v
```

Synthetic fixtures check the intended static-background invariance and input
non-mutation, exclusion of Empty calibration, vote alignment, duplicate/mismatched
prediction rejection, complete frozen-inference diagnostics, unchanged cache and
weight hashes, matching optimizer update counts and a tensor-identical legacy
baseline relative to the prior implementation. These checks do not test whether
real CSI remains informative after the new representation.

## Related research

BE-Fi motivates investigating background effects in device-free localization:
https://doi.org/10.1016/j.phycom.2026.103169 . This add-on is a simple controlled
log-amplitude representation experiment; it does not reproduce BE-Fi or claim
that centering an amplitude residual isolates human scattering physically.
MixStyle is a possible later feature-augmentation comparison:
https://github.com/KaiyangZhou/mixstyle-release . It is not implemented here.
