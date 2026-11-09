# TriCSI paper figures and results

These figures were plotted with Python from your saved experimental CSV/JSON files. No generative image tool was used, no model was retrained, and the uploaded project files were not changed.

## Start here

Open `TriCSI_Results_and_Figures.docx`. It contains six editable results subsections, each followed by its figure and caption. This is a replacement results section and figure package, not a revision of the entire earlier paper. Figure numbers R1–R6 are temporary: renumber them after inserting your methods figures and photographs.

Use the PNG files in Word. SVG and PDF versions are also provided for vector-quality resizing. The SVG text remains editable. Keep each figure with its caption, including the definition of what is being measured.

## Which graphs to use

| File prefix | What it shows | Suggested use |
|---|---|---|
| R1_early_models | Early Random Forest/CNN performance within sessions and across sessions | Main results: explains the initial problem |
| R2_model_components | Two-link model with and without its two main components | Main results: explains the development tests |
| R3_frozen_two_link_sessions | S8 and S9 performance and correct predictions by zone | Main results: report both sessions |
| R4_three_link_comparison | All seven link subsets, during development and on final 3TX_S3 | Main results: controlled transmitter comparison |
| R5_s3_confusion_matrix | Which classes the final three-link model confused | Main results: explains the 84% result |
| R6_middle_link_gain | Accuracy gain over the outer pair, with its uncertainty interval | Main results, or combine with R4 if space is limited |
| A1_north_facing_diagnostic | North-facing diagnostic, separating window and trial scores | Optional appendix |
| A2_two_link_confusion_matrices | Full S8/S9 prediction counts | Optional appendix; R3 is the shorter main-text version |
| A3_intermediate_development_tests | Selected intermediate RF and TriCSI-Adapt tests | Optional development-history appendix |

You do not need a graph for every saved run. R1 preserves the early-model results without turning the paper into a log of every experiment. Use A1 or A3 only if you discuss those specific tests. The intermediate tests use different sessions and cannot establish a controlled improvement from one model version to the next.

## Important corrections to carry into the whole paper

- The 71.5%, 83.6%, and approximately 39% scores in the earlier draft belong to the north-facing Random Forest **window-level diagnostic**. They are not the general whole-trial results. A1 preserves them with the correct labels. R1 uses consistently aggregated whole-trial predictions.
- The early all-orientation CNN's S2 whole-trial score is 62%; do not substitute its approximately 62.6% window score without changing the metric label.
- The frozen two-link model reached 95% on S8 but 52.5% on S9. The pooled result is 73.75% (59/80). Do not describe 95% as its overall unseen-session performance, or omit S9 from the abstract/discussion merely because its result was lower.
- The two-link neural-network experiment and the three-link Extra Trees experiment use different models and recordings. The fair transmitter comparison is between link subsets of the same three-link trials, not between S8's 95% and 3TX_S3's 84%.
- Three-link development sessions are `3TX_S1` and `3TX_S2`; the final test is `3TX_S3`. They are distinct from the older two-link S1, S2, and S3. Use those prefixes in the surrounding methods text.
- The saved three-link error score uses distances between zone centers. Occupied trials predicted as EMPTY receive a fixed 1.873 m penalty; actual EMPTY trials are excluded from the average. The 0.239 m result therefore is not evidence of 24 cm continuous tracking precision.
- All five EMPTY trials were recognized correctly, but four occupied trials were incorrectly called EMPTY. "100% EMPTY recall" is not the same as perfect occupancy detection.
- The component-removal tests used different random training seeds for different variants. They are exploratory evidence, not proof that every observed difference was caused solely by removing a component.
- The paired bootstrap interval uses trials from the available recordings. Its positive S3 interval supports improvement in that session; it does not establish reliability across new rooms, people, or many future sessions.

## What you still need to photograph

Take an original close-up of your actual completed module and a wide photograph showing the receiver, three transmitters, and floor grid. Do not move the hardware just to improve the photograph if preserving the setup matters. Avoid faces, private labels, and identifying household details where practical. Add simple labels and a caption describing the actual arrangement; do not use a generated room image as evidence of the experiment.

The existing manufacturer board picture can illustrate the board type if its source and permitted use are documented. Your own module photograph is better evidence of the system you built. These result graphs do not require photographs or figures borrowed from previous studies.

## Reproduce the figures

The package includes the small source CSV/JSON files needed to rebuild the graphs, so raw packet data and saved model objects are not needed for plotting. It does not replace a full backup of your experiment.

Assuming you extract this folder into `C:\ESP32-CSI-Tool\tricsi_localization`, run these commands from that project directory:

```powershell
Set-Location C:\ESP32-CSI-Tool\tricsi_localization

.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib python-docx

.\.venv\Scripts\python.exe `
  .\TriCSI_Python_Figures_20260916\make_verified_tricsi_figures.py
```

This reads the packaged `inputs` folder and writes a new `output` folder inside the package. It does not train or evaluate a model again. Running it again replaces files in that output folder; use a different `--output` path to retain an earlier version.

To rebuild from the result files currently in your project instead:

```powershell
.\.venv\Scripts\python.exe `
  .\TriCSI_Python_Figures_20260916\make_verified_tricsi_figures.py `
  --project-root C:\ESP32-CSI-Tool\tricsi_localization `
  --output .\TriCSI_Figures_Rebuilt
```

The script expects the same result-folder names as in the uploaded archive. If a source file is missing or summaries disagree with predictions, it stops instead of silently replacing a value.

## Contents and verification

- `R*.png`, `.svg`, `.pdf`: six main figures.
- `A*.png`, `.svg`, `.pdf`: three optional figures.
- `TriCSI_Results_and_Figures.docx` and `.pdf`: results section with the main figures.
- `Results_and_Captions.md`: plain-text results and captions, including optional figures.
- `*_data.csv` and `R5_s3_confusion_counts.csv`: chart data.
- `inputs/`: the 38 source files read by the plotting script.
- `reference_code/`: copies of relevant original methods scripts, for reference rather than automatic execution.
- `verification.json`: 71 numerical reconciliation checks, input SHA-256 fingerprints, and the occupied-to-EMPTY penalty.
- `make_verified_tricsi_figures.py`: complete graph and Word-document generator.

The audit checks saved predictions against summaries and reproduces the stored paired confidence intervals. It is not an independent rerun of training or raw-data quality control. Inspect the original collection logs and research notes when describing why a session changed.

Keep the AI-use record accurate: the measurements are your experimental data; the plotting and results-drafting code was developed with ChatGPT assistance. Review the wording and include only methods and conclusions you can explain yourself. The captions record this assistance, but your paper's acknowledgment or disclosure section should also describe it accurately.
