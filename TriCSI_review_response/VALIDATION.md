# Validation and limits

Checked October 28, 2026.

- All supplied Python files compile; each new command-line entry point displays help.
- The reporting script reads the student's actual saved S3 prediction file and reproduces 35/50 correct for the outer pair, 42/50 for all three links, seven corrected trials, and zero harmed trials. This is a check of saved predictions, not a rerun on raw CSI.
- Feature selection was checked against the original module's prefixed names: RSSI-only selects the three RSSI summaries per link and CSI-without-RSSI selects the other 159 features.
- A complete synthetic 50-trial, three-link acquisition fixture was generated solely for software checks. Both development-transfer directions, both link subsets, and all four baseline conditions completed with one seed. No synthetic scores are scientific results or included in the deliverable.
- A synthetic frozen model and separate synthetic session exercised extraction, prediction, result saving, and reporting. Training-data overlap, a wrong model hash, and an existing output folder were rejected.
- The shared tricsi_3tx_modeling.py is copied from the original modeling bundle without feature changes.
- Full neural training was NOT run. PyTorch is absent from this workspace, and the raw S4-S7 trial CSVs were not uploaded. The matched-seed runner requires the original neural modules and their existing working PyTorch environment on the student's PC. Its code was syntax-checked and reviewed against the original functions; it has not been validated through actual training here.
- No real-data baseline fits, new-session collection, new-session evaluation, or published-method reproduction was completed here. The archive has summary outputs and source code, but no raw trial CSVs.
- This toolkit does not establish a new methodological contribution, confirm a future-session advantage, or certify STS authorship compliance.
