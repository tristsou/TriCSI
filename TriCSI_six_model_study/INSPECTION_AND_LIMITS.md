# Inspected pipeline and scope

Source archive: TriCSI_paper_files_20260915_213415.zip supplied earlier in this conversation; original modeling bundle also recovered. The current turn attached a context TXT, not new program bytes. This package uses those actual previously supplied sources and verifies the extractor against the PC copy before proceeding. No reconstructed TriCSI_One_Day_Code parser is used.

## Original three-link behavior

The original module reads structured *_trial*.csv files with pandas. Schedule CSVs are not selected. It selects packets by source MAC, sorts by host_time_s when present (otherwise local_timestamp), reads iq_000 through iq_127, and excludes nonfinite saved IQ/RSSI rows. It does not use the declared len field. The collector truncates each serial CSI vector to the first 128 saved values.

Those 128 integers represent 64 complex values. Magnitudes use hypot(real, imaginary), then log1p. The retained indices are 6-31 and 33-58 inclusive: 52 complex subcarriers. Each trial produces seven evenly spaced windows of 64 packets per link, aligned by fractional progress through each link's recording, not exact timestamp-level packet synchronization. Windows overlap and are correlated; trial aggregation and trial/session splits are therefore essential.

Per packet, the median log amplitude across active subcarriers is the gain summary. Subtracting it leaves spectral shape. Each of 52 subcarriers supplies temporal median, interquartile range, and mean absolute consecutive-packet difference: 156 values. Gain supplies the same three summaries, and RSSI supplies three more. Full features are 162 per link; CSI without RSSI is 159; RSSI alone is three. The CSI-only condition retains packet gain and is not pure spectral shape. No phase feature is used. Motion summaries are per-packet differences, not validated gait measures.

The classifier is ExtraTreesClassifier: 400 trees, max_features=sqrt, min_samples_leaf=2, class_weight=balanced, fixed random_state, n_jobs=-1; other defaults are recorded. The three-link pipeline does not fit a target-session standardizer or subtract an EMPTY reference. EMPTY is one of its ten supervised classes; all 50 trials are scored. This differs from the two-link neural EMPTY-calibration protocol.

Class decisions use mean predicted class probabilities across the seven windows, then argmax. They do not use majority voting. Occupied-zone accuracy excludes true EMPTY; EMPTY recall uses actual EMPTY only. The original occupied-trial distance score assigns distances between zone centers and the diagonal penalty to occupied-to-EMPTY errors.

## Historical seed and serialization findings

The original development comparison uses seed 2026 for S1-to-S2 and 2027 for S2-to-S1. Final seeds are 2026 plus the subset enumeration index: outer pair 2029, all three 2032. The format-2 joblib bundle stores seven models, subset links, grid coordinates, training-session CSV hashes, and feature configuration. The inspected archived estimator objects record scikit-learn version 1.9.0.

The recent review-response feature module is byte-identical to the original archived module. Its baselines select feature columns after the same extraction and repeat both session directions at seeds 2026, 2027, and 2028. Their reported repeated-seed averages should not replace the historical single-seed-per-direction values as if they were the same experiment.

The six NEW controls use one common seed. The two unchanged historical models are evaluated in a separate lane, avoiding a silent substitution of the historical experiment. Keeping historical seeds unchanged preserves reproduction but does not eliminate their original seed difference. Neither lane licenses a causal physical interpretation of feature differences.

## Collector findings

collect_three_link_trials.py actually supports --session, --raw-dir, --port, --baud, --trials-per-zone, --seconds, --empty-baseline-seconds, --settle, --seed, and the documented preflight options. Its defaults implement 50 trials, 12-second first EMPTY, 8-second other trials, 5-second move/settle, and N/E/S/W/N orientations. It writes a manifest/schedule even in preflight-only mode and skips existing trial filenames on resumption. The toolkit protects historical folders from accidental default-preflight writes and records prospective collection receipts.

## What is and is not verified

The saved historical prediction table contains 35/50 outer-pair correct and 42/50 all-three correct. Seven outer-pair errors are corrected and no correct outer-pair trial becomes wrong. Running the ORIGINAL bootstrap on those saved predictions reproduces the reported +14-point estimate and +6 to +22 interval. This is verification of saved outputs, not inference from actual raw S3.

Real raw S1/S2/S3 or S8/S9 were not in the supplied project archive. The actual single S3 row described in the prompt cannot be inspected here. Its effect must be determined by the local audit, without imposing a new declared-length rule. The new audit distinguishes unused metadata damage from IQ/RSSI rejection and preserves all original bytes.

The verification runtime has scikit-learn 1.8.0 and therefore cannot certify inference equivalence for the archived 1.9.0 bundle; the released CLI stops on that mismatch. Local raw-data reproduction in the student's original environment is an explicit gate. No historical model was refitted to match its reported accuracy.

The complete workflow has synthetic integration tests, not new participant experiments. See VALIDATION.md for actual completed checks. Full serial hardware behavior, actual new-session accuracy, and neural prototype equivalence remain to be tested locally.

## Scientific bounds

Two later same-day sessions can address short-term repeatability in this setup. They do not establish general environmental, cross-day, cross-person, cross-home, gait, or fall-detection performance. New six-model choices were made after S3 was known; S3 is historical reproduction, not an untouched test of those newly chosen feature conditions. S4/S5 must genuinely follow freezing.

No direct implementation/comparison of a published domain-adaptation method is included. These controls clarify an empirical feature/link question, but do not by themselves establish a methodological advance over prior literature or guarantee acceptance by JHSS/STS.

For software persistence background: https://scikit-learn.org/stable/model_persistence.html
