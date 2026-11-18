# Validation record

Software verification completed October 30, 2026. No new participant recordings were collected here.

## Verified with the actual supplied files

- Inspected the original feature extractor, trainer/freezer, frozen evaluator, collector, recent baseline script, archived frozen-bundle metadata, and relevant two-link evaluator source.
- The archived original and recent review-response feature modules have identical bytes (SHA256 02e7914f39394498bef4915fba73f96f26e9daa2646443fa4db085ac29935e81).
- The saved S3 prediction CSV reproduces 35/50 and 42/50 correct, seven corrected errors, and zero newly introduced errors.
- Calling the original paired bootstrap on those saved predictions reproduces +0.14 with interval [0.06, 0.22]. This is verification of stored predictions, not raw-data inference.
- The archived original estimators record outer-pair seed 2029 and three-link seed 2032 and scikit-learn 1.9.0.

## Verified with synthetic fixtures only

- Realistic 160-column CSVs generated with the unchanged collector's exact header, MAC mapping, grid, and trial schedule definitions.
- Valid declared len=128/256/384 values accepted without comparing them to saved IQ count.
- Invalid unused len text logged without changing feature selection; a separate nonnumeric IQ row logged and excluded by the unchanged original extractor.
- Full-feature matrices are exactly equal to original combine_links output; CSI and RSSI dimensions checked.
- Inspect CLI, prospective six-model fitting/freezing with 400 trees each, per-model serialization, consistent seed 2026, and unchanged original model bytes.
- Collection wrapper exercised with a mocked hardware subprocess; supported original collector flags, receipts, and audits executed. No actual serial board was contacted.
- Both new synthetic sessions evaluated with six matched-seed plus two historical models; 400 prediction rows/session, no refitting.
- Report creates 800 combined trial predictions, 21 scoped planned contrasts, class/orientation tables, paired score intervals, descriptive session variability, and figures.
- Original historical accuracy mismatch stops with diagnostics rather than retuning the synthetic bundle to reach 70/84.
- Existing evaluation outputs, reusing already-recorded sessions as prospective data, and altered frozen-model bytes are rejected.
- Optional S8/S9 inventory exercised with synthetic two-link files; both inspected original scoring layouts select 40 trials.
- Paired accuracy and difference-in-differences arithmetic checked with constructed prediction controls based on saved prediction rows; these controls are not new experimental observations.
- Plots rendered to PNG/PDF/SVG and visually inspected for label/layout legibility using synthetic outputs. Those synthetic plots/data are NOT in the deliverable ZIP.

## Not verified here

- Actual raw S1/S2/S3 or new S4/S5 feature extraction/inference, because the archive contains results/source code but not those trial CSVs.
- Physical serial acquisition or unchanged room/hardware geometry.
- Actual S3 row 418; its fields must be inspected by the local audit.
- Historical inference under the original model's software version. This verification runtime has scikit-learn 1.8.0 and the archived model records 1.9.0. The released CLI stops on this mismatch, rather than treating a cross-version load as validated.
- Neural S8/S9 prototype/normalization equivalence; original raw data/checkpoint are unavailable here and the optional S8/S9 script is an acquisition inventory, not neural inference.
- A direct published-method benchmark, scientific novelty, clinical monitoring capability, or STS authorship compliance.

Run the actual inspection and historical reproduction gates on the PC before freezing new controls. The README documents the exact sequence and the remaining local checkpoints.
