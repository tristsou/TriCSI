# TriCSI

TriCSI is a Wi-Fi channel state information (CSI) research project using ESP32-S3 boards to classify stationary occupancy in a nine-zone indoor grid and distinguish it from an empty room. The repository collects the available project source, including earlier experiments and analysis utilities.

Start in [`workflows/`](workflows/). The scripts there have descriptive names and run the original source with the same arguments, so older commands and source verification checks still work.

| Task | Script | Guide |
| --- | --- | --- |
| Collect two-link development trials | [`collect_two_link_anchor_trials.py`](workflows/collect_two_link_anchor_trials.py) | [Two-link Anchor notes](docs/TRICSI_ANCHOR_FINAL_PLAN.md) |
| Collect two-link confirmatory trials | [`collect_two_link_confirmatory_trials.py`](workflows/collect_two_link_confirmatory_trials.py) | [Two-link Anchor notes](docs/TRICSI_ANCHOR_FINAL_PLAN.md) |
| Train the two-link model | [`train_two_link_anchor_model.py`](workflows/train_two_link_anchor_model.py) | [Two-link Anchor notes](docs/TRICSI_ANCHOR_FINAL_PLAN.md) |
| Evaluate the saved two-link model | [`evaluate_two_link_anchor_model.py`](workflows/evaluate_two_link_anchor_model.py) | [Two-link Anchor notes](docs/TRICSI_ANCHOR_FINAL_PLAN.md) |
| Collect trials from three Wi-Fi links | [`collect_three_link_trials.py`](workflows/collect_three_link_trials.py) | [Three-link modeling](tricsi_3tx_modeling_bundle/README_3TX_MODELING.md) |
| Train and compare link combinations | [`train_and_compare_three_link_subsets.py`](workflows/train_and_compare_three_link_subsets.py) | [Three-link modeling](tricsi_3tx_modeling_bundle/README_3TX_MODELING.md) |
| Evaluate the saved three-link models | [`evaluate_frozen_three_link_models.py`](workflows/evaluate_frozen_three_link_models.py) | [Three-link modeling](tricsi_3tx_modeling_bundle/README_3TX_MODELING.md) |
| Compare signal inputs and baseline classifiers | [`compare_three_link_feature_baselines.py`](workflows/compare_three_link_feature_baselines.py) | [Development controls](TriCSI_review_response/README.md) |
| Compare two-link model components | [`compare_two_link_model_components.py`](workflows/compare_two_link_model_components.py) or [matched seeds](workflows/compare_two_link_components_matched_seeds.py) | [Development controls](TriCSI_review_response/README.md) |
| Run the six-model follow-up | [`run_six_model_followup_study.py`](workflows/run_six_model_followup_study.py) | [Six-model study](TriCSI_six_model_study/README.md) |
| Audit two-link test sessions | [`audit_two_link_test_sessions.py`](workflows/audit_two_link_test_sessions.py) | [Six-model study](TriCSI_six_model_study/README.md) |
| Generate paper figures from saved results | [`generate_paper_figures.py`](workflows/generate_paper_figures.py) | [Figure inputs and instructions](figures/verified/README_START_HERE.md) |

Other source directories cover [serial capture and live display](python_utils/), the [one-day experiment workflow](TriCSI_One_Day_Code/README_FIRST.md), and [illustrative examples](examples/).

### Running the code

Run commands from the repository root. For example, use `python workflows/evaluate_frozen_three_link_models.py --help` to see the evaluator's options. These scripts came from separate experiments and have different dependencies and input formats. The linked guides list the setup instructions where available. Before collecting data, review the serial port, baud rate, and transmitter mappings in the selected collector and its configuration.

Collection requires compatible ESP32 firmware that emits CSI data. The full ESP32 firmware tree was not in the recovered sources; a packet-timing patch reference is in `archive/starter/`. Some historical scripts assume a Windows project directory and expect existing dataset, checkpoint, or result files. Supply those locally before running them. Raw captures, fitted checkpoints, and generated outputs are not part of this source export. The example with hypothetical accuracies is illustrative and does not contain experimental evidence.

`archive/` preserves earlier packages and distinct versions found in the sources. [SOURCE_MANIFEST.tsv](SOURCE_MANIFEST.tsv) records where each imported file came from and its SHA-256 digest. The active one-day workflow combines the v2 package with the v3 parser patch; the separate versions are preserved in the archive.

### Hardware addresses in the public source

Transmitter MAC addresses from the original setup have been replaced with consistent `02:00:00:00:00:xx` placeholders. Set the addresses in the collector and its configuration before collecting data. For an existing dataset, configure its corresponding transmitter addresses locally; workflows that compare against reference copies require the same mapping in both copies. Keep those local edits and raw captures outside public commits. The source manifest and reference hashes describe this public copy.
