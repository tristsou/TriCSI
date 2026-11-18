# Availability and attribution

## Present

Research source, original parser/network implementations, matched experiment
code, schema/split documentation, migration hashes, synthetic checks, actual
owner-supplied metric tables, and real-data saved-model reproduction receipts.
The complete owner bundle was received and all 2,018 file checksums verified.
Its release attachment is pending; see `metadata/assets.json` for the exact
archive size/hash and download status.

## Still needed for the full reproducibility release

- Public hosting of the received recordings, original frozen models and full
  result directories in the verified bundle.
- Confirmation that the captured current working environment matches the
  environment used for each older training run. The supplied actual pip freeze
  and Windows/Python details are recorded in `metadata`.
- Setup records identifying physical zones, device/antenna positions and changes.
- Actual transmitter mapping matching the recorded three-link MACs.
- Full ESP32 firmware source/build settings and the upstream revision used.
  The recovered packet-timing note in `archive/starter` is not the full firmware.
- Confirmation of historical seed/epoch settings when available.

The public code alone still lacks the raw/model download. Using the uploaded
bundle, the original fitted models were re-evaluated without fitting and all 180
compared trial labels matched their stored predictions. This validates saved-model
reproduction on those recordings, not model retraining or a new recording study.

## License and reuse

The owner has not selected a repository code license or a dataset license here.
No open-source or data-reuse grant is invented by this cleanup. Add owner-approved
terms before labeling a release openly reusable, and preserve any applicable
third-party notices. Contributors can identify issues and propose changes while
the owner determines those terms. Use GitHub history and `SOURCE_MANIFEST.tsv`
for recovered-source provenance; the manifest does not establish authorship rights.

## Related work and assistance

The background-relative shape experiment is a controlled log-amplitude experiment,
not a reproduction of BE-Fi: https://doi.org/10.1016/j.phycom.2026.103169 .
MixStyle (https://github.com/KaiyangZhou/mixstyle-release) is a possible future
comparison, not an implemented component. Implementation/documentation include
AI assistance and require researcher review. Keep reported results tied to the
actual data and saved models, including negative findings.
