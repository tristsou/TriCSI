# Availability and attribution

## Present

Research source, original parser/network implementations, matched experiment
code, schema/split documentation, log-derived result summaries, migration hashes
and synthetic software checks.

## Still needed for the full reproducibility release

- Actual raw recordings, original frozen models and full result tables/receipts.
- The original working environment records, not a newly invented package lock.
- Setup records identifying physical zones, device/antenna positions and changes.
- Actual transmitter mapping matching the recorded three-link MACs.
- Full ESP32 firmware source/build settings and the upstream revision used.
  The recovered packet-timing note in `archive/starter` is not the full firmware.
- Confirmation of historical seed/epoch settings when available.

The architecture/code can be inspected now, but the historical scores cannot be
independently validated from the public code alone. The owner supplied the
reported scores; the ledger distinguishes terminal-log evidence from artifacts.

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
