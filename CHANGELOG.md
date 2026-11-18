# Changes

## Repository organization — 2026-11-17

- Added one supported CLI, six explicit recipes, setup/reproduction/data guides,
  architecture and contribution instructions, and a result ledger.
- Moved 100 historical utility/package files into `legacy` without changing bytes;
  retained the root engine modules required by historical hashes/imports.
- Added a migration map and exact hashes for the preserved public engines.
- Added a runtime three-link transmitter map so recorded MACs need not be edited
  into protected source files.
- Added byte-preserving research bundles with environment records and SHA-256
  verification, plus CI on synthetic fixtures.
- Kept raw data/models marked unpublished; no download URL or historical model
  hash is invented. Existing owner backups are not modified by this change.
