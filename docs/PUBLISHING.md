# Publish the missing reproducibility assets

The repository provides code and reported summaries. Publication is incomplete
until recordings, original fitted models, full outputs, environment and setup
notes are available through the [asset catalog](../metadata/assets.json).

## Owner: build one bundle

Keep the original project and backup intact. From the reorganized checkout, use
the Python executable of the original working environment:

```powershell
& C:\ESP32-CSI-Tool\tricsi_localization\.venv\Scripts\python.exe -m tricsi bundle --project-root C:\ESP32-CSI-Tool\tricsi_localization --output C:\Users\ch_ts\TriCSI-reproduction-v1.zip
& C:\ESP32-CSI-Tool\tricsi_localization\.venv\Scripts\python.exe -m tricsi verify-bundle C:\Users\ch_ts\TriCSI-reproduction-v1.zip
```

Finish collection/training before packaging. The tool requires the original
two-link engine, S4–S9 recordings and named original checkpoint, copies all
eligible project files and checks for changes during each read. Its presence
checks do not establish original model identity; compare saved run hashes and
reference predictions. It includes three-link assets that are actually present,
records their sessions, and does not invent missing assets.

The ZIP contains `project/` (the exact working snapshot), `environment.json`,
`requirements-lock.txt` and a per-file `manifest.json`. Old ZIPs, virtual
environments, Git metadata, bytecode, `.env` files, key files and symlink targets
are excluded. Original recorded MACs and source mappings are preserved. The
environment is the one running this command, not automatically the environment
used months earlier. Retain any older lock/environment files already in the
project as well. No data are uploaded automatically.

## Publish and connect it to a code version

Use a versioned GitHub Release asset or an archival dataset host, rather than
committing hundreds of megabytes of generated files into ordinary Git history.
GitHub currently blocks ordinary Git files over 100 MiB and allows release
assets under 2 GiB each:
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
and https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases .

After the owner has selected a code commit and dataset/code licensing terms:

1. Upload the verified ZIP as a versioned release asset. Attach its external
   SHA-256 (printed by the tool) to the release notes. A checksum detects accidental
   change; it is not independent proof of scientific validity or authenticity.
2. Record the real asset URL, file size, SHA-256 and code commit in
   `metadata/assets.json`; replace `not_published` only after a successful upload.
3. Include original model identity/expected predictions and setup/firmware notes.
4. On a fresh directory/computer, download, verify and restore the snapshot, then
   run the frozen evaluations. Compare per-trial predictions and both sessions.
5. Mark the release independently reproduced only after that check; preserve any
   mismatch with its environment details instead of replacing the checkpoint.

## Reader: restore

Verify with `python -m tricsi verify-bundle PATH.zip` before extracting. Extract
into a new directory. `project/` is a complete snapshot of the owner's working
tree; it may have the older layout. Restore its environment using supplied records
and use its original commands for exact historical reruns. Alternatively, point
the new CLI at it with `--project-root PATH/project`; the adapter executes that
snapshot's code. Keep relative paths relative to the selected project root.
Do not merge snapshots from different runs merely because filenames match.
