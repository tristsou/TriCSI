# Research metadata

| File | Meaning |
| --- | --- |
| [assets.json](assets.json) | Published bundle URL, upload checksum, release commit and required assets |
| [sessions.json](sessions.json) | Roles, calibration and scoring rules |
| [engine_hashes.json](engine_hashes.json) | Preserved public engine bytes; not hashes of the owner's raw data/models |
| [migration.json](migration.json) | Old path → new path, with unchanged moved-file hashes |
| [link_map.example.json](link_map.example.json) | Placeholder three-link mapping; use actual recorded values locally |
| [setup_template.json](setup_template.json) | Fill from actual session/hardware/firmware notes; unknown values remain unknown |
| [software_test_environment.json](software_test_environment.json) | Environment for software checks, not historical training |
| [reproduction_bundle_manifest.json](reproduction_bundle_manifest.json) | Every path, byte count and SHA-256 in the verified owner bundle |
| [owner_environment.json](owner_environment.json) | Captured Windows working environment; not proof of every historical training environment |
| [owner_requirements_lock.txt](owner_requirements_lock.txt) | Actual pip freeze supplied in the owner bundle |

Per-run receipts remain with each result directory. Exact raw/model hashes must
come from the actual files; terminal output and remembered scores do not supply
those hashes. `local_link_map.json` is ignored by Git until publication is
deliberately coordinated with the dataset's real mapping.
