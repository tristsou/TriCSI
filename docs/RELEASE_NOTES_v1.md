# TriCSI reproducibility bundle v1

Published asset: [TriCSI-reproduction-v1.zip](https://github.com/tristsou/TriCSI/releases/tag/reproducibility-v1)

- Archive size: 246,014,376 bytes.
- Manifest files: 2,018; all hashes verified after receipt.
- SHA-256: `8a394293c8a52ab53acbcdf00124c7eab5616828d166db3c18c4aa4f9653661d`
- Status: published; GitHub reports matching upload size and SHA-256.
- Release tag: `reproducibility-v1`; code commit: `86d91a54b88160c81928c3c6daea00e98374425f`.

## Contents

`project/` preserves the owner's exact working code, two-link S1–S9 recordings,
available three-link sessions, prepared caches, original fitted models,
development/evaluation outputs and prediction tables. It is the working snapshot's
layout, which differs from the reorganized public checkout. Top-level environment
records and the per-file manifest support restoration and integrity checks.
No virtual environment, Git credentials/history or redundant ZIP files are included.

## Saved-model reproduction

| Model / session | Correct | Accuracy | Stored trial labels matched? |
| --- | --- | --- | --- |
| Original two-link / S8 | 38/40 | 95% | All 40 |
| Original two-link / S9 | 21/40 | 52.5% | All 40 |
| Outer pair / 3TX_S3 | 35/50 | 70% | All 50 |
| All three links / 3TX_S3 | 42/50 | 84% | All 50 |

No fitting or label changes were performed. The checker used Linux/Python 3.12.14,
CPU PyTorch 2.12.1 and scikit-learn 1.9.0. The owner's captured environment is
Windows/Python 3.14.6. Tiny numerical probability differences did not change labels.
This reproduces evaluation of saved models on the same historical recordings,
not the training trajectory or new independent-session accuracy.

## Restore

Verify with `python -m tricsi verify-bundle PATH.zip` before extraction into a new
directory. Follow `docs/PUBLISHING.md` and `docs/REPRODUCE.md` for environment,
snapshot paths, link mapping and exact evaluation commands. Raw source MAC mappings
are retained in the snapshot. Public source engines use placeholders; do not
substitute them without the documented runtime mapping.

## Remaining limitations

The complete firmware build and physical setup records are not in this bundle.
The current environment capture does not prove all older training environments.
S8/S9 have informed extension development, so new selected scores on them are
development results. No code/data license has yet been selected by the owner;
this release does not invent a reuse grant.
