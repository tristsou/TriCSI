# Improve TriCSI

Start with [overview](docs/OVERVIEW.md), [architecture](docs/ARCHITECTURE.md) and
[data/splits](docs/DATA.md). See [availability](docs/AVAILABILITY.md) for missing
assets and current licensing status.

1. Open an issue describing the specific failure and a measurable hypothesis.
2. Branch from a recorded code commit. Preserve original code/data/checkpoints.
3. Add the candidate in `experiments/`, with a matched baseline and an explicit
   recipe. Keep windows grouped by trial and session roles disjoint.
4. Test meaningful behavior: calibration exclusion, label preservation assumptions,
   matching updates, frozen inference or failure on malformed inputs as relevant.
5. Run the three test commands in [setup](docs/SETUP.md). Report software checks
   separately from real-data results.
6. Report every declared seed and session, trial counts, source/development roles,
   checkpoint selection and negative results. Save the actual predictions/config.
7. In the PR, explain the concrete behavior change, validation and limitations.

Do not tune on final-session labels, silently relabel zones, remove difficult
valid sessions or call overlapping windows independent observations. Since S8/S9
have already informed extension development, reserve fresh sessions before making
a new independent generalization claim. Model weights loaded by PyTorch/joblib
should come from a source you trust; those formats can execute code in some loaders.

Change preserved engine files only deliberately in a new version. Old run hashes
must continue to identify the old code; do not patch old receipts to hide changes.
See [publishing](docs/PUBLISHING.md) to supply artifacts that let another researcher
check your result instead of relying on a screenshot or one maximum accuracy.
