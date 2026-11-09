# TriCSI-Anchor: Final Research Plan After S7

## What S7 proved

S4-S6 leave-one-session-out validation was optimistic relative to a genuinely harder session. On S7, background-calibrated zero-shot exact accuracy fell to 52.5%, and the single N-facing target prototype reduced accuracy further to 42.5%.

The most revealing failures were:

- Z7: zero-shot 1.00 -> one-shot 0.00
- Z3: zero-shot 0.75 -> one-shot 0.00

That means the target prototype was not orientation invariant. A single N-facing calibration sample was being used as though it represented E/S/W as well.

S7 is therefore development data from this point onward. It may be used to train the next source model, but it is no longer a valid untouched final test.

## Final research question

Can a low-cost two-link ESP32-S3 device-free localization system adapt to a new recording session with only:

- one short EMPTY background measurement;
- two short orthogonal orientation anchors per location;
- no gradient retraining on the target session?

## Why not hard-code LEFT for Z1/Z4/Z7 and RIGHT for Z3/Z6/Z9?

That geometry is a useful intuition, but CSI is a Fresnel/multipath measurement rather than a nearest-sensor measurement. Reflections, body orientation, and session changes can change which link is more informative. S7 itself showed RIGHT sometimes outperforming LEFT in locations where a simple path-distance rule would predict the opposite.

TriCSI-Anchor therefore keeps independent LEFT/RIGHT encoders and a learned soft gate.

## Source model

Train on S4+S5+S6+S7 using:

1. session-specific EMPTY subtraction;
2. LEFT and RIGHT link-specific CNN encoders;
3. learned sample-wise link fusion;
4. zone classification;
5. occupancy classification;
6. coordinate regression;
7. spatial metric learning;
8. extra same-zone alignment across both different sessions and different orientations.

The same-zone metric term is strongest when a positive pair changes both session and orientation.

## S8 calibration budget

For each occupied zone:

- trial01: N, 4 seconds
- trial02: E, 4 seconds

Total labeled occupied calibration recording = 8 seconds per zone, or 72 seconds for all nine zones.

The encoder is frozen. The two calibration embeddings are pooled into the target-session prototype for that zone.

## S8 final evaluation

Occupied test trials are separate recordings:

- trial03: S, 8 seconds
- trial04: W, 8 seconds
- trial05: N, 8 seconds
- trial06: E, 8 seconds

EMPTY:

- trial01: background calibration only
- trials02-05: final evaluation

All evaluation trials are randomized by the collector.

## Final comparisons

Keep the paper/model comparison small:

1. plain RF baseline;
2. fixed two-link RF ensemble;
3. background-calibrated zero-shot TriCSI-Anchor;
4. two-anchor TriCSI-Anchor.

## Final commands

Place the files in `C:\ESP32-CSI-Tool\tricsi_localization\`.

Collect S8:

```cmd
python collect_s8_anchor_final.py
```

Then evaluate exactly once:

```cmd
python train_tricsi_anchor_final.py --test-session S8
```

Do not tune the architecture after seeing S8 if S8 is to remain the final held-out test.
