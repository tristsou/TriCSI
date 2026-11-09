# TriCSI-Adapt: Frozen Final Research Direction

## One-sentence project

**Cross-session device-free indoor localization using two low-cost ESP32-S3 CSI links, background-relative CSI, spatially aware multi-link metric learning, and one-shot target-session prototype calibration.**

## Why this replaces the V2-V6 diagnostic loop

The existing work already established the important engineering facts:

- the hardware and packet rates are healthy;
- fixed geometry matters;
- S4/S5 cross-session performance became much stronger after physical standardization;
- remaining live failures are systematic fingerprint/domain-shift failures, not serial lag;
- temporal filtering can stabilize noisy predictions but cannot repair a consistently wrong fingerprint;
- fixed/adaptive LEFT/RIGHT probability fusion gives only marginal improvement.

The final model therefore moves upstream: **make the CSI representation itself more session-invariant and permit a very small amount of explicit target-session calibration.**

## Research gap

Recent CSI literature repeatedly identifies temporal/domain generalization as a major barrier to deployment. Recent ESP32 DFL work demonstrates that low-cost multi-link hardware can localize people, but current ESP32 studies are still largely controlled/single-environment and explicitly leave domain adaptation as future work. Few-shot CSI research is dominated by HAR rather than device-free localization.

TriCSI-Adapt targets the intersection:

1. low-cost ESP32;
2. device-free localization;
3. multiple spatial viewpoints;
4. cross-session temporal drift;
5. very low calibration burden;
6. reproducible evaluation with whole-session/whole-trial splits.

## Why LEFT is not simply "the left-column sensor"

With the standardized geometry, the direct LEFT TX→RX ray passes very close to
Z7 and Z4, and the RIGHT ray passes very close to Z9 and Z6. So the intuition is
partly correct.

It is not a hard rule because:

- both rays converge near the RX, making upper/center positions visible to both;
- a person's effect is governed by Fresnel zones and multipath, not nearest-device distance;
- reflected paths from walls/furniture can be more discriminative than the direct ray;
- which link is best can change between sessions;
- current clean live data already showed cases where RIGHT outperformed LEFT at Z7.

So geometry should be a **soft structural prior**, not a rule such as
"LEFT decides Z1/Z4/Z7."

## Final architecture

```text
LEFT CSI                             RIGHT CSI
  │                                     │
  ├─ subtract session EMPTY baseline    ├─ subtract session EMPTY baseline
  │                                     │
  ▼                                     ▼
LEFT 2-channel CSI image             RIGHT 2-channel CSI image
(abs residual + packet-centered)     (abs residual + packet-centered)
  │                                     │
  ▼                                     ▼
small LEFT CNN                       small RIGHT CNN
  │                                     │
  └──────────────┬──────────────────────┘
                 │
          sample-wise link gate
                 │
                 ▼
          fused 64-D embedding
                 │
     ┌───────────┼──────────────┐
     │           │              │
 zone CE     occupancy BCE    coordinate loss
     │
 spatial cross-session metric loss
     │
     ▼
session-invariant spatial embedding
```

At a new session the neural model is frozen.

```text
10–12 s EMPTY
      +
one 8 s calibration trial / zone
      ↓
9 target-session prototypes
      ↓
cosine-nearest prototype localization
```

There is **no target-session gradient retraining**.

## Final S7 protocol

`collect_s7_tricsi_adapt.py` creates one new standardized session:

- EMPTY trial01: background calibration only;
- Z1–Z9 trial01: prototype calibration only, facing N;
- trials02–05: untouched final evaluation;
- evaluation orientations: E, S, W, N;
- randomized evaluation order;
- 5 s move/settle countdown after ENTER;
- 8 s recording per occupied trial.

The final evaluation therefore answers both:

- **zero-shot:** can S4–S6 generalize directly?
- **one-shot:** how much does one short calibration example per location recover?

## Commands

First, optional one-time sanity check on existing S4–S6:

```cmd
python train_tricsi_adapt_final.py --mode loso
```

Do not tune from this repeatedly.

Then collect the single final session:

```cmd
python collect_s7_tricsi_adapt.py
```

Then run the frozen final evaluation:

```cmd
python train_tricsi_adapt_final.py --mode final --test-session S7
```

## Primary paper metrics

Report:

- exact whole-trial zone accuracy;
- macro F1;
- exact-or-adjacent occupied accuracy;
- median coordinate error;
- P90 coordinate error;
- zero-shot vs one-shot calibration gain;
- per-zone recall;
- learned mean LEFT/RIGHT gate weight by zone/session;
- inference latency;
- calibration burden in seconds.

## Core baselines

Keep the comparison small:

1. plain all-feature RF;
2. existing fixed two-link RF ensemble;
3. TriCSI-Adapt zero-shot;
4. TriCSI-Adapt one-shot.

That is enough. Do not create another long chain of fusion/filter variants.

## Scientific claim to aim for

The strong claim is **not** "we get perfect 3×3 localization."

It is:

> A reproducible two-link ESP32-S3 device-free localization system can substantially reduce temporal/session fingerprint failure using background-relative CSI and spatial metric learning, while one short labeled calibration trial per grid point adapts the localization map without retraining the neural encoder.

That directly addresses the practical gap exposed by the live experiments.
