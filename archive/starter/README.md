# TriCSI: from dual-link motion sensing to 3x3 human localization

This package is built for the exact hardware currently in use:

- **COM9 / left transmitter / MAC `02:00:00:00:00:01`**
- **COM10 / right transmitter / MAC `02:00:00:00:00:02`**
- **COM11 / AP + CSI receiver / 921600 baud**
- ESP32-S3 boards using Steven M. Hernandez's ESP32-CSI-Tool
- Python 3.14 on Windows

The goal of this stage is not to infer geometric distance directly. It is to
learn the room's **multi-link CSI fingerprint** at nine known floor positions,
then evaluate whether the two simultaneous wireless links contain enough
spatial information to distinguish those positions.

## Why this is the correct next experiment

Your current LEFT/RIGHT detector compresses each link's 64-subcarrier CSI
pattern into one motion score. That is useful for responsive motion sensing,
but throws away most spatial information.

This package instead keeps:

1. the full 128 raw LLTF I/Q values from every packet;
2. absolute amplitude fingerprints;
3. packet-centered spectral-shape fingerprints;
4. per-subcarrier variability/IQR;
5. adjacent-subcarrier variation;
6. adjacent-frame Euclidean CSI change;
7. RSSI;
8. cross-link differences.

The use of full CSI fingerprints is consistent with prior CSI fingerprint
localization work such as DeepFi. The emphasis on multiple wireless links is
consistent with LiFS and CoMuTe. Wi-CaL is especially relevant because it used
ESP32 nodes and features including IQR, adjacent-subcarrier differences, and
Euclidean change between adjacent CSI curves. Its long six-second windows were
appropriate for its crowd-state task; here the default is one second because
you are targeting responsive single-person localization.

## TODAY'S FAST PATH

### A. Freeze and measure the physical setup — 10 minutes

Do not move the boards after data collection starts.

Recommended geometry for a roughly 6 m x 4 m room:

```
y=4 m
                    RX/AP COM11
                     (3.0,3.8)
                         *
                       /   \
                     /       \
                   /           \
                 /               \
     COM9 TX *                       * COM10 TX
           (0.3,0.3)               (5.7,0.3)
y=0 m
```

If your current triangle is already giving strong LEFT/RIGHT separation, keep
it. Measure its actual coordinates with a tape measure and edit
`project_config.py` rather than moving it just to match this drawing.

Mount all antennas at the same approximate height (about torso height is a
reasonable starting point), keep antenna orientation identical, and secure
boards/cables so they cannot move.

Mark nine floor positions:

```
Z7          Z8          Z9
(1,3.3)     (3,3.3)     (5,3.3)

Z4          Z5          Z6
(1,2.0)     (3,2.0)     (5,2.0)

Z1          Z2          Z3
(1,0.7)     (3,0.7)     (5,0.7)
```

Use painter's tape and mark the center of each location.

### B. Check firmware — 2 minutes

Keep the transmitter timing fix that uses `pdMS_TO_TICKS()`.

For the fast first dataset, the present roughly 30-40 usable packets/s/link is
already sufficient. Do not change packet rate while collecting one dataset.

AP:
- COM11
- CSI collection ON
- LLTF-only ON
- serial 921600

TX boards:
- COM9 and COM10 powered
- CSI collection OFF
- same SSID/channel as AP

### C. Create Python environment — 3-8 minutes

From Command Prompt:

```
cd /d C:\ESP32-CSI-Tool
mkdir tricsi_localization
```

Copy all files from this package into that folder, then:

```
cd /d C:\ESP32-CSI-Tool\tricsi_localization
setup_windows.bat
```

Or manually:

```
py -3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### D. Edit actual anchor coordinates — 2 minutes

Open:

```
notepad project_config.py
```

Measure and edit:

```
ANCHORS = {
    "TX_LEFT":  (..., ...),
    "TX_RIGHT": (..., ...),
    "RX_AP":    (..., ...),
}
```

Do not change the MAC addresses.

### E. Quick dataset — about 20-30 minutes

Close ESP-IDF Monitor and every program using COM11.

Activate the environment:

```
.venv\Scripts\activate
```

Collect five independent trials for every class:

```
python collect_grid.py --session S1 --trials-per-zone 5 --seconds 8 --settle 2
```

That creates 50 trials:
- 5 x EMPTY
- 5 x each of Z1-Z9

The schedule is randomized automatically. For human trials, orientations cycle
through N/E/S/W. Follow the prompt exactly.

During each trial:
- place both feet around the center tape mark;
- face the requested direction;
- stand mostly still;
- natural breathing and slight sway are desirable;
- do not deliberately wave;
- do not touch boards/cables.

The purpose is to learn location, not an activity.

### F. Data quality check — 1 minute

```
python inspect_dataset.py
```

Repeat any trial marked:
- MISSING LINK
- LOW RATE
- IMBALANCED

For the current setup, roughly 25-40 packets/s/link is healthy.

### G. Train two baselines — usually a few minutes

```
python train_localizer.py
```

The script:
- forms 1.0-second windows every 0.25 s;
- requires both links in every window;
- keeps all overlapping windows from one physical trial together;
- holds out entire trials for testing;
- trains Random Forest and distance-weighted kNN;
- reports zone accuracy and macro-F1;
- computes probability-weighted x/y localization error;
- saves confusion matrices and feature importance.

Important: never randomly split individual windows. Overlapping windows from
one 8-second trial are highly correlated and would create leakage.

Outputs:

```
models/localizer_rf.joblib
models/localizer_knn.joblib
results/metrics.json
results/confusion_matrix_random_forest.png
results/confusion_matrix_knn.png
results/feature_importance.csv
results/feature_importance_top25.png
results/window_features.csv
results/dataset_quality.csv
```

### H. Run real-time localization

Close any program using COM11, then:

```
python live_localize.py
```

The console reports:

```
Z5               conf=0.72 xy=(3.10,2.08) ...
```

and the graph shows the estimated point in the room.

The x/y point is a probability-weighted combination of the nine zone centers.
That is an interpolation visualization, not proof of centimeter-level
continuous ranging.

Use terminal-only mode if plotting affects responsiveness:

```
python live_localize.py --no-plot
```

## How to interpret the first result

### Excellent first result
- held-out-trial zone accuracy >= 80%;
- most errors occur between adjacent zones;
- median x/y error <= roughly one grid spacing;
- EMPTY is well separated.

Then immediately collect a second session and test generalization.

### Moderate result
- 55-80% zone accuracy;
- confusion is mostly adjacent locations.

This is still useful. Add trials and inspect feature importance. A third
wireless link is likely to help.

### Weak result
- less than ~50%;
- distant zones are heavily confused;
- training accuracy is much higher than held-out-trial accuracy.

First check physical placement and trial independence rather than jumping to a
deep network.

## Stronger second experiment: cross-session generalization

After the quick S1 experiment works, wait at least 20-30 minutes or power-cycle
the system. Do **not** move the anchors.

Collect another 3-5 trials/class:

```
python collect_grid.py --session S2 --trials-per-zone 3 --seconds 8 --settle 2 --seed 123
```

Then evaluate a true session holdout:

```
python train_localizer.py --test-session S2
```

This is a much stronger scientific result than random-window accuracy.

## Third experiment: add the fourth ESP32 as a third transmitter

Do this only after quantifying the two-link baseline.

With 3 TX + 1 RX you get three spatial views while keeping:
- one receiver clock;
- one serial stream;
- simple MAC-based separation.

Then repeat the exact same S1/S2 protocol. The clean research question becomes:

> How much does localization accuracy/error improve when the number of
> spatially separated CSI links increases from two to three?

That gives you a controlled ablation rather than changing hardware and model at
the same time.

## References that motivated this design

1. Steven M. Hernandez and Eyuphan Bulut, "WiFi Sensing on the Edge: Signal
   Processing Techniques and Challenges for Real-World Systems," IEEE
   Communications Surveys & Tutorials, 2023. DOI: 10.1109/COMST.2022.3209144

2. Xuyu Wang, Lingjun Gao, Shiwen Mao, Santosh Pandey, "CSI-based
   Fingerprinting for Indoor Localization: A Deep Learning Approach,"
   IEEE Transactions on Vehicular Technology, 2017.
   DOI: 10.1109/TVT.2016.2545523

3. Ju Wang et al., "LiFS: Low Human-Effort, Device-Free Localization with
   Fine-Grained Subcarrier Information," ACM MobiCom 2016 / IEEE TMC 2018.
   DOI: 10.1145/2973750.2973776

4. Hyuckjin Choi et al., "Wi-CaL: WiFi Sensing and Machine Learning Based
   Device-Free Crowd Counting and Localization," IEEE Access, 2022.
   DOI: 10.1109/ACCESS.2022.3155812

5. Tahsina Farah Sanam and Hana Godrich, "CoMuTe: A Convolutional Neural
   Network Based Device Free Multiple Target Localization Using CSI," 2020.
   arXiv:2003.05734
