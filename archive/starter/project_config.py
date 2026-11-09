from pathlib import Path

# ---------------------------------------------------------------------------
# YOUR CURRENT HARDWARE
# ---------------------------------------------------------------------------

SERIAL_PORT = "COM11"
SERIAL_BAUD = 921600

LEFT_MAC = "02:00:00:00:00:01"   # COM9 transmitter
RIGHT_MAC = "02:00:00:00:00:02"  # COM10 transmitter

MAC_TO_LINK = {
    LEFT_MAC: "left",
    RIGHT_MAC: "right",
}

# ---------------------------------------------------------------------------
# ROOM GEOMETRY
# ---------------------------------------------------------------------------
# These are the approximate coordinates we have been using for a 6 m x 4 m
# room. MEASURE YOUR ACTUAL BOARD LOCATIONS and edit these three coordinates
# before the final research dataset. Units are meters.
#
# Coordinate convention:
#   (0, 0) = lower-left corner when looking at the room map
#   +x     = toward the right
#   +y     = toward the far wall
#
ROOM_WIDTH_M = 6.0
ROOM_HEIGHT_M = 4.0

ANCHORS = {
    "TX_LEFT":  (0.30, 0.30),
    "TX_RIGHT": (5.70, 0.30),
    "RX_AP":    (3.00, 3.80),
}

# 3 x 3 localization grid. These are zone-center coordinates.
ZONE_CENTERS = {
    "Z1": (1.0, 0.7),
    "Z2": (3.0, 0.7),
    "Z3": (5.0, 0.7),
    "Z4": (1.0, 2.0),
    "Z5": (3.0, 2.0),
    "Z6": (5.0, 2.0),
    "Z7": (1.0, 3.3),
    "Z8": (3.0, 3.3),
    "Z9": (5.0, 3.3),
}

CLASS_ORDER = ["EMPTY", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9"]
ORIENTATIONS = ["N", "E", "S", "W"]

# ---------------------------------------------------------------------------
# CSI FORMAT
# ---------------------------------------------------------------------------
# Steven Hernandez's tool prints the original ESP-IDF CSI buffer length in the
# CSV metadata. With "Collect only LLTF" enabled, the payload may nevertheless
# contain only the first 128 signed int8 values. We always preserve those first
# 128 LLTF values: 64 complex samples stored as imag, real, imag, real, ...
LLTF_RAW_COUNT = 128
LLTF_COMPLEX_COUNT = 64

# ---------------------------------------------------------------------------
# LOCALIZATION WINDOWING
# ---------------------------------------------------------------------------
# Your fixed firmware is currently producing roughly 33 packets/s per link.
# A 1.0 s window therefore contains about 33 packets/link: enough to describe
# both static CSI shape and short-term dynamics while remaining responsive.
WINDOW_SECONDS = 1.00
HOP_SECONDS = 0.25
MIN_PACKETS_PER_LINK = 12

# Live predictions are causally smoothed; this does NOT look into the future.
LIVE_PROBABILITY_ALPHA = 0.65
LIVE_MIN_CONFIDENCE = 0.35

# ---------------------------------------------------------------------------
# DATA DIRECTORIES
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for path in (RAW_DATA_DIR, MODEL_DIR, RESULTS_DIR):
    path.mkdir(parents=True, exist_ok=True)
