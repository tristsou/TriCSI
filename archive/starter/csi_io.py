from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from project_config import (
    LLTF_RAW_COUNT,
    MAC_TO_LINK,
)


IQ_COLUMNS = [f"iq_{i:03d}" for i in range(LLTF_RAW_COUNT)]


@dataclass(slots=True)
class CSIRecord:
    host_time_s: float
    device_time_s: float
    mac: str
    link: str
    rssi: int
    reported_len: int
    payload_len: int
    iq: np.ndarray


def parse_csi_line(raw_line: bytes | str) -> Optional[CSIRecord]:
    """
    Parse one Steven Hernandez ESP32-CSI-Tool AP row.

    Important:
    - Only rows from the two known transmitter MAC addresses are returned.
    - The first 128 signed CSI values are preserved as the LLTF block.
    - A row may *report* len=384 while printing only 128 values when the
      firmware's "collect only LLTF" option is enabled. That is valid.
    """
    if isinstance(raw_line, str):
        raw = raw_line.encode("utf-8", errors="replace")
    else:
        raw = raw_line

    marker = b"CSI_DATA,AP,"
    start = raw.find(marker)
    if start < 0:
        return None

    fields = raw[start:].split(b",", 25)
    if len(fields) < 26:
        return None

    try:
        mac = fields[2].strip().decode("ascii", errors="ignore").upper()
        link = MAC_TO_LINK.get(mac)
        if link is None:
            return None

        rssi = int(fields[3].strip())
        reported_len = int(fields[24].strip())

        # field 23 is the steady-clock timestamp emitted by Hernandez's tool.
        # Fallback to ESP-IDF's microsecond rx timestamp if required.
        try:
            device_time_s = float(fields[23].strip())
        except ValueError:
            device_time_s = int(fields[18].strip()) / 1_000_000.0

        payload = fields[25]
        left = payload.find(b"[")
        right = payload.rfind(b"]")
        if left < 0 or right <= left:
            return None

        values = np.fromstring(
            payload[left + 1:right].decode("ascii", errors="ignore"),
            sep=" ",
            dtype=np.int16,
        )

        if len(values) < LLTF_RAW_COUNT:
            return None

        iq = values[:LLTF_RAW_COUNT].astype(np.int8, copy=False)

        return CSIRecord(
            host_time_s=time.time(),
            device_time_s=device_time_s,
            mac=mac,
            link=link,
            rssi=rssi,
            reported_len=reported_len,
            payload_len=int(len(values)),
            iq=iq,
        )

    except (ValueError, IndexError, OverflowError):
        return None


def trial_csv_header() -> list[str]:
    return [
        "host_time_s",
        "device_time_s",
        "session",
        "zone",
        "x_m",
        "y_m",
        "trial",
        "orientation",
        "link",
        "mac",
        "rssi",
        "reported_len",
        "payload_len",
        *IQ_COLUMNS,
    ]


def write_record(
    writer: csv.DictWriter,
    record: CSIRecord,
    *,
    session: str,
    zone: str,
    x_m: str | float,
    y_m: str | float,
    trial: int,
    orientation: str,
) -> None:
    row = {
        "host_time_s": f"{record.host_time_s:.6f}",
        "device_time_s": f"{record.device_time_s:.6f}",
        "session": session,
        "zone": zone,
        "x_m": x_m,
        "y_m": y_m,
        "trial": trial,
        "orientation": orientation,
        "link": record.link,
        "mac": record.mac,
        "rssi": record.rssi,
        "reported_len": record.reported_len,
        "payload_len": record.payload_len,
    }
    row.update({name: int(value) for name, value in zip(IQ_COLUMNS, record.iq)})
    writer.writerow(row)


def load_trial_csv(path: str | Path) -> dict:
    """
    Load one trial file.

    Returns:
        {
          "metadata": {...},
          "left":  {"time": ..., "rssi": ..., "iq": ...},
          "right": {"time": ..., "rssi": ..., "iq": ...},
        }
    """
    path = Path(path)

    per_link = {
        "left": {"time": [], "rssi": [], "iq": []},
        "right": {"time": [], "rssi": [], "iq": []},
    }
    metadata = None

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            link = row["link"].strip().lower()
            if link not in per_link:
                continue

            if metadata is None:
                metadata = {
                    "session": row["session"],
                    "zone": row["zone"],
                    "x_m": row["x_m"],
                    "y_m": row["y_m"],
                    "trial": row["trial"],
                    "orientation": row["orientation"],
                    "path": str(path),
                }

            try:
                iq = np.asarray([int(row[name]) for name in IQ_COLUMNS], dtype=np.int8)
                per_link[link]["time"].append(float(row["device_time_s"]))
                per_link[link]["rssi"].append(float(row["rssi"]))
                per_link[link]["iq"].append(iq)
            except (ValueError, KeyError):
                continue

    if metadata is None:
        raise ValueError(f"No usable CSI rows found in {path}")

    for link in ("left", "right"):
        if per_link[link]["time"]:
            order = np.argsort(np.asarray(per_link[link]["time"], dtype=np.float64))
            per_link[link]["time"] = np.asarray(per_link[link]["time"], dtype=np.float64)[order]
            per_link[link]["rssi"] = np.asarray(per_link[link]["rssi"], dtype=np.float64)[order]
            per_link[link]["iq"] = np.asarray(per_link[link]["iq"], dtype=np.int8)[order]
        else:
            per_link[link]["time"] = np.empty(0, dtype=np.float64)
            per_link[link]["rssi"] = np.empty(0, dtype=np.float64)
            per_link[link]["iq"] = np.empty((0, LLTF_RAW_COUNT), dtype=np.int8)

    return {
        "metadata": metadata,
        "left": per_link["left"],
        "right": per_link["right"],
    }
