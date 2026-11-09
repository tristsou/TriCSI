
from __future__ import annotations

import argparse
import time
from collections import Counter

import serial


KNOWN_MACS = {
    "02:00:00:00:00:01": "LEFT / COM9",
    "02:00:00:00:00:02": "RIGHT / COM10",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM11")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--seconds", type=float, default=15.0)
    args = p.parse_args()

    print(f"Opening {args.port} at {args.baud} baud...")
    ser = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        timeout=0.02,
        write_timeout=1.0,
    )
    ser.dtr = False
    ser.rts = False

    try:
        ser.reset_input_buffer()
        buffer = bytearray()
        start = time.monotonic()
        total_bytes = 0
        total_lines = 0
        csi_any = 0
        csi_ap = 0
        mac_counts = Counter()
        printed = 0

        print(f"Watching raw serial text for {args.seconds:.1f} s...\n")

        while time.monotonic() - start < args.seconds:
            waiting = ser.in_waiting
            chunk = ser.read(waiting if waiting > 0 else 4096)
            if not chunk:
                continue

            total_bytes += len(chunk)
            buffer.extend(chunk)

            while True:
                i = buffer.find(b"\n")
                if i < 0:
                    break

                raw = bytes(buffer[:i]).rstrip(b"\r")
                del buffer[:i+1]
                total_lines += 1

                text = raw.decode("utf-8", errors="replace")

                if "CSI_DATA" in text:
                    csi_any += 1
                    if "CSI_DATA,AP," in text:
                        csi_ap += 1

                upper = text.upper()
                for mac, label in KNOWN_MACS.items():
                    if mac in upper:
                        mac_counts[mac] += 1

                # Print startup/log lines and the first few CSI rows.
                if printed < 40:
                    print(f"{total_lines:04d}: {text[:300]}")
                    printed += 1

        print("\n" + "=" * 70)
        print("SUMMARY")
        print(f"bytes received : {total_bytes}")
        print(f"lines received : {total_lines}")
        print(f"any CSI_DATA   : {csi_any}")
        print(f"CSI_DATA,AP    : {csi_ap}")
        for mac, label in KNOWN_MACS.items():
            print(f"{label:14s}: {mac_counts[mac]} lines containing MAC")

        if csi_any == 0:
            print("\nNo CSI rows appeared during the watch period.")
            print("The printed startup/log lines above should reveal whether the AP")
            print("is booting, waiting for Wi-Fi stations, or running with CSI disabled.")
        elif csi_ap == 0:
            print("\nCSI exists, but the role text is not 'AP'.")
            print("Paste a CSI line so the collector parser can be adjusted.")
        elif any(mac_counts[m] == 0 for m in KNOWN_MACS):
            print("\nAP CSI is running, but one/both configured transmitter MACs are absent.")
        else:
            print("\nBoth expected transmitter MACs were seen in AP CSI rows.")

    finally:
        ser.close()


if __name__ == "__main__":
    main()
