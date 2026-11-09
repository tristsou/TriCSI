
from __future__ import annotations

import argparse
import time
import serial
from serial.tools import list_ports


def list_all_ports() -> None:
    print("\nDetected serial ports:")
    ports = list(list_ports.comports())
    if not ports:
        print("  NONE")
        return

    for p in ports:
        print(
            f"  {p.device:6s}  "
            f"description={p.description!r}  "
            f"hwid={p.hwid!r}"
        )


def probe(port: str, baud: int, seconds: float) -> int:
    print(f"\nProbing {port} at {baud} baud for {seconds:.1f} s...")
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.05
    ser.write_timeout = 1.0

    # Keep reset-control lines deasserted before opening.
    ser.dtr = False
    ser.rts = False

    try:
        ser.open()
    except serial.SerialException as exc:
        print(f"ERROR opening {port}: {exc}")
        return 2

    try:
        try:
            ser.set_buffer_size(rx_size=262_144, tx_size=16_384)
        except Exception:
            pass

        # Some ESP32-S3 USB/serial paths reset on open. Give it time to boot.
        time.sleep(2.5)

        total_bytes = 0
        total_lines = 0
        csi_lines = 0
        ap_lines = 0
        sta_lines = 0
        sample_lines: list[str] = []
        buffer = bytearray()

        start = time.monotonic()
        while time.monotonic() - start < seconds:
            waiting = ser.in_waiting
            chunk = ser.read(waiting if waiting > 0 else 4096)

            if not chunk:
                continue

            total_bytes += len(chunk)
            buffer.extend(chunk)

            while True:
                idx = buffer.find(b"\n")
                if idx < 0:
                    break

                raw = bytes(buffer[:idx]).rstrip(b"\r")
                del buffer[:idx + 1]
                total_lines += 1

                if b"CSI_DATA" in raw:
                    csi_lines += 1
                if b"ACTIVE_AP" in raw or b"CSI_DATA,AP," in raw:
                    ap_lines += 1
                if b"ACTIVE_STA" in raw or b"CSI_DATA,STA," in raw:
                    sta_lines += 1

                if len(sample_lines) < 8:
                    sample_lines.append(raw.decode("utf-8", errors="replace")[:220])

        print("\nRESULT")
        print(f"  bytes received : {total_bytes}")
        print(f"  complete lines : {total_lines}")
        print(f"  CSI lines      : {csi_lines}")
        print(f"  AP-like lines  : {ap_lines}")
        print(f"  STA-like lines : {sta_lines}")

        if sample_lines:
            print("\nFirst sample lines:")
            for line in sample_lines:
                print("  ", line)
        else:
            print("\nNo serial text was received.")

        print("\nINTERPRETATION")
        if total_bytes == 0:
            print("  -> This port is producing no serial bytes.")
            print("  -> Check port identity, USB cable/power, firmware, and AP boot state.")
        elif csi_lines > 0 and ap_lines > 0:
            print("  -> This looks like the working AP/CSI receiver.")
        elif ap_lines > 0:
            print("  -> This looks like the AP, but CSI rows were not seen.")
        elif sta_lines > 0:
            print("  -> This looks like an ACTIVE_STA transmitter, not the AP.")
        else:
            print("  -> Serial is alive, but the role is unclear from this short probe.")

        return 0
    finally:
        ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe an ESP32 serial port before CSI collection."
    )
    parser.add_argument("--port", default="COM11")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    list_all_ports()
    if args.list_only:
        return

    probe(args.port, args.baud, args.seconds)


if __name__ == "__main__":
    main()
