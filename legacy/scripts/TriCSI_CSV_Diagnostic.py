"""TriCSI headered CSV diagnostic: read-only, independent of the experimental parser.
Run from the project root using Python 3.9+.
It never changes raw recordings or any saved models.
"""
from __future__ import annotations
import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

MAC_PATTERN = re.compile(r'^[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}$')
CLASSES = ['EMPTY'] + [f'Z{i}' for i in range(1, 10)]


def classify(row, header, indices):
    if len(row) != len(header):
        return 'bad_row_width', f'expected={len(header)},actual={len(row)}'
    mac = row[indices['mac']].strip()
    if not MAC_PATTERN.fullmatch(mac):
        return 'bad_mac', repr(mac[:120])
    rssi_text = row[indices['rssi']].strip()
    try:
        rssi = float(rssi_text)
        if not math.isfinite(rssi):
            return 'bad_rssi', repr(rssi_text[:120])
    except ValueError:
        return 'bad_rssi', repr(rssi_text[:120])
    length_text = row[indices['len']].strip()
    try:
        length = int(length_text)
        if length < 128:
            return 'short_declared_csi_len', repr(length_text[:120])
    except ValueError:
        return 'malformed_declared_csi_len', repr(length_text[:120])
    for j, col_idx in enumerate(indices['iq']):
        value = row[col_idx].strip()
        try:
            n = float(value)
            if not math.isfinite(n):
                return 'nonfinite_iq', f'iq_{j:03d}={value[:100]!r}'
        except ValueError:
            return 'nonnumeric_iq', f'iq_{j:03d}={value[:100]!r}'
    return 'valid_prefix', f'mac={mac},rssi={rssi_text},len={length_text},iq_000={row[indices["iq"][0]]!r},iq_127={row[indices["iq"][-1]]!r}'


def run(root, sessions, max_files):
    for session in sessions:
        folder = root / session
        files = sorted(folder.glob('*.csv'))
        if max_files: files = files[:max_files]
        print(f'\n=== {session}: inspecting {len(files)} CSV files ===')
        reasons, links, macs, length_counts, classes = Counter(), Counter(), Counter(), Counter(), Counter()
        examples = {}
        for path in files:
            with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
                reader = csv.reader(f)
                hdr = next(reader, [])
                header = [x.strip().lower() for x in hdr]
                if not header:
                    reasons['missing_header'] += 1
                    continue
                try:
                    indices = {'mac': header.index('mac'), 'rssi': header.index('rssi'),
                               'len': header.index('len'), 'link': header.index('link'),
                               'iq': [header.index(f'iq_{j:03d}') for j in range(128)]}
                except ValueError as e:
                    reasons['missing_expected_column'] += 1
                    examples.setdefault('missing_expected_column', (path.name, str(e)))
                    continue
                classes[path.name.split('_trial')[0]] += 1
                for line, row in enumerate(reader, start=2):
                    if not row or not any(t.strip() for t in row): continue
                    key, detail = classify(row, header, indices)
                    reasons[key] += 1
                    if len(examples.get(key, [])) < 2:
                        examples.setdefault(key, []).append((f'{path.name}:{line}', detail))
                    if key == 'valid_prefix':
                        links[row[indices['link']].strip().lower()] += 1
                        macs[row[indices['mac']].strip().lower()] += 1
                    if len(row) > indices['len']:
                        lenval = row[indices['len']].strip()
                        if lenval.isdigit(): length_counts[lenval] += 1
        print('Rows by validation result:',dict(reasons))
        print('Valid rows by link:',dict(links))
        print('Valid rows by MAC:',dict(macs))
        print('Numeric declared CSI lengths:',length_counts.most_common(5))
        for reason, pairs in examples.items():
            print(f'  Example {reason}: {pairs}')
        print('This is an independent diagnostic; it does not prove parity with the original 3TX feature extractor.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', default='data_3tx/raw')
    parser.add_argument('--sessions', nargs='+', default=['3TX_S1','3TX_S2','3TX_S3'])
    parser.add_argument('--max-files', type=int, default=1,
                        help='First N CSVs in each session, or 0 for all')
    args=parser.parse_args()
    run(Path(args.root),args.sessions,args.max_files)
