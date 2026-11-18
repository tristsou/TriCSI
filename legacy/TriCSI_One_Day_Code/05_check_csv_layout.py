"""Read-only diagnostic for the ESP32 TriCSI headered trial CSV layout.
Does not alter data and does not train or fit models.
"""
import argparse, csv
from collections import Counter
from pathlib import Path
from common import trials, raw_rows


def inspect(root, sessions):
    for session in sessions:
        paths = trials(root, session)
        widths, raw_lengths, io_cols, valid, rows, macs = Counter(),Counter(),Counter(),0,0,Counter()
        seen_header = False
        for path in paths:
            with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, [])
                if not header: continue
                keys = [k.strip().lower() for k in header]
                iq_count = sum(k.startswith('iq_') and k[3:].isdigit() for k in keys)
                io_cols[iq_count] += 1
                if not seen_header:
                    print(f'First CSV header in {session}: columns={len(keys)}, IQ_columns={iq_count}, last_IQ={next((k for k in reversed(keys) if k.startswith("iq_")), "not found")}')
                    seen_header = True
                try: len_idx = keys.index('len')
                except ValueError: len_idx = -1
                for row in reader:
                    if not row: continue
                    rows += 1
                    widths[len(row)] += 1
                    if len_idx >= 0 and len_idx < len(row):
                        raw_lengths[row[len_idx].strip()] += 1
            for rec in raw_rows(path):
                valid += 1
                macs[rec['mac']] += 1
        print(f'{session}: trial_files={len(paths)} raw_rows={rows} parsed_rows={valid} parsed_pct={100*valid/rows:.1f}%' if rows else f'{session}: no raw rows')
        print('  Header IQ column counts (files):', dict(io_cols))
        print('  Most common declared raw CSI lengths:', raw_lengths.most_common(7))
        print('  Most common actual CSV row widths:', widths.most_common(7))
        print('  Parsed MAC counts:', dict(macs))
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--sessions',nargs='+',default=['3TX_S1','3TX_S2','3TX_S3'])
    args=parser.parse_args()
    inspect(args.root,args.sessions)
