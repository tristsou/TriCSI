#!/usr/bin/env python3
"""Generate the prespecified 200-trial prediction sheet and randomized schedules.

This script does not operate any ESP32 board and does not generate experimental predictions.
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = [
    ('3TX_S4', 'A_normal', 42026),
    ('3TX_S5', 'B_changed', 52026),
    ('3TX_S6', 'B_changed', 62026),
    ('3TX_S7', 'A_normal', 72026),
]
FIELDS = ['session', 'condition', 'trial_id', 'true_label', 'orientation', 'pred_2link', 'pred_3link']
ORDER = ['session', 'condition', 'capture_order', 'trial_id', 'true_label', 'orientation', 'duration_s']

def trial_list():
    trials = []
    for i in range(1, 6):
        trials.append({'trial_id': f'EMPTY_trial{i:02d}_NA.csv', 'true_label': 'EMPTY', 'orientation': 'NA', 'duration_s': 12 if i == 1 else 8})
    facing = ['N', 'E', 'S', 'W', 'N']
    for zone in range(1, 10):
        for i, facing_code in enumerate(facing, 1):
            trials.append({'trial_id': f'Z{zone}_trial{i:02d}_{facing_code}.csv', 'true_label': f'Z{zone}', 'orientation': facing_code, 'duration_s': 8})
    assert len(trials) == 50
    return trials


def main():
    predictions, schedule = [], []
    for session, condition, seed in PLAN:
        trials = trial_list()
        assert trials[0]['trial_id'] == 'EMPTY_trial01_NA.csv'
        rest = trials[1:]
        random.Random(seed).shuffle(rest)
        for sequence, t in enumerate([trials[0], *rest], 1):
            schedule.append({'session': session, 'condition': condition, 'capture_order': sequence,
                             'trial_id': t['trial_id'], 'true_label': t['true_label'],
                             'orientation': t['orientation'], 'duration_s': t['duration_s']})
        for t in trials:
            predictions.append({'session': session, 'condition': condition,
                                'trial_id': t['trial_id'], 'true_label': t['true_label'],
                                'orientation': t['orientation'], 'pred_2link': '', 'pred_3link': ''})
    for name, columns, rows in [('prediction_template.csv', FIELDS, predictions),
                                ('collection_order.csv', ORDER, schedule)]:
        path = ROOT / name
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
        print(f'Wrote {path} ({len(rows)} trials)')

if __name__ == '__main__':
    main()
