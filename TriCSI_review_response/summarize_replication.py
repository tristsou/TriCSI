"""AI-assisted reporting of session-level replication; no session-level CI inferred.

Accept original s3_trial_predictions.csv or new trial_predictions.csv files.
Report S3 separately from prospectively collected replication sessions.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(paths):
    records, seen = [], set()
    for path in paths:
        f = pd.read_csv(path)
        needed = {'session', 'subset', 'trial_uid', 'true_zone', 'predicted_zone'}
        if not needed.issubset(f.columns):
            raise ValueError(f'{path}: missing {needed - set(f.columns)}')
        for name, frame in f.groupby('session', sort=False):
            if name in seen:
                raise ValueError(f'Session {name} supplied more than once.')
            seen.add(name)
            selected = frame[frame.subset.isin(['3tx_all', '2tx_outer'])].copy()
            selected['correct'] = (selected.true_zone == selected.predicted_zone).astype(int)
            if selected.duplicated(['subset', 'trial_uid']).any():
                raise ValueError('Duplicated trial predictions.')
            outer = selected[selected.subset == '2tx_outer']
            three = selected[selected.subset == '3tx_all']
            pair = outer.merge(three, on=['trial_uid', 'true_zone'],
                               suffixes=('_outer', '_3tx'), validate='one_to_one')
            if len(pair) != len(outer) or len(pair) != len(three) or not len(pair):
                raise ValueError('Both subsets must predict the identical trial set.')
            counts = pair.true_zone.value_counts()
            if set(counts.index) != {'EMPTY', *[f'Z{i}' for i in range(1, 10)]} or not (counts == 5).all():
                raise ValueError('Expected 50 trials: five in each of ten classes.')
            records.append(dict(
                session=name, role='initial_test' if name == '3TX_S3' else 'replication',
                n_trials=len(pair), outer_accuracy=pair.correct_outer.mean(),
                three_accuracy=pair.correct_3tx.mean(),
                gain_pp=100 * (pair.correct_3tx - pair.correct_outer).mean(),
                improved_trials=int(((pair.correct_3tx == 1) & (pair.correct_outer == 0)).sum()),
                harmed_trials=int(((pair.correct_3tx == 0) & (pair.correct_outer == 1)).sum())))
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    frame = summarize(a.predictions)
    a.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(a.output / 'per_session_results.csv', index=False)
    rep = frame[frame.role == 'replication']
    report = {'replication_session_count': len(rep),
              'note': 'Sessions are repeated measurements in one setup; no cross-person or cross-room claim.'}
    if len(rep):
        report.update(mean_session_gain_pp=float(rep.gain_pp.mean()),
                      minimum_session_gain_pp=float(rep.gain_pp.min()),
                      maximum_session_gain_pp=float(rep.gain_pp.max()),
                      sessions_with_positive_gain=int((rep.gain_pp > 0).sum()))
    (a.output / 'descriptive_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(max(6, len(frame) * 1.3), 4.5))
    ax.bar(x - .18, 100 * frame.outer_accuracy, .36, label='Outer pair')
    ax.bar(x + .18, 100 * frame.three_accuracy, .36, label='All three links')
    ax.set(xticks=x, xticklabels=frame.session, ylabel='Whole-trial accuracy (%)', ylim=(0, 110),
           title='Frozen model results by recording session')
    for i, r in frame.iterrows():
        ax.text(i, 103, f'{r.gain_pp:+.0f} pp', ha='center', fontsize=9)
    ax.legend(loc='lower right')
    fig.text(.01, .015, 'AI-assisted graph using Python and Matplotlib from student-collected data, 2026.', fontsize=7)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(a.output / 'accuracy_by_session.png', dpi=300)
    plt.close(fig)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
