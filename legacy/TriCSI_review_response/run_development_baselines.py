"""AI-assisted simple baselines on 3TX_S1/S2 only.

These are diagnostic controls, NOT reproductions of published DAFI/other methods.
No S3 or later session is opened, and no final model is replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from tricsi_3tx_modeling import (
    combine_links, evaluate_classifier, extract_session_features,
    make_classifier, session_data_sha256,
)


def feature_view(data, links, view):
    values = []
    for link in links:
        names = [n.rsplit('__', 1)[-1] for n in data.feature_names_by_link[link]]
        if view == 'rssi_only':
            idx = [i for i, n in enumerate(names) if n.startswith('rssi_')]
        elif view == 'csi_without_rssi':
            idx = [i for i, n in enumerate(names) if not n.startswith('rssi_')]
        else:
            idx = list(range(len(names)))
        values.append(data.features_by_link[link][:, idx])
    return np.concatenate(values, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=Path('data_3tx/raw'))
    p.add_argument('--output', type=Path, default=Path('results/3tx_development_baselines'))
    p.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    a = p.parse_args()
    if len(a.seeds) != len(set(a.seeds)):
        raise ValueError('Seeds must be unique.')
    a.output.mkdir(parents=True, exist_ok=False)
    data = {s: extract_session_features(a.raw_root / s) for s in ('3TX_S1', '3TX_S2')}
    if data['3TX_S1'].zone_coords != data['3TX_S2'].zone_coords:
        raise ValueError('Development grids differ.')
    hashes = {s: session_data_sha256(d.source_dir) for s, d in data.items()}
    if len(set(hashes.values())) != 2:
        raise ValueError('Duplicated development session bytes.')
    conditions = [('ExtraTrees_full', 'full', 'et'),
                  ('ExtraTrees_RSSI_only', 'rssi_only', 'et'),
                  ('ExtraTrees_CSI_without_RSSI', 'csi_without_rssi', 'et'),
                  ('RandomForest_full', 'full', 'rf')]
    subsets = {'2tx_outer': ('back_left', 'back_right'),
               '3tx_all': ('back_left', 'middle_back', 'back_right')}
    rows, predictions = [], []
    (a.output / 'run_config.json').write_text(json.dumps({
        'sessions': list(data), 'seeds': a.seeds, 'conditions': conditions,
        'training_data_sha256': hashes,
        'shared_feature_module_sha256': hashlib.sha256(
            Path(__file__).with_name('tricsi_3tx_modeling.py').read_bytes()).hexdigest(),
        'note': 'Post-review development controls; not a published-method benchmark.'
    }, indent=2), encoding='utf-8')
    for source_name, target_name in [('3TX_S1', '3TX_S2'), ('3TX_S2', '3TX_S1')]:
        source, target = data[source_name], data[target_name]
        for seed in a.seeds:
            for subset, links in subsets.items():
                for name, view, kind in conditions:
                    train = feature_view(source, links, view)
                    test = feature_view(target, links, view)
                    model = (make_classifier(seed=seed) if kind == 'et' else RandomForestClassifier(
                        n_estimators=400, max_features='sqrt', min_samples_leaf=2,
                        class_weight='balanced', random_state=seed, n_jobs=-1))
                    model.fit(train, source.metadata.zone)
                    metrics, pred, _ = evaluate_classifier(model, test, target.metadata, target.zone_coords)
                    meta = dict(training_session=source_name, test_session=target_name,
                                seed=seed, subset=subset, condition=name)
                    rows.append({**meta, **metrics})
                    for k, v in meta.items():
                        pred[k] = v
                    predictions.append(pred)
                    pd.DataFrame(rows).to_csv(a.output / 'completed_metrics.csv', index=False)
                    print(meta, metrics['trial_accuracy'], flush=True)
    frame = pd.DataFrame(rows)
    pd.concat(predictions, ignore_index=True).to_csv(a.output / 'trial_predictions.csv', index=False)
    frame.groupby(['test_session', 'subset', 'condition']).trial_accuracy.agg(
        ['mean', 'std', 'min', 'max']).to_csv(a.output / 'summary_by_session.csv')
    print('Finished development controls. Frozen models are unchanged.')


if __name__ == '__main__':
    main()
