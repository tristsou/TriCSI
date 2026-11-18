"""AI-assisted evaluator for new three-link sessions using the ORIGINAL frozen model.

No fitting or tuning. The supplied expected SHA256 pins the model chosen before
collection. Use only a trusted joblib file produced by your own project.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from tricsi_3tx_modeling import (
    combine_links, evaluate_classifier, extract_session_features,
    session_data_sha256, stratified_paired_bootstrap,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-model-sha256', required=True)
    a = p.parse_args()
    actual_hash = hashlib.sha256(a.model.read_bytes()).hexdigest()
    if actual_hash != a.expected_model_sha256.lower().strip():
        raise ValueError('Frozen model SHA256 differs from the recorded model.')
    if a.output.exists():
        raise FileExistsError('Use a new output folder; results are never overwritten.')
    bundle = joblib.load(a.model)
    if bundle.get('format_version') != 2 or bundle['primary_subset'] != '3tx_all':
        raise ValueError('Expected original format-2 bundle with primary subset 3tx_all.')
    config = bundle['feature_config']
    data = extract_session_features(
        a.session, window_packets=int(config['window_packets']),
        windows_per_trial=int(config['windows_per_trial']))
    data_hash = session_data_sha256(data.source_dir)
    if data.session_name in bundle['training_sessions'] or data_hash in set(
            bundle.get('training_data_sha256', {}).values()):
        raise ValueError('Session duplicates training data or has a training-session name.')
    if data.zone_coords != bundle['zone_coords']:
        raise ValueError('Grid coordinates differ from the frozen setup.')
    rows, frames = [], []
    for subset, model in bundle['models_by_subset'].items():
        metrics, predictions, _ = evaluate_classifier(
            model, combine_links(data, bundle['subset_links'][subset]),
            data.metadata, data.zone_coords)
        rows.append({'session': data.session_name, 'subset': subset, **metrics})
        predictions.insert(0, 'subset', subset)
        frames.append(predictions)
    predictions = pd.concat(frames, ignore_index=True)
    pair = predictions[predictions.subset == '2tx_outer'].merge(
        predictions[predictions.subset == '3tx_all'],
        on=['session', 'trial_uid', 'true_zone'], suffixes=('_outer', '_3tx'),
        validate='one_to_one')
    pair['gain'] = pair.correct_3tx - pair.correct_outer
    interval = stratified_paired_bootstrap(pair, 'gain', iterations=5000, seed=2026)
    a.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(a.output / 'subset_metrics.csv', index=False)
    predictions.to_csv(a.output / 'trial_predictions.csv', index=False)
    pair.to_csv(a.output / 'paired_predictions.csv', index=False)
    (a.output / 'evaluation.json').write_text(json.dumps({
        'session': data.session_name, 'model_sha256': actual_hash,
        'session_data_sha256': data_hash, 'primary_subset': bundle['primary_subset'],
        'training_sessions': bundle['training_sessions'],
        'accuracy_gain_3tx_minus_outer': interval,
        'interpretation': 'Interval resamples this session only, not future sessions.',
        'error_measure': 'Zone-center error with frozen penalty for occupied-to-EMPTY errors.',
        'fitting_performed': False,
    }, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows)[['subset', 'trial_accuracy', 'mean_position_error_m']].to_string(index=False))
    print(f'Saved {a.output}; no models retrained.')


if __name__ == '__main__':
    main()
