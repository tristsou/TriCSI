"""Re-run the original source preparation and trainer without overwriting history.

This is a script-default reconstruction, not proof of the settings used for the
historical 95% checkpoint. Evaluation uses source prototypes and Empty-only
target calibration, as in evaluate_frozen_anchor_confirmatory.py.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.session_robustness.data import (
    assert_disjoint, calibrate, read_sessions, require_all_classes,
)
from experiments.session_robustness.study import ROOT, code_hashes, metrics, sha, write_json


SOURCE = ['S4', 'S5', 'S6', 'S7']


def run(args):
    import torch
    from experiments.session_robustness.model import (
        load_anchor, predict_trials, save_checkpoint, source_prototypes,
    )
    if args.epochs <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError('Epochs, batch size and threads must be positive.')
    if args.seed < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Use a nonnegative seed and finite positive learning rate.')
    torch.set_num_threads(args.threads)
    anchor = load_anchor(args.project_root)
    raw = read_sessions(args.cache_dir, SOURCE + ['S8', 'S9'])
    cache_hashes = {s: sha(args.cache_dir / f'{s}.npz') for s in raw}
    source_sets = []
    for name in SOURCE:
        require_all_classes(raw[name])
        if not np.any((raw[name].y_text == 'EMPTY') & (raw[name].trial_number == 1)):
            raise ValueError(f'{name}: missing Empty trial 1; inspect compatibility.')
        # Call the ORIGINAL implementation: all source Empty trials set the
        # background, and Empty trial 1 remains eligible for training.
        source_sets.append(anchor.prepare_source_session(raw[name]))
    source = anchor.combine(source_sets)
    if len(source.X) < args.batch_size:
        raise ValueError('Original drop_last=True would produce zero updates; reduce batch size.')
    targets = {}
    for name in ['S8', 'S9']:
        # Require the dedicated target reference; do not use the old fallback.
        scored, _ = calibrate(raw[name], [1, 2] if name == 'S8' else [])
        require_all_classes(scored)
        assert_disjoint(source, scored)
        targets[name] = scored
        print(f'Historical reconstruction: scoring {name}: '
              f'{len(set(scored.group.astype(str)))} whole trials; '
              'Empty trial 1 excluded', flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'checkpoints').mkdir()
    config = dict(
        complete=False, train_sessions=SOURCE, development_sessions=['S8', 'S9'],
        epochs=args.epochs, seed=args.seed, batch_size=args.batch_size, lr=args.lr,
        threads=args.threads, source_trials=len(set(source.group.astype(str))),
        source_windows=len(source.X), cache_sha256=cache_hashes,
        source_code_sha256=code_hashes(args.project_root),
        reconstruction_code_sha256=sha(Path(__file__)),
        source_background='Original all-Empty reference; Empty trial 1 included in training',
        sampler='Original shuffled DataLoader, drop_last=True; no epoch validation',
        checkpoint_selection='Fixed final epoch only; no S8/S9 checkpoint selection',
        development_trial_exclusions={'S8': [1, 2], 'S9': []},
        settings_provenance='Original script defaults unless explicitly overridden. Actual historical run settings remain unverified.',
        interpretation='Historical development reconstruction; this does not replace or identify the original frozen checkpoint. Physical zone compatibility must be checked from collection notes.',
    )
    write_json(args.output / 'run_config.json', config)
    started = time.monotonic()
    # Delegate directly: no rewritten optimizer, minibatch or augmentation code.
    model, norm = anchor.train_model(source, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    protos = source_prototypes(model, norm, source, anchor)
    updates = (len(source.X) // args.batch_size) * args.epochs
    path = args.output / 'checkpoints' / 'historical_reconstruction.pt'
    save_checkpoint(path, model, norm, protos, source_sessions=SOURCE,
        seed=args.seed, selected_epoch=args.epochs, updates=updates,
        reconstruction=True, target_labels_used_for_selection=False)
    rows, predictions = [], []
    for name, target in targets.items():
        predicted = predict_trials(model, norm, protos, target, anchor)
        predictions.extend(predicted)
        row = dict(session=name, seed=args.seed, epoch=args.epochs, **metrics(predicted))
        rows.append(row)
        print(f"RECONSTRUCTION {name}: {row['trial_accuracy']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(args.output / 'reconstruction_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(args.output / 'reconstruction_predictions.csv', index=False)
    if cache_hashes != {s: sha(args.cache_dir / f'{s}.npz') for s in raw}:
        raise ValueError('Input caches changed during reconstruction.')
    write_json(args.output / 'completed.json', dict(complete=True,
        checkpoint_sha256=sha(path), updates=updates, seconds=time.monotonic()-started))
    print('Reconstruction complete. Compare both sessions; original checkpoint was not changed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=ROOT)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=70)
    p.add_argument('--seed', type=int, default=26082026)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--threads', type=int, default=2)
    try:
        run(p.parse_args())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'STOP: {exc}') from exc


if __name__ == '__main__':
    main()
