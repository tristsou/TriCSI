"""AI-assisted development-only component checks; no changes to the frozen model.

Requires the original project scripts and raw S4-S7 recordings on your PC.
Matching seeds reduce a confound but do not guarantee identical random draws
after different preprocessing/training paths. Repeated seeds remain paired
algorithm runs, not additional independent recording sessions.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path.cwd())
    p.add_argument('--raw-dir', type=Path)
    p.add_argument('--output', type=Path, default=Path('results/anchor_matched_seeds'))
    p.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    p.add_argument('--epochs', type=int, default=70)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.001)
    return p.parse_args()


def main():
    a = parse_args()
    root = a.project_root.resolve()
    sys.path.insert(0, str(root))
    checks = importlib.import_module('run_tricsi_anchor_ablations')
    if tuple(checks.SESSIONS) != ('S4', 'S5', 'S6', 'S7'):
        raise ValueError('Expected DEVELOPMENT sessions S4-S7 only.')
    if len(a.seeds) != len(set(a.seeds)):
        raise ValueError('Seeds must be unique.')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw_dir = a.raw_dir.resolve() if a.raw_dir else checks.RAW_DATA_DIR
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ['run_tricsi_anchor_ablations.py', 'train_tricsi_anchor_final.py',
                           'train_csi_cnn.py', 'project_config.py']}
    (out / 'run_config.json').write_text(json.dumps({
        'seeds': a.seeds, 'epochs': a.epochs, 'batch_size': a.batch_size,
        'lr': a.lr, 'development_sessions': list(checks.SESSIONS),
        'source_script_sha256': hashes,
        'note': 'Post-review development analysis; no S8/S9 or frozen-model refitting.'
    }, indent=2), encoding='utf-8')
    raw = {s: checks.anchor.load_session(raw_dir, s) for s in checks.SESSIONS}
    rows = []
    variants = ('FULL', 'NO_BACKGROUND', 'NO_ORIENTATION_INVARIANCE')
    for seed in a.seeds:
        for fold, target_name in enumerate(checks.SESSIONS):
            training_names = [s for s in checks.SESSIONS if s != target_name]
            training_seed = seed + fold * 1000  # Deliberately no variant offset.
            for variant in variants:
                print(f'{seed=} {target_name=} {variant=} {training_seed=}', flush=True)
                source = checks.anchor.combine([
                    checks.prepare_source(raw[s], variant) for s in training_names])
                target = checks.prepare_target(raw[target_name], variant)
                model, norm = checks.train_variant(
                    source, variant=variant, epochs=a.epochs,
                    batch_size=a.batch_size, lr=a.lr, seed=training_seed)
                metrics = checks.evaluate_zero_shot(model, norm, source, target)
                rows.append(dict(seed=seed, training_seed=training_seed, variant=variant,
                                 test_session=target_name, **metrics))
                pd.DataFrame(rows).to_csv(out / 'completed_runs.csv', index=False)
    frame = pd.DataFrame(rows)
    paired = frame.pivot(index=['seed', 'test_session'], columns='variant',
                         values='exact_accuracy').reset_index()
    paired['full_minus_no_background_pp'] = 100 * (paired.FULL - paired.NO_BACKGROUND)
    paired['full_minus_no_orientation_pp'] = 100 * (
        paired.FULL - paired.NO_ORIENTATION_INVARIANCE)
    paired.to_csv(out / 'paired_effects.csv', index=False)
    frame.groupby(['test_session', 'variant']).exact_accuracy.agg(
        ['mean', 'std', 'min', 'max']).to_csv(out / 'per_session_summary.csv')
    print(f'Saved development results to {out}')


if __name__ == '__main__':
    main()
