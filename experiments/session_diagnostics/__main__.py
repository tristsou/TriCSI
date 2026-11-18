"""Independent diagnostics; preserves the earlier session-robustness package."""
import argparse
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.session_robustness.data import assert_disjoint, combine, read_sessions, require_all_classes
from experiments.session_robustness.study import ROOT, audit_trial, checkpoint_rank, code_hashes, metrics, resolve_exclusions, sha, write_json
from .core import ARMS, embedding_trials, prediction_reports, prepare_arm, signal_diagnostics


def check_frozen_inputs(run, cache, root):
    if not (run / 'completed.json').is_file():
        raise ValueError('Choose the completed 150-epoch development directory, not a predictions CSV.')
    config = json.loads((run / 'run_config.json').read_text())
    # Original experiment package is deliberately unchanged by this addition.
    for name, expected in config['source_code_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError(f'Original experiment code changed: {name}. Use its original version.')
    for session, expected in config['cache_sha256'].items():
        if sha(cache / f'{session}.npz') != expected:
            raise ValueError(f'Cache changed since training: {session}')
    return config


def plot_embeddings(source_frame, source_means, target_frame, target_means, prototypes, output, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    occupied = source_frame['true'].to_numpy() != 'EMPTY'
    # Fit projection on SOURCE trial centroids only, never on target labels/data.
    pca = PCA(n_components=2).fit(source_means[occupied])
    sxy, txy = pca.transform(source_means), pca.transform(target_means)
    pxy = pca.transform(prototypes)
    sessions = list(target_frame.session.unique())
    fig, axes = plt.subplots(1, len(sessions), figsize=(5 * len(sessions), 4), squeeze=False)
    for j, name in enumerate(sessions):
        ax = axes[0, j]
        for color, zone in zip(('C0', 'C1'), ('Z5', 'Z8')):
            sm = source_frame['true'].to_numpy() == zone
            tm = (target_frame['true'].to_numpy() == zone) & (target_frame.session.to_numpy() == name)
            ax.scatter(sxy[sm, 0], sxy[sm, 1], c=color, alpha=.35, s=18, label=f'Source {zone}')
            ax.scatter(txy[tm, 0], txy[tm, 1], c=color, marker='x', s=65, label=f'{name} {zone}')
            i = int(zone[1:]) - 1
            ax.scatter(pxy[i, 0], pxy[i, 1], c=color, marker='*', s=140)
        ax.set(title=name, xlabel='Source-fitted principal component 1', ylabel='Source-fitted principal component 2')
        ax.legend(fontsize=8)
    fig.suptitle('Neural trial centroids: projection only; stars = source prototypes')
    fig.tight_layout()
    fig.savefig(output / f'{stem}_embeddings.png', dpi=180)
    plt.close(fig)


def audit(args):
    import torch
    from experiments.session_robustness.model import load_anchor, load_checkpoint
    torch.set_num_threads(args.threads)
    run, cache = args.development_run.resolve(), args.cache_dir.resolve()
    config = check_frozen_inputs(run, cache, args.project_root)
    train_names, dev_names = config['train_sessions'], config['development_sessions']
    if set(train_names) & set(dev_names):
        raise ValueError('Source and development sessions overlap.')
    raw = read_sessions(cache, train_names + dev_names)
    exclusions = config.get('development_trial_exclusions')
    if exclusions is None:
        exclusions = resolve_exclusions(dev_names, config.get('exclude_dev_occupied_trials', []))
    anchor = load_anchor(args.project_root)
    source = combine([prepare_arm(raw[s])[0] for s in train_names])
    targets = {s: prepare_arm(raw[s], exclusions=exclusions[s])[0] for s in dev_names}
    for target in targets.values():
        assert_disjoint(source, target)
    index = json.loads((run / 'checkpoint_index.json').read_text())
    records = [r for r in index if r['method'] in args.methods]
    if not records or set(args.methods) - {r['method'] for r in records}:
        raise ValueError('Requested checkpoint methods are absent from the development run.')
    for r in records:
        if sha(run / r['path']) != r['sha256']:
            raise ValueError('Checkpoint hash mismatch.')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'diagnostic_receipt.json', dict(
        complete=False, fitting_performed=False, development_run=str(run),
        run_config_sha256=sha(run/'run_config.json'), train_sessions=train_names,
        development_sessions=dev_names, exclusions=exclusions,
        cache_sha256={s: sha(cache/f'{s}.npz') for s in raw},
        checkpoint_sha256={r['path']: r['sha256'] for r in records},
        diagnostic_code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        note='Exploratory diagnostics. File metadata cannot verify where a person physically stood. No relabeling or exclusion is performed.'))
    signal_diagnostics(raw, train_names, exclusions, args.output)
    print('Wrote measured CSI profiles and trial metadata.', flush=True)
    if args.raw_dir:
        import train_csi_cnn as cnn
        reports = []
        for session in raw:
            expected = set(zip(raw[session].y_text.astype(str), raw[session].trial_number.tolist(), raw[session].orientation.astype(str)))
            paths = cnn.trial_paths_for_session(args.raw_dir, session)
            metadata_path = cache / f'{session}.json'
            cached_sources = json.loads(metadata_path.read_text()).get('sources', {}) if metadata_path.is_file() else {}
            seen = set()
            for path in paths:
                result = audit_trial(path)
                result['sha256'] = sha(path)
                identity = (result['zone'], int(result['trial']) if str(result['trial']).isdigit() else -1, result['orientation'])
                result['identity_in_cache'] = identity in expected and result['session'] == session
                candidates = [h for p, h in cached_sources.items()
                              if str(p).replace('\\', '/').split('/')[-1] == path.name]
                result['matches_cached_source_hash'] = (result['sha256'] == candidates[0]
                    if len(candidates) == 1 else 'not_recorded_or_ambiguous')
                result['duplicate_identity'] = identity in seen
                seen.add(identity)
                reports.append(result)
            missing = expected - seen
            for zone, trial, orientation in sorted(missing):
                reports.append(dict(session=session, zone=zone, trial=trial,
                                    orientation=orientation, status='cached_identity_missing_from_raw'))
        pd.DataFrame(reports).to_csv(args.output / 'raw_metadata_audit.csv', index=False)
    predicted = []
    for record in records:
        path = run / record['path']
        model, norm, prototypes, metadata = load_checkpoint(path, anchor)
        if metadata.get('source_sessions') != train_names:
            raise ValueError('Checkpoint source sessions differ from its run config.')
        source_frame, source_mean = embedding_trials(anchor.embed_raw(model, source, norm), source, prototypes, anchor)
        frames, means = [], []
        for name, target in targets.items():
            frame, mean = embedding_trials(anchor.embed_raw(model, target, norm), target, prototypes, anchor)
            frame['method'], frame['seed'], frame['strength'] = record['method'], record['seed'], record['strength']
            frame['selected_epoch'] = metadata.get('selected_epoch')
            frames.append(frame)
            means.append(mean)
        target_frame, target_mean = pd.concat(frames, ignore_index=True), np.concatenate(means)
        stem = path.stem
        target_frame.to_csv(args.output / f'{stem}_trial_probabilities.csv', index=False)
        source_frame.to_csv(args.output / f'{stem}_source_probabilities.csv', index=False)
        np.savez_compressed(args.output / f'{stem}_trial_embeddings.npz', source=source_mean,
                           target=target_mean, prototypes=prototypes,
                           source_group=source_frame.group.to_numpy(dtype=str),
                           target_group=target_frame.group.to_numpy(dtype=str))
        plot_embeddings(source_frame, source_mean, target_frame, target_mean, prototypes, args.output, stem)
        predicted.append(target_frame)
        print(f'Diagnosed {stem}; no weights changed.', flush=True)
    frame = pd.concat(predicted, ignore_index=True)
    # Ensure inference exactly reproduces saved development labels before calling
    # the result an audit of those models. No target labels enter inference.
    original = pd.read_csv(run / 'development_predictions.csv')
    keys = ['method', 'strength', 'seed', 'session', 'group']
    joined = frame.merge(original[keys + ['pred', 'true']], on=keys, suffixes=('', '_saved'), validate='one_to_one')
    if len(joined) != len(frame) or not ((joined.pred == joined.pred_saved) & (joined['true'] == joined.true_saved)).all():
        raise ValueError('Inference does not reproduce saved predictions. Treat this audit as incomplete.')
    frame.to_csv(args.output / 'diagnostic_predictions.csv', index=False)
    # Multiple strengths are kept distinct rather than silently averaged/voted.
    if not frame.duplicated(['method','seed','session','group']).any():
        prediction_reports(frame, args.output)
    receipt = json.loads((args.output / 'diagnostic_receipt.json').read_text())
    receipt.update(complete=True, saved_predictions_reproduced=True)
    write_json(args.output / 'diagnostic_receipt.json', receipt)
    print('AUDIT COMPLETE: saved predictions reproduced; no model fitting performed.')
    # Snapshot before opening the ZIP so it cannot include itself. This contains
    # diagnostics only, never the input recordings, caches or model checkpoints.
    diagnostic_files = sorted(p for p in args.output.iterdir() if p.is_file())
    archive = args.output / 'diagnostic_results.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in diagnostic_files:
            zipped.write(path, arcname=path.name)
    print(f'Share this diagnostic bundle: {archive}')


def compare(args):
    import torch
    from experiments.session_robustness.model import load_anchor, predict_trials, source_prototypes, save_checkpoint, train
    torch.set_num_threads(args.threads)
    if len(args.train) < 2 or set(args.train) & set(args.dev):
        raise ValueError('Use >=2 source sessions disjoint from development sessions.')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError('Seeds must be unique and nonnegative.')
    exclusions = resolve_exclusions(args.dev, per_session=args.exclude_dev_occupied_by_session)
    anchor = load_anchor(args.project_root)
    raw = read_sessions(args.cache_dir, args.train + args.dev)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'checkpoints').mkdir()
    config = dict(train_sessions=args.train, development_sessions=args.dev, seeds=args.seeds,
        arms=list(ARMS), augmentation='existing', max_epochs=args.epochs,
        eval_every=args.eval_every, batch_size=args.batch_size, lr=args.lr,
        development_trial_exclusions=exclusions,
        cache_sha256={s: sha(args.cache_dir / f'{s}.npz') for s in raw},
        original_code_sha256=code_hashes(args.project_root),
        diagnostic_code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        selection='One checkpoint per arm/seed: highest worst development-session accuracy, then mean accuracy, then macro-F1; earlier epoch on ties.',
        interpretation='Development only. S8/S9 labels select checkpoints. Changing the representation requires retraining; old weights are never reused with new inputs.')
    write_json(args.output / 'run_config.json', config)
    curves, rows, predictions, checkpoints = [], [], [], []
    for arm in ARMS:
        source = combine([prepare_arm(raw[s], arm)[0] for s in args.train])
        targets = {s: prepare_arm(raw[s], arm, exclusions[s])[0] for s in args.dev}
        require_all_classes(source)
        for session, target in targets.items():
            assert_disjoint(source, target)
            require_all_classes(target)
            print(f'{arm}: scoring {session}: {len(set(target.group.astype(str)))} whole trials; '
                  f'occupied exclusions={exclusions[session]}; Empty trial 1 excluded', flush=True)
        for seed in args.seeds:
            started = time.monotonic()
            def score(candidate, norm, epoch, updates, loss):
                protos = source_prototypes(candidate, norm, source, anchor)
                session_rows = []
                for session, target in targets.items():
                    scored = dict(session=session, **metrics(predict_trials(candidate, norm, protos, target, anchor)))
                    session_rows.append(scored)
                    curves.append(dict(arm=arm, seed=seed, epoch=epoch, updates=updates, training_loss=loss, **scored))
                pd.DataFrame(curves).to_csv(args.output / 'learning_curves.csv', index=False)
                rank = checkpoint_rank(session_rows)
                detail = ' '.join(f"{r['session']}={r['trial_accuracy']:.3f}" for r in session_rows)
                print(f'{arm} seed={seed} epoch={epoch}: {detail}', flush=True)
                return dict(rank=list(rank), sessions=session_rows)
            # None is safe: the existing-augmentation arm never uses a generator.
            model, norm, updates, selection = train(source, anchor, None, method='existing',
                strength=0., seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, on_epoch=score, eval_every=args.eval_every)
            protos = source_prototypes(model, norm, source, anchor)
            path = args.output / 'checkpoints' / f'{arm}_seed{seed}.pt'
            save_checkpoint(path, model, norm, protos, representation=arm, method='existing',
                            source_sessions=args.train, seed=seed, updates=updates, **selection)
            checkpoints.append(dict(arm=arm, seed=seed, path=str(path.relative_to(args.output)), sha256=sha(path), updates=updates))
            for session, target in targets.items():
                predicted = embedding_trials(anchor.embed_raw(model, target, norm), target, protos, anchor)[0].to_dict('records')
                predictions.extend(dict(arm=arm, seed=seed, selected_epoch=selection['selected_epoch'], **r) for r in predicted)
                rows.append(dict(arm=arm, seed=seed, session=session, updates=updates,
                    selected_epoch=selection['selected_epoch'], train_seconds=time.monotonic()-started, **metrics(predicted)))
            pd.DataFrame(rows).to_csv(args.output / 'comparison_metrics.csv', index=False)
            pd.DataFrame(predictions).to_csv(args.output / 'comparison_predictions.csv', index=False)
            write_json(args.output / 'checkpoint_index.json', checkpoints)
    if len({r['updates'] for r in checkpoints}) != 1:
        raise ValueError('Unequal optimizer update counts; comparison invalid.')
    frame = pd.DataFrame(rows)
    per_seed = frame.groupby(['arm','seed']).trial_accuracy.agg(mean_session_accuracy='mean', worst_session_accuracy='min')
    summary = per_seed.groupby('arm').agg(['mean','std'])
    summary.to_csv(args.output / 'comparison_summary.csv')
    write_json(args.output / 'completed.json', dict(complete=True, fits=len(checkpoints)))
    print(summary.to_string())
    print('COMPARISON COMPLETE. Report all seeds and both development sessions.')


def predictions(args):
    frame = pd.read_csv(args.predictions)
    args.output.mkdir(parents=True, exist_ok=False)
    prediction_reports(frame, args.output)
    write_json(args.output / 'receipt.json', dict(fitting_performed=False,
        predictions_sha256=sha(args.predictions), note='Exploratory existing/generic/session vote; ties go to existing.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=ROOT)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit', help='Read-only metadata, measured CSI and frozen-neural diagnostics')
    a.add_argument('--development-run', type=Path, required=True)
    a.add_argument('--cache-dir', type=Path, required=True)
    a.add_argument('--raw-dir', type=Path)
    a.add_argument('--methods', nargs='+', default=['existing'], choices=['none','existing','generic','session'])
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--threads', type=int, default=2)
    a.set_defaults(func=audit)
    c = sub.add_parser('compare', help='Matched legacy vs background-relative shape, trained from scratch')
    c.add_argument('--cache-dir', type=Path, required=True)
    c.add_argument('--train', nargs='+', required=True)
    c.add_argument('--dev', nargs='+', required=True)
    c.add_argument('--exclude-dev-occupied-by-session', nargs='+', default=[])
    c.add_argument('--seeds', nargs='+', type=int, default=[2026,2027,2028])
    c.add_argument('--epochs', type=int, default=150)
    c.add_argument('--eval-every', type=int, default=5)
    c.add_argument('--batch-size', type=int, default=128)
    c.add_argument('--lr', type=float, default=.001)
    c.add_argument('--threads', type=int, default=2)
    c.add_argument('--output', type=Path, required=True)
    c.set_defaults(func=compare)
    r = sub.add_parser('predictions', help='Summarize uploaded predictions without PyTorch or caches')
    r.add_argument('--predictions', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.set_defaults(func=predictions)
    args = p.parse_args()
    for name in ('epochs','eval_every','batch_size','threads'):
        if hasattr(args,name) and getattr(args,name) <= 0:
            p.error(f'{name} must be positive')
    if hasattr(args,'lr') and (not np.isfinite(args.lr) or args.lr <= 0):
        p.error('lr must be finite and positive')
    try:
        args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'STOP: {exc}') from exc


if __name__ == '__main__':
    main()
