"""Source-referenced diagnostics. Distances are feature distances, not meters."""
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.session_robustness.data import calibrate

ARMS = ('legacy', 'relative_shape')


def prepare_arm(raw, arm='legacy', exclusions=()):
    if arm not in ARMS:
        raise ValueError(f'Unknown representation: {arm}')
    scored, reference = calibrate(raw, exclusions)
    if arm == 'relative_shape':
        # These absolute channels have ALREADY had the dedicated Empty profile
        # subtracted. Center their residuals, not the original occupied signal.
        residual = scored.X[:, [0, 2]]
        scored.X[:, [1, 3]] = residual - np.median(residual, axis=-1, keepdims=True)
    return scored, reference


def trial_profiles(raw, channels=(0, 2)):
    rows, vectors = [], []
    for group in sorted(set(raw.group.astype(str))):
        mask = raw.group == group
        i = np.flatnonzero(mask)[0]
        rows.append(dict(group=group, session=str(raw.session[i]), true=str(raw.y_text[i]),
            orientation=str(raw.orientation[i]), trial=int(raw.trial_number[i]),
            source_file=str(raw.source_file[i]), windows=int(mask.sum()),
            left_packets_min=int(raw.left_packets[mask].min()),
            right_packets_min=int(raw.right_packets[mask].min())))
        vectors.append(np.median(raw.X[mask][:, channels], axis=(0, 2)))
    return pd.DataFrame(rows), np.stack(vectors)


def signal_diagnostics(raw, train_names, exclusions, output, focus=('Z5', 'Z8')):
    """No network fitting, target normalization, or occupied-target calibration."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    profile_rows, distance_rows, inventory_rows = [], [], []
    for name, r in raw.items():
        meta, _ = trial_profiles(r)
        meta['role'] = 'source' if name in train_names else 'development'
        meta['scored'] = ~((meta.true == 'EMPTY') & (meta.trial == 1))
        if name not in train_names:
            meta.loc[(meta.true != 'EMPTY') & meta.trial.isin(exclusions[name]), 'scored'] = False
        inventory_rows.extend(meta.to_dict('records'))
    for mode in ('raw_amplitude', 'empty_relative', 'legacy_shape', 'relative_shape'):
        profiles = {}
        for name, r in raw.items():
            scored, _ = prepare_arm(r, 'relative_shape' if mode == 'relative_shape' else 'legacy',
                                   exclusions.get(name, ()))
            if mode == 'raw_amplitude':
                keep = np.isin(r.group, scored.group)
                scored = r.subset(keep)  # same trial set as every other representation
            channels = (1, 3) if 'shape' in mode else (0, 2)
            profiles[name] = trial_profiles(scored, channels)
        source_meta = pd.concat([profiles[s][0] for s in train_names], ignore_index=True)
        source_vectors = np.concatenate([profiles[s][1] for s in train_names])
        occupied = source_meta.true.to_numpy() != 'EMPTY'
        # Standardize distances with occupied SOURCE trial variation only.
        scale = source_vectors[occupied].std(axis=0)
        positive = scale[scale > 1e-8]
        floor = max(float(np.median(positive)) * .05 if len(positive) else 1e-4, 1e-6)
        scale = np.maximum(scale, floor)
        refs = {z: np.median(source_vectors[source_meta.true.to_numpy() == z], axis=0)
                for z in focus}
        for name, (meta, vectors) in profiles.items():
            for i, row in meta.iterrows():
                if row['true'] not in focus:
                    continue
                for link, link_name in enumerate(('left', 'right')):
                    for k, value in enumerate(vectors[i, link]):
                        profile_rows.append(dict(**row.to_dict(), representation=mode,
                            role='source' if name in train_names else 'development',
                            link=link_name, bin_index=k, value=float(value)))
                    if name not in train_names:
                        for zone, ref in refs.items():
                            distance_rows.append(dict(**row.to_dict(), representation=mode,
                                link=link_name, source_reference_zone=zone,
                                standardized_rms_distance=float(np.sqrt(np.mean(
                                    ((vectors[i, link] - ref[link]) / scale[link]) ** 2)))))
        dev_names = [s for s in raw if s not in train_names]
        fig, axes = plt.subplots(len(dev_names), 2, figsize=(11, 3.3 * len(dev_names)), squeeze=False)
        for j, name in enumerate(dev_names):
            meta, vec = profiles[name]
            for link, link_name in enumerate(('left', 'right')):
                ax = axes[j, link]
                for k, z in enumerate(focus):
                    ax.plot(refs[z][link], color=f'C{k}', linestyle='--', label=f'Source {z}')
                    selected = meta.true.to_numpy() == z
                    if selected.any():
                        ax.plot(np.median(vec[selected, link], axis=0), color=f'C{k}', label=f'{name} {z}')
                ax.set(title=f'{name}: {link_name} link', xlabel='Retained CSI bin index', ylabel='Log-amplitude feature')
                ax.legend(fontsize=8)
        fig.suptitle(mode.replace('_', ' ') + ': medians of whole-trial summaries')
        fig.tight_layout()
        fig.savefig(Path(output) / f'csi_{mode}.png', dpi=180)
        plt.close(fig)
    pd.DataFrame(inventory_rows).to_csv(Path(output) / 'trial_metadata.csv', index=False)
    pd.DataFrame(profile_rows).to_csv(Path(output) / 'signal_profiles.csv', index=False)
    pd.DataFrame(distance_rows).to_csv(Path(output) / 'signal_distances.csv', index=False)


def embedding_trials(out, raw, prototypes, anchor):
    """Exact existing whole-trial rule plus diagnostic conditional probabilities."""
    q = anchor.prototype_probabilities(out['embedding'], prototypes)
    rows, means = [], []
    for group in sorted(set(raw.group.astype(str))):
        mask = raw.group == group
        i = np.flatnonzero(mask)[0]
        probabilities = q[mask].mean(axis=0)
        occ = float(out['occ_prob'][mask].mean())
        pred = 'EMPTY' if occ < .5 else anchor.OCCUPIED[int(probabilities.argmax())]
        centroid = out['embedding'][mask].mean(axis=0)
        distances = 1 - anchor.normalize_rows(centroid[None])[0] @ anchor.normalize_rows(prototypes).T
        row = dict(session=str(raw.session[i]), group=group, true=str(raw.y_text[i]),
            pred=pred, orientation=str(raw.orientation[i]), occupied_probability=occ)
        for j, z in enumerate(anchor.OCCUPIED):
            row[f'p_{z}'] = float(probabilities[j])
            row[f'cosine_distance_{z}'] = float(distances[j])
        rows.append(row)
        means.append(centroid)
    return pd.DataFrame(rows), np.stack(means)


def prediction_reports(frame, output):
    """Unweighted vote is descriptive; no labels are used to choose its members."""
    required = {'method', 'seed', 'session', 'group', 'true', 'pred', 'orientation'}
    if not required <= set(frame.columns):
        raise ValueError(f'Missing prediction columns: {sorted(required - set(frame.columns))}')
    frame = frame.copy()
    if 'strength' not in frame:
        frame['strength'] = 0.
    frame['orientation'] = frame.orientation.fillna('NA')
    if frame.duplicated(['method', 'seed', 'session', 'group']).any():
        raise ValueError('Choose one strength/checkpoint per method; duplicate trial predictions found.')
    if frame.groupby(['seed', 'session', 'group'])['true'].nunique().max() != 1:
        raise ValueError('Methods disagree on trial labels.')
    if frame.groupby(['seed', 'session', 'group']).orientation.nunique().max() != 1:
        raise ValueError('Methods disagree on trial orientations.')
    frame['correct'] = frame['true'] == frame.pred
    frame.groupby(['method', 'seed', 'session']).correct.agg(['sum', 'count', 'mean']).to_csv(Path(output)/'accuracy.csv')
    frame.groupby(['method', 'seed', 'session', 'true']).correct.agg(['sum', 'count', 'mean']).to_csv(Path(output)/'zone_accuracy.csv')
    frame.groupby(['method', 'seed', 'session', 'orientation']).correct.agg(['sum', 'count', 'mean']).to_csv(Path(output)/'orientation_accuracy.csv')
    frame[~frame.correct].to_csv(Path(output)/'errors.csv', index=False)
    common, votes = [], []
    for (seed, session), group in frame.groupby(['seed', 'session']):
        p = group.pivot(index='group', columns='method', values='pred')
        if p.isna().any().any():
            raise ValueError('Methods have different trial sets.')
        truth = group.groupby('group')['true'].first().reindex(p.index)
        correct = p.eq(truth, axis=0)
        for trial in p.index[~correct.any(axis=1)]:
            common.append(dict(seed=int(seed), session=session, group=trial, true=truth[trial],
                               **{f'pred_{m}': p.loc[trial, m] for m in p.columns}))
        members = ['existing', 'generic', 'session']
        if set(members) <= set(p.columns):
            for trial, predictions in p[members].iterrows():
                counts = predictions.value_counts()
                tie = int(counts.iloc[0]) == 1
                vote = predictions['existing'] if tie else counts.index[0]
                votes.append(dict(seed=int(seed), session=session, group=trial, true=truth[trial],
                                  pred=vote, tie=tie, correct=vote == truth[trial]))
    pd.DataFrame(common).to_csv(Path(output)/'common_errors.csv', index=False)
    pd.DataFrame(votes).to_csv(Path(output)/'majority_vote_predictions.csv', index=False)
    if votes:
        pd.DataFrame(votes).groupby(['seed','session']).correct.agg(['sum','count','mean']).to_csv(Path(output)/'majority_vote_accuracy.csv')
