"""AI-assisted audit and orchestration around the ORIGINAL TriCSI extractor.

This file does not implement a replacement CSI parser, feature extractor, or
trial aggregator. Raw audit uses csv.reader; inference calls the original code.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

HERE = Path(__file__).resolve().parent
CLASS_ORDER = ['EMPTY'] + [f'Z{i}' for i in range(1, 10)]
OUTER = ['back_left', 'back_right']
ALL = ['back_left', 'middle_back', 'back_right']
SPECS = {
    'M1': dict(links=OUTER, view='full', label='Outer pair / CSI + gain + RSSI'),
    'M2': dict(links=ALL, view='full', label='Three links / CSI + gain + RSSI'),
    'M3': dict(links=OUTER, view='csi', label='Outer pair / CSI + gain, no RSSI'),
    'M4': dict(links=ALL, view='csi', label='Three links / CSI + gain, no RSSI'),
    'M5': dict(links=OUTER, view='rssi', label='Outer pair / RSSI only'),
    'M6': dict(links=ALL, view='rssi', label='Three links / RSSI only'),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ast_sha(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    return hashlib.sha256(ast.dump(ast.parse(text), include_attributes=False).encode()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str), encoding='utf-8')


def fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment():
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__, scipy=scipy.__version__)


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original(root):
    path = Path(root) / 'tricsi_3tx_modeling_bundle' / 'tricsi_3tx_modeling.py'
    if not path.is_file():
        raise FileNotFoundError(f'Original module missing: {path}. Do not substitute reconstructed parsers.')
    reference = HERE / 'reference_sources' / 'tricsi_3tx_modeling.py'
    if ast_sha(path) != ast_sha(reference):
        raise ValueError(f'Original extractor differs from the inspected source: {path}. '
                         'Stop and compare it with reference_sources; no automatic replacement was made.')
    return import_path('_tricsi_original_study', path)


def load_trusted(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            return joblib.load(path)
    except InconsistentVersionWarning as e:
        raise RuntimeError('scikit-learn version differs from the saved model. Use its original '
                           'working virtual environment. Do not upgrade or refit to force a match. '
                           f'Detail: {e}') from e


def legacy_path(root):
    return Path(root) / 'results/3tx_dev_comparison/frozen_3tx_model.joblib'


def legacy_bundle(path):
    b = load_trusted(path)
    if b.get('format_version') != 2 or b.get('classes') != CLASS_ORDER:
        raise ValueError('Unexpected original frozen-model format or class order.')
    if b.get('training_sessions') != ['3TX_S1', '3TX_S2']:
        raise ValueError('Expected original S1+S2 frozen bundle.')
    for name, links in [('2tx_outer', OUTER), ('3tx_all', ALL)]:
        if b['subset_links'].get(name) != links or name not in b['models_by_subset']:
            raise ValueError(f'Unexpected historical link mapping for {name}.')
    return b


def matrix(data, links, view):
    pieces, selected = [], []
    for link in links:
        names = data.feature_names_by_link[link]
        bare = [x.rsplit('__', 1)[-1] for x in names]
        keep = [i for i, n in enumerate(bare)
                if view == 'full' or (view == 'csi' and not n.startswith('rssi_'))
                or (view == 'rssi' and n.startswith('rssi_'))]
        expected = {'full': 162, 'csi': 159, 'rssi': 3}[view]
        if len(keep) != expected:
            raise ValueError(f'Unexpected {view} feature count for {link}: {len(keep)}')
        pieces.append(data.features_by_link[link][:, keep])
        selected.extend(names[i] for i in keep)
    return np.concatenate(pieces, axis=1), selected


def integer(value):
    x = float(value)
    if not math.isfinite(x) or x != int(x):
        raise ValueError('not a finite integer')
    return int(x)


def audit_session(core, directory, output):
    """Read-only audit. Unused bad len metadata is warned, not a rejection rule.

    Non-finite IQ/RSSI rows are logged; the original extractor already excludes
    them. No new cleanup file is written and original bytes are never modified.
    Structural/identity errors stop execution after the audit has been saved.
    """
    directory, output = Path(directory).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    issues, trial_rows, identities, fatal = [], [], [], []
    trial_files = core.discover_trial_files(directory)
    iq = [f'iq_{i:03d}' for i in range(128)]
    mac_to_link = {m: l for l, m in core.LINK_TO_MAC.items()}
    total = valid_total = rejected_total = 0
    manifest = directory / 'session_manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding='utf-8-sig'))
        if m.get('session', directory.name) != directory.name:
            fatal.append('Manifest session name differs from directory name.')
        if 'mac_to_link' in m and m['mac_to_link'] != mac_to_link:
            fatal.append('Manifest MAC/link mapping differs from original module.')
    else:
        issues.append(dict(file='session_manifest.json', line=0, severity='warning',
                           reason='Manifest missing; original default grid would be used.'))
    for path in trial_files:
        match = re.fullmatch(r'(EMPTY|Z[1-9])_trial(\d+)_(NA|N|E|S|W)', path.stem)
        if not match:
            fatal.append(f'Unexpected trial filename: {path.name}')
            continue
        zone, trial, facing = match.group(1), int(match.group(2)), match.group(3)
        identities.append((zone, trial))
        expected_facing = 'NA' if zone == 'EMPTY' else {1:'N',2:'E',3:'S',4:'W',5:'N'}.get(trial)
        if facing != expected_facing:
            fatal.append(f'{path.name}: orientation differs from five-trial protocol.')
        counts, accepted, previous, times = Counter(), Counter(), {}, {l: [] for l in ALL}
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                if len(header) != len(set(header)):
                    fatal.append(f'{path.name}: duplicate column names.')
                if {c for c in header if c.startswith('iq_')} != set(iq):
                    fatal.append(f'{path.name}: expected exactly the 128 saved IQ columns.')
                missing = set(['mac','rssi', *iq]) - set(header)
                if missing:
                    fatal.append(f'{path.name}: missing columns {sorted(missing)[:6]}')
                    continue
                for row in reader:
                    line = reader.line_num
                    total += 1
                    if len(row) != len(header):
                        fatal.append(f'{path.name}:{line}: {len(row)} fields, expected {len(header)}.')
                        issues.append(dict(file=path.name, line=line, severity='fatal', reason='CSV field-count mismatch'))
                        continue
                    d = dict(zip(header, row))
                    mac = core.normalize_mac(d['mac'])
                    if mac not in mac_to_link:
                        fatal.append(f'{path.name}:{line}: unknown MAC {mac}')
                        continue
                    link = mac_to_link[mac]
                    counts[link] += 1
                    try:
                        if d.get('zone',zone).upper() != zone or integer(d.get('trial',trial)) != trial:
                            raise ValueError('trial identity')
                        if d.get('session',directory.name) != directory.name:
                            raise ValueError('session name')
                        if d.get('orientation',facing).upper() != facing:
                            raise ValueError('orientation')
                        if d.get('link',link).lower() != link:
                            raise ValueError('link/MAC mapping')
                    except (ValueError, TypeError, OverflowError) as e:
                        fatal.append(f'{path.name}:{line}: label metadata mismatch ({e}).')
                    try:
                        raw = [float(d[c]) for c in iq]
                        rssi = float(d['rssi'])
                        good = all(math.isfinite(x) for x in [*raw, rssi])
                    except (ValueError, TypeError, OverflowError):
                        good = False
                    if not good:
                        rejected_total += 1
                        issues.append(dict(file=path.name, line=line, severity='excluded_by_original_extractor',
                                           reason='Non-finite/non-numeric saved IQ or RSSI'))
                    else:
                        accepted[link] += 1
                        valid_total += 1
                        if any(x != int(x) for x in raw):
                            fatal.append(f'{path.name}:{line}: IQ values are not integers as collected.')
                    if 'len' in d:
                        try:
                            if integer(d['len']) not in (128,256,384):
                                raise ValueError('unexpected declared CSI length')
                        except (ValueError, TypeError, OverflowError):
                            issues.append(dict(file=path.name, line=line, severity='unused_metadata_warning',
                                               reason='Invalid/unexpected len metadata; not used by original extractor'))
                    time_col = 'host_time_s' if 'host_time_s' in d else 'local_timestamp'
                    if time_col in d:
                        try:
                            value = float(d[time_col])
                            if not math.isfinite(value):
                                raise ValueError('nonfinite time')
                            if value <= previous.get(link, -float('inf')):
                                issues.append(dict(file=path.name, line=line, severity='timing_warning',
                                                   reason='Repeated or nonincreasing timestamp; original sorts by timestamp'))
                            previous[link] = value
                            if time_col == 'host_time_s':
                                times[link].append(value)
                        except (ValueError, TypeError):
                            fatal.append(f'{path.name}:{line}: invalid sorting timestamp.')
            except (csv.Error, StopIteration) as e:
                fatal.append(f'{path.name}: CSV read failed: {e}')
        for link in ALL:
            if accepted[link] < 64:
                fatal.append(f'{path.name}: {link} has {accepted[link]} valid packets; need 64.')
            span = max(times[link]) - min(times[link]) if len(times[link]) > 1 else 0
            trial_rows.append(dict(trial_file=path.name, zone=zone, trial=trial, link=link,
                                   raw_packets=counts[link], valid_packets=accepted[link],
                                   host_span_s=span, estimated_rate_hz=(len(times[link])-1)/span if span>0 else None))
    expected = {(z,t) for z in CLASS_ORDER for t in range(1,6)}
    if len(identities) != 50 or set(identities) != expected or len(set(identities)) != len(identities):
        fatal.append('Expected exactly five distinct trials per class and 50 files.')
    pd.DataFrame(issues, columns=['file','line','severity','reason']).to_csv(output/'row_issues.csv',index=False)
    pd.DataFrame(trial_rows).to_csv(output/'trial_link_counts.csv',index=False)
    report = dict(session=directory.name, trial_files=len(trial_files), raw_rows=total,
                  usable_feature_rows=valid_total, rows_excluded_by_original_extractor=rejected_total,
                  issue_counts=dict(Counter(x['severity'] for x in issues)), fatal_errors=fatal,
                  raw_csv_sha256=core.session_data_sha256(directory),
                  individual_trial_sha256={p.name:sha(p) for p in trial_files},
                  manifest_sha256=sha(manifest) if manifest.exists() else None,
                  policy='Original raw bytes unchanged. len metadata never compared to saved IQ count.')
    write_json(output/'audit.json',report)
    if fatal:
        raise ValueError('Acquisition audit failed. Inspect '+str(output/'audit.json')+'; first issues: '+str(fatal[:3]))
    return report


def read_features(core, directory, output, config):
    audit = audit_session(core, directory, output)
    data = core.extract_session_features(directory, **config)
    if data.session_name != Path(directory).name:
        raise ValueError('Session identity mismatch.')
    if len(data.metadata) != 350:
        raise ValueError('Expected seven windows for each of 50 trials.')
    if not all(np.isfinite(v).all() for v in data.features_by_link.values()):
        raise ValueError('Nonfinite extracted feature values.')
    return data, audit


def score(core, model, data, links, view, model_id, lane):
    x, names = matrix(data, links, view)
    if model.n_features_in_ != x.shape[1] or set(model.classes_) != set(CLASS_ORDER):
        raise ValueError(f'{model_id}: model feature dimensions or classes differ.')
    metrics, pred, confusion = core.evaluate_classifier(model, x, data.metadata, data.zone_coords)
    pred.insert(0,'model',model_id)
    pred.insert(0,'lane',lane)
    identity = data.metadata.drop_duplicates('trial_uid').set_index('trial_uid')
    pred['orientation'] = pred.trial_uid.map(identity.orientation)
    row = dict(session=data.session_name, model=model_id, lane=lane, **metrics)
    occupied = pred.true_zone != 'EMPTY'
    row['occupied_to_empty_count'] = int((occupied & (pred.predicted_zone=='EMPTY')).sum())
    row['occupied_to_empty_rate'] = float((pred.loc[occupied,'predicted_zone']=='EMPTY').mean())
    return row,pred,confusion
