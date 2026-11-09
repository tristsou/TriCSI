"""Shared ESP32 CSI file parsing, feature extraction and reproducibility utilities.
Independent implementation based on the uploaded TriCSI methods; NOT the original pipeline.
"""
from __future__ import annotations
import csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

CLASSES = ['EMPTY'] + [f'Z{i}' for i in range(1,10)]
LINKS = ['BACK_LEFT', 'MIDDLE_BACK', 'BACK_RIGHT']
TRIAL_RE = re.compile(r'^(EMPTY|Z[1-9])_trial(\d+)_([A-Z]+)\.csv$', re.I)
MAC_RE = re.compile(r'^[0-9a-f]{2}(?::[0-9a-f]{2}){5}$', re.I)
INT_RE = re.compile(r'[-+]?\d+')
ESP_HEADER = 'type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data'.split(',')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def load_config(path):
    p = Path(path).resolve()
    d = json.loads(p.read_text(encoding='utf-8'))
    root = Path(d['data_root'])
    if not root.is_absolute():
        root = (p.parent / root).resolve()
    d['data_root'] = str(root)
    bins = d['populated_indices']
    if len(bins) != 52 or len(set(bins)) != 52 or any(not 0 <= int(x) < 64 for x in bins):
        raise ValueError('populated_indices must contain 52 distinct indices in 0..63. VERIFY against original pipeline.')
    if d.get('packet_csi_offset', 0) < 0:
        raise ValueError('packet_csi_offset must be nonnegative')
    return d


def trials(root, session):
    folder = Path(root) / session
    if not folder.exists():
        raise FileNotFoundError(f'Missing session directory: {folder}')
    found = {}
    for p in folder.rglob('*.csv'):
        mt = TRIAL_RE.match(p.name)
        if not mt:
            continue
        key = p.name.lower()
        if key in found:
            raise ValueError(f'Duplicate file name for session {session}: {p} / {found[key]}')
        found[key] = p
    return sorted(found.values(), key=lambda x: x.name)


def trial_info(path):
    m = TRIAL_RE.match(Path(path).name)
    if not m: raise ValueError(f'Unexpected filename {path}')
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


def source_map(cfg, session):
    return cfg.get('macs_by_session', {}).get(session, cfg.get('macs', {}))


def parse_csi(s):
    if s is None: return None
    s = str(s)
    i, j = s.find('['), s.rfind(']')
    if i < 0 or j < i: return None
    v = np.fromiter((int(x) for x in INT_RE.findall(s[i+1:j])), dtype=np.float32)
    return v if v.size >= 128 else None


def raw_rows(path):
    """Yield dicts from headered and common headerless ESP32-CSI-Tool CSV files.
    Supports serial lines containing CSI_DATA; intentionally fails closed for unknown formats.
    """
    with Path(path).open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or not any(c.strip() for c in row): continue
            low = [x.strip().lower() for x in row]
            if 'data' in low and ('mac' in low or 'rssi' in low):
                header = low
                continue
            if row[0].strip().upper() != 'CSI_DATA':
                continue
            if header is not None:
                if len(row) > len(header):
                    # e.g. an unquoted CSI payload: collapse trailing items to the data cell.
                    try:
                        data_i = header.index('data')
                        if data_i == len(header)-1: row = row[:data_i] + [','.join(row[data_i:])]
                    except ValueError: pass
                d = dict(zip(header, row))
            elif len(row) >= 25 and MAC_RE.fullmatch(row[2].strip()):
                d = dict(zip(ESP_HEADER, row[:24] + [','.join(row[24:])]))
            else:
                # Common project variant with id removed; search MAC and next RSSI.
                mac_idxs = [i for i,v in enumerate(row[:8]) if MAC_RE.fullmatch(v.strip())]
                if not mac_idxs: continue
                mi = mac_idxs[0]
                d = {'mac':row[mi], 'rssi':row[mi+1] if mi+1 < len(row) else '',
                     'data':','.join(row[mi+2:])}
                # If no quoted data after the mac, search the entire reconstructed row.
                if '[' not in d['data']: d['data'] = ','.join(row)
            if 'data' not in d or 'mac' not in d: continue
            mac = str(d['mac']).strip().lower()
            if not MAC_RE.fullmatch(mac): continue
            try: rssi = float(d.get('rssi',''))
            except (ValueError, TypeError): continue
            vec = parse_csi(d['data'])
            if vec is None: continue
            try: ts = float(d.get('local_timestamp','nan'))
            except (ValueError, TypeError): ts = float('nan')
            yield {'mac':mac, 'rssi':rssi, 'iq':vec, 'timestamp':ts, 'csi_len':len(vec),
                   'channel':str(d.get('channel','')), 'first_word':str(d.get('first_word','')),
                   'sig_mode':str(d.get('sig_mode','')), 'bandwidth':str(d.get('bandwidth',''))}


def load_packet_groups(path, cfg, session, require_all=True):
    mapping = source_map(cfg, session)
    if not mapping or any(k not in mapping or 'FILL' in str(mapping[k]) for k in LINKS):
        raise ValueError(f'Enter all 3 physical transmitter MAC addresses in config.json (or macs_by_session.{session}).')
    inv = {str(v).lower():k for k,v in mapping.items()}
    if len(inv) != len(mapping): raise ValueError('Duplicated MAC assignment in config')
    groups = defaultdict(list)
    lengths, channels, all_macs = Counter(),Counter(),Counter()
    for r in raw_rows(path):
        lengths[r['csi_len']] += 1
        all_macs[r['mac']] += 1
        if r['channel']: channels[r['channel']] += 1
        if r['mac'] in inv: groups[inv[r['mac']]].append(r)
    if require_all and any(len(groups[link]) < 64 for link in LINKS):
        raise ValueError(f'{path}: fewer than 64 valid packets on at least one link: '+str({k:len(groups[k]) for k in LINKS}))
    return groups, {'raw_counts':dict(all_macs), 'lengths':dict(lengths), 'channels':dict(channels)}


def summarise_vector(a):
    a = np.asarray(a, dtype=np.float32)
    return (np.median(a, axis=0),
            np.percentile(a,75,axis=0)-np.percentile(a,25,axis=0),
            np.mean(np.abs(np.diff(a,axis=0)),axis=0))


def packet_features(rows, cfg):
    """7x162: (52 subcarrier medians,IQR,mean-absolute-change) + gain (3) + RSSI (3)."""
    off = cfg.get('packet_csi_offset',0)
    selected = np.array(cfg['populated_indices'],dtype=int)
    arr = np.stack([p['iq'][off:off+128] for p in rows]).reshape(-1,64,2)
    # I,Q order does not matter for amplitude. ESP32 typically uses imaginary,real.
    amplitude = np.log1p(np.sqrt(np.sum(arr*arr, axis=-1)))[:,selected]
    gain = np.median(amplitude,axis=1)
    shape = amplitude-gain[:,None]
    rssi = np.array([p['rssi'] for p in rows], dtype=np.float32)
    if len(rows)<64: raise ValueError('Need at least 64 packets to construct a window')
    starts = np.rint(np.linspace(0,len(rows)-64,7)).astype(int)
    out=[]
    for st in starts:
        a = shape[st:st+64]; g = gain[st:st+64]; r = rssi[st:st+64]
        med, iqr, change = summarise_vector(a)
        gm,gi,gc = summarise_vector(g)
        rm,ri,rc = summarise_vector(r)
        f=np.concatenate([med, iqr, change, np.array([gm,gi,gc,rm,ri,rc],dtype=np.float32)])
        assert f.size == 162
        out.append(f)
    return np.stack(out).astype(np.float32)


def feature_rows(groups,cfg):
    pieces=[packet_features(groups[k],cfg) for k in LINKS]
    return np.concatenate(pieces,axis=1)  # [7,486]


def subset_columns(subset, features):
    idx = []
    use_links = ['BACK_LEFT','BACK_RIGHT'] if subset=='2tx_outer' else LINKS
    use_parts = {'full':list(range(162)), 'csi':list(range(156))+list(range(156,159)),
                 'rssi':list(range(159,162))}[features]
    # csi includes the three CSI packet-wide gain summaries, excludes RSSI.
    for name in use_links:
        offset = LINKS.index(name)*162
        idx.extend(offset+v for v in use_parts)
    return np.asarray(idx,dtype=int)


def config_fingerprint(cfg):
    keys = ['populated_indices','packet_csi_offset','macs']
    relevant = {k:cfg.get(k) for k in keys}
    relevant['development_session_macs'] = {s:cfg.get('macs_by_session',{}).get(s) for s in ['3TX_S1','3TX_S2']}
    # Newly observed test-session MAC addresses can be mapped later without altering frozen feature semantics.
    return hashlib.sha256(json.dumps(relevant,sort_keys=True).encode()).hexdigest()
