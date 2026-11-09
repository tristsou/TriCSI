"""Convert unchanged raw CSI trial CSVs into per-window features, with raw-file checksums.
Does NOT do any training, fitting, or model selection.
"""
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from common import (CLASSES,LINKS,config_fingerprint,digest,feature_rows,load_config,
                    load_packet_groups,trial_info,trials)


def build(cfg,sessions,outpath,allow_missing):
    X=[]; y=[]; sess=[]; trial_ids=[]; win_id=[]; qc=[]
    for s in sessions:
        files=trials(cfg['data_root'],s)
        print(f'{s}: {len(files)} trial files')
        counts=Counter(trial_info(f)[0] for f in files)
        if not allow_missing and (len(files)!=50 or any(counts[c]!=5 for c in CLASSES)):
            raise ValueError(f'{s}: inventory invalid: {dict(counts)}; expected 5 trials for each of 10 classes')
        for i,p in enumerate(files,1):
            label, trial_no, orientation = trial_info(p)
            groups,extra=load_packet_groups(p,cfg,s)
            features=feature_rows(groups,cfg)
            if features.shape != (7,486): raise AssertionError('Expected seven windows × 486 features')
            for w in range(7):
                X.append(features[w]);y.append(label);sess.append(s);trial_ids.append(p.name);win_id.append(w)
            qc.append({'session':s,'trial':p.name,'label':label,'orientation':orientation,
                'sha256':digest(p),**{f'{link}_packets':len(groups[link]) for link in LINKS},
                'all_MAC_counts':json.dumps(extra['raw_counts']),
                'CSI_lengths':json.dumps(extra['lengths']),
                'WiFi_channels':json.dumps(extra['channels'])})
            if i%10==0 or i==len(files): print(f'  {i}/{len(files)}')
    o=Path(outpath);o.parent.mkdir(parents=True,exist_ok=True)
    if o.exists():raise FileExistsError(f'{o} exists; do not overwrite a frozen feature snapshot')
    np.savez_compressed(o, X=np.asarray(X,dtype=np.float32),y=np.asarray(y),
        session=np.asarray(sess),trial=np.asarray(trial_ids),window=np.asarray(win_id,dtype=np.int8),
        config_fingerprint=np.asarray(config_fingerprint(cfg)))
    with o.with_suffix('.qc.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(qc[0]));w.writeheader();w.writerows(qc)
    print(f'Saved {len(set(zip(sess,trial_ids)))} trials / {len(X)} windows to {o}')
    print('Feature SHA256:',digest(o))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',default='config.json')
    a.add_argument('--sessions',nargs='+',required=True);a.add_argument('--output',required=True)
    a.add_argument('--allow-incomplete',action='store_true')
    v=a.parse_args();build(load_config(v.config),v.sessions,v.output,v.allow_incomplete)
