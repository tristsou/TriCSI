"""Train six specified ExtraTrees models (2/3 links x full/CSI/RSSI) using S1+S2 ONLY.
Freeze before any new held-out S4/S5 labels are inspected. Never overwrite this folder.
"""
import argparse,json,platform,sys
from datetime import datetime, timezone
from pathlib import Path
import joblib,numpy as np, sklearn
from sklearn.ensemble import ExtraTreesClassifier
from common import CLASSES,digest,subset_columns


def train(features,outdir,seeds):
    d=Path(outdir)
    if d.exists(): raise FileExistsError(f'Freeze destination exists, refusing overwrite: {d}')
    with np.load(features,allow_pickle=False) as f:
        X=f['X'];y=f['y'];sessions=f['session']; trials=f['trial'];cfg_fp=str(f['config_fingerprint'])
    used=np.isin(sessions,['3TX_S1','3TX_S2'])
    unexpected=set(sessions)-{'3TX_S1','3TX_S2'}
    if unexpected:raise ValueError(f'Training feature file must contain ONLY 3TX_S1 and 3TX_S2. Found {unexpected}')
    if set(sessions) != {'3TX_S1','3TX_S2'}:raise ValueError('Both training sessions required')
    expected=50*7*2
    if len(X)!=expected or np.any(np.isnan(X)):
        raise ValueError(f'Expected {expected} finite windows, got {len(X)}')
    if len(set(zip(sessions,trials)))!=100:raise ValueError('Expected 100 distinct development trials')
    d.mkdir(parents=True,exist_ok=False)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'role':'NEW POST-REVIEW MODELS; NOT ORIGINAL S3-FROZEN ARTIFACTS',
      'data_sha256':digest(features),'config_fingerprint':cfg_fp,'trained_sessions':['3TX_S1','3TX_S2'],
      'training_windows':len(X),'distinct_trials':100,'python':sys.version,'sklearn':sklearn.__version__,
      'seeds':seeds,'hyperparams':{'n_estimators':400,'max_features':'sqrt','min_samples_leaf':2,
                                 'class_weight':'balanced'},'models':[]}
    for seed in seeds:
        for subset in ['2tx_outer','3tx_all']:
            for typ in ['full','csi','rssi']:
                cols=subset_columns(subset,typ)
                mdl=ExtraTreesClassifier(n_estimators=400,max_features='sqrt',min_samples_leaf=2,
                                         class_weight='balanced',random_state=seed,n_jobs=-1)
                mdl.fit(X[:,cols],y)
                path=d/f'extratrees_{subset}_{typ}_seed{seed}.joblib'
                joblib.dump(mdl,path,compress=3)
                manifest['models'].append({'file':path.name,'sha256':digest(path),'subset':subset,
                              'features':typ,'columns':cols.tolist(),'seed':seed,
                              'classes':list(mdl.classes_)})
                print('FROZEN',path.name,'n_features',len(cols))
    (d/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Created',d,'\nKeep this folder read-only. Do not use S3/S4/S5 to select models.')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--features',required=True);a.add_argument('--out',required=True)
    a.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    v=a.parse_args();train(v.features,v.out,v.seeds)
