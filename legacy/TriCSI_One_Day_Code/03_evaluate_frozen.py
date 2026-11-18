"""Use ONLY predict_proba on held-out complete sessions. Verify serialized-model hashes."""
import argparse,csv,json
from pathlib import Path
import joblib,numpy as np
from common import CLASSES,digest


def evaluate(frozen,features,out,requested):
    folder=Path(frozen);manifest=json.loads((folder/'manifest.json').read_text())
    with np.load(features,allow_pickle=False) as f:
        X=f['X'];sess=f['session'];y=f['y'];trial=f['trial'];win=f['window'];fp=str(f['config_fingerprint'])
    if fp != manifest['config_fingerprint']:raise ValueError('Feature configuration differs from frozen training: MAC mapping / CSI settings changed')
    if set(sess)&set(manifest['trained_sessions']):
        raise ValueError('Refusing to score training sessions as held out')
    if set(sess)!=set(requested):
        raise ValueError(f'Feature file sessions {sorted(set(sess))} do not equal requested {requested}')
    groups={}
    for ix,(s,t) in enumerate(zip(sess,trial)):
        groups.setdefault((str(s),str(t)),[]).append(ix)
    for (s,t),inds in groups.items():
        if len(inds)!=7 or not np.array_equal(np.sort(win[inds]),np.arange(7)):
            raise ValueError(f'{s} {t}: expected exactly 7 windows numbered 0..6')
        if len(set(y[inds]))!=1:raise ValueError('Mixed ground truth within one trial')
    rows=[]
    for info in manifest['models']:
        path=folder/info['file']
        if digest(path)!=info['sha256']:raise ValueError(f'Frozen model checksum mismatch: {path}')
        clf=joblib.load(path)
        if list(clf.classes_)!=info['classes']:raise ValueError('Class ordering changed!')
        probs=clf.predict_proba(X[:,info['columns']])
        for (s,t),inds in groups.items():
            p=np.mean(probs[inds],axis=0)
            pred=str(clf.classes_[int(np.argmax(p))]);actual=str(y[inds[0]])
            rows.append({'session':s,'trial_id':t,'actual':actual,'predicted':pred,
              'subset':info['subset'],'features':info['features'],'seed':info['seed'],
              'correct':int(actual==pred),'max_probability':round(float(np.max(p)),6),
              'model_sha256':info['sha256']})
    o=Path(out);o.parent.mkdir(parents=True,exist_ok=True)
    if o.exists():raise FileExistsError(f'{o} exists; avoid silent replacement of experiment results')
    with o.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Wrote {len(rows)} model × trial results to {o}')
    print('Note: models were newly fitted, even if they reproduce earlier S3 accuracy.')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--models',required=True);a.add_argument('--features',required=True)
    a.add_argument('--sessions',nargs='+',required=True);a.add_argument('--out',required=True)
    v=a.parse_args();evaluate(v.models,v.features,v.out,v.sessions)
