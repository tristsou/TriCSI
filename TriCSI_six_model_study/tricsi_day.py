"""AI-assisted one-day orchestration using the original TriCSI codebase.

Run from the main Windows project directory. Existing data/models are read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from study_core import (
    ALL, CLASS_ORDER, HERE, OUTER, SPECS, ast_sha, audit_session, environment,
    fresh, import_path, legacy_bundle, legacy_path, load_trusted, matrix, original,
    read_features, score, sha, utc, write_json,
)

DEFAULT_RUN = Path('results/3tx_six_model_20261030_v1')


def collector_info(root,core,bundle):
    path=root/'collect_three_link_trials.py'
    c=import_path('_tricsi_collector_inspect',path)
    if c.MAC_TO_LINK != {v:k for k,v in core.LINK_TO_MAC.items()} or list(c.CLASS_ORDER)!=CLASS_ORDER:
        raise ValueError('Collector and original extractor disagree on MACs/classes.')
    if c.ZONE_CENTERS != bundle['zone_coords']:
        raise ValueError('Collector grid differs from frozen model.')
    schedule=c.build_schedule(trials_per_zone=5,trial_seconds=8,empty_baseline_seconds=12,seed=312)
    if len(schedule)!=50 or schedule[0]['zone']!='EMPTY' or schedule[0]['seconds']!=12:
        raise ValueError('Unexpected original collector schedule.')
    facings=[next(x['orientation'] for x in schedule if x['zone']=='Z1' and x['trial']==t)
             for t in range(1,6)]
    if facings!=['N','E','S','W','N']:
        raise ValueError('Unexpected original orientation sequence.')
    return dict(path=str(path.resolve()),sha256=sha(path),ast_sha256=ast_sha(path),
                default_port=c.DEFAULT_PORT,baud=c.DEFAULT_BAUD,csv_column_count=len(c.CSV_HEADER),
                trial_count=50,ordinary_seconds=8,first_empty_seconds=12,settle_seconds=5,
                orientations=facings,reference_ast_matches=ast_sha(path)==ast_sha(HERE/'reference_sources/collect_three_link_trials.py'))


def inspect(a):
    out=fresh(a.output)
    write_json(out/'environment.json',environment())
    core=original(a.project_root)
    path=legacy_path(a.project_root)
    b=legacy_bundle(path)
    info=collector_info(a.project_root,core,b)
    modules={}
    for rel in ['tricsi_3tx_modeling_bundle/tricsi_3tx_modeling.py',
                'tricsi_3tx_modeling_bundle/compare_and_freeze_3tx.py',
                'tricsi_3tx_modeling_bundle/evaluate_frozen_3tx.py',
                'TriCSI_review_response/tricsi_3tx_modeling.py']:
        f=a.project_root/rel
        modules[rel]={'exists':f.exists()}
        if f.exists():
            modules[rel].update(sha256=sha(f),ast_sha256=ast_sha(f))
    review=a.project_root/'TriCSI_review_response/tricsi_3tx_modeling.py'
    if review.exists() and ast_sha(review)!=ast_sha(a.project_root/'tricsi_3tx_modeling_bundle/tricsi_3tx_modeling.py'):
        raise ValueError('Recent review-response extractor differs from the original; inspect module hashes.')
    model_info={name:dict(params=model.get_params(),n_features=int(model.n_features_in_))
                for name,model in b['models_by_subset'].items()}
    report=dict(created_utc=utc(),original_model=str(path.resolve()),original_model_sha256=sha(path),
                models=model_info,feature_config=b['feature_config'],training_sessions=b['training_sessions'],
                training_data_sha256=b['training_data_sha256'],collector=info,modules=modules,
                findings=['Historical outer seed 2029 and all-three seed 2032 in inspected artifact; actual values are listed above.',
                          'Feature extractor ignores len metadata and consumes exactly iq_000..iq_127.',
                          'Recent CSI-without-RSSI condition includes three packet-gain features per link.',
                          'Original collector writes schedule/manifest before a preflight-only exit.'])
    write_json(out/'inspection.json',report)
    print('INSPECTION PASS')
    print('Environment:',environment())
    print('Historical seeds:',b['models_by_subset']['2tx_outer'].random_state,b['models_by_subset']['3tx_all'].random_state)
    print('Collector:',info['default_port'],info['baud'],info['csv_column_count'],'CSV columns')
    print('Saved',out)


def historical_scores(core,b,data):
    rows,frames=[],[]
    for name,model_id in [('2tx_outer','H1'),('3tx_all','H2')]:
        row,pred,_=score(core,b['models_by_subset'][name],data,b['subset_links'][name],
                         'full',model_id,'historical_frozen')
        rows.append(row);frames.append(pred)
    return pd.DataFrame(rows),pd.concat(frames,ignore_index=True)


def reproduce(a):
    out=fresh(a.output)
    core=original(a.project_root)
    path=legacy_path(a.project_root)
    b=legacy_bundle(path)
    config=b['feature_config']
    data,audit=read_features(core,a.project_root/'data_3tx/raw/3TX_S3',out/'raw_audit',config)
    if data.zone_coords!=b['zone_coords']:
        raise ValueError('S3 grid differs from frozen grid.')
    if audit['raw_csv_sha256'] in b['training_data_sha256'].values():
        raise ValueError('S3 bytes duplicate training data.')
    metrics,pred=historical_scores(core,b,data)
    metrics.to_csv(out/'historical_metrics.csv',index=False)
    pred.to_csv(out/'historical_trial_predictions.csv',index=False)
    comparison=[]
    old=a.project_root/'results/3tx_s3_final/s3_trial_predictions.csv'
    if old.is_file():
        saved=pd.read_csv(old)
        for subset,model_id in [('2tx_outer','H1'),('3tx_all','H2')]:
            left=saved[saved.subset==subset][['trial_uid','true_zone','predicted_zone']]
            right=pred[pred.model==model_id][['trial_uid','true_zone','predicted_zone']]
            c=left.merge(right,on='trial_uid',how='outer',suffixes=('_saved','_reproduced'),
                         validate='one_to_one',indicator=True)
            c['model']=model_id
            c['matches']=(c['_merge']=='both')&(c.true_zone_saved==c.true_zone_reproduced)&(c.predicted_zone_saved==c.predicted_zone_reproduced)
            comparison.append(c)
        pd.concat(comparison).to_csv(out/'saved_prediction_comparison.csv',index=False)
    actual=metrics.set_index('model').trial_accuracy.to_dict()
    matches=all(np.isclose(actual[k],v) for k,v in {'H1':.70,'H2':.84}.items())
    matches=matches and (not comparison or all(c.matches.all() for c in comparison))
    core_pair=pred[pred.model=='H1'].merge(pred[pred.model=='H2'],on=['trial_uid','true_zone'],suffixes=('_outer','_3tx'))
    core_pair['gain']=core_pair.correct_3tx-core_pair.correct_outer
    interval=core.stratified_paired_bootstrap(core_pair,'gain',iterations=5000,seed=2026)
    write_json(out/'reproduction.json',dict(created_utc=utc(),expected={'H1':.70,'H2':.84},
                 observed=actual,historical_accuracy_matches=matches,saved_predictions_available=old.exists(),
                 accuracy_gain=interval,environment=environment(),original_model_sha256=sha(path),
                 raw_csv_sha256=audit['raw_csv_sha256'],feature_config=config,
                 note='No fitting or parameter adjustment. This is historical reproduction, not a new test.'))
    print(metrics[['model','trial_accuracy','occupied_trial_accuracy','empty_recall']].to_string(index=False))
    if not matches:
        raise RuntimeError('Historical reproduction differs. Preserve outputs and diagnose raw/model/code/environment hashes; do not tune to match.')
    print('HISTORICAL REPRODUCTION PASS: H1=70%, H2=84%. Saved',out)


def freeze(a):
    root=a.project_root
    for name in a.future_sessions:
        if not name.startswith('3TX_') or name in ['3TX_S1','3TX_S2','3TX_S3'] or Path(name).name!=name:
            raise ValueError('Future session names must be new simple 3TX_ names.')
        if list((root/'data_3tx/raw'/name).glob('*_trial*.csv')):
            raise ValueError(f'{name} already contains trial data. Choose genuinely future recordings, not a retrospective frozen test.')
    if len(set(a.future_sessions))!=2:
        raise ValueError('Specify exactly two different future sessions.')
    out=fresh(a.run_dir)
    core=original(root)
    path=legacy_path(root)
    b=legacy_bundle(path)
    if b['feature_config']!={'window_packets':64,'windows_per_trial':7}:
        raise ValueError('Historical feature configuration differs from inspected 64/7 protocol.')
    collector=collector_info(root,core,b)
    datasets=[];audits={}
    for name in ('3TX_S1','3TX_S2'):
        data,audit=read_features(core,root/'data_3tx/raw'/name,out/'training_audit'/name,b['feature_config'])
        if audit['raw_csv_sha256']!=b['training_data_sha256'].get(name):
            raise ValueError(f'{name} bytes differ from original frozen-model training hash. No automatic refitting/relabeling allowed.')
        if data.zone_coords!=b['zone_coords']:
            raise ValueError('Training grid differs from original frozen model.')
        datasets.append(data);audits[name]=audit
    seed=2026
    models_dir=out/'models';models_dir.mkdir()
    manifest=dict(format='tricsi-six-v1',status='training',started_utc=utc(),
                  primary_replication_contrast='H2 minus H1 (unchanged original frozen models)',
                  matched_seed=seed,feature_config=b['feature_config'],classes=CLASS_ORDER,
                  zone_coords=b['zone_coords'],training_sessions=['3TX_S1','3TX_S2'],
                  training_data_sha256=b['training_data_sha256'],environment=environment(),collector=collector,
                  future_sessions=[dict(session=n,collection_seed=312+i) for i,n in enumerate(a.future_sessions)],
                  matched_feature_contrasts=['M2-M1','M4-M3','M6-M5','(M4-M3)-(M6-M5)','M2-M6','M1-M5'],
                  definition='CSI-only retains gain. Fixed-room, single-participant, short-term session replication.',
                  artifacts={},specs=SPECS,
                  original_module_sha256=sha(root/'tricsi_3tx_modeling_bundle/tricsi_3tx_modeling.py'),
                  original_module_ast_sha256=ast_sha(root/'tricsi_3tx_modeling_bundle/tricsi_3tx_modeling.py'),
                  toolkit_sha256={f.name:sha(f) for f in HERE.glob('*.py')})
    write_json(out/'manifest_in_progress.json',manifest)
    metadata=pd.concat([d.metadata for d in datasets],ignore_index=True)
    rows=[]
    for model_id,spec in SPECS.items():
        matrices=[matrix(d,spec['links'],spec['view']) for d in datasets]
        names=matrices[0][1]
        if matrices[1][1]!=names:
            raise ValueError('Training feature names differ.')
        x=np.concatenate([m[0] for m in matrices],axis=0)
        model=core.make_classifier(seed=seed,n_estimators=400)
        model.fit(x,metadata.zone)
        file=models_dir/(model_id+'.joblib')
        joblib.dump(model,file,compress=3)
        manifest['artifacts'][model_id]=dict(file='models/'+file.name,sha256=sha(file),
                    seed=seed,n_features=x.shape[1],n_training_windows=len(x),
                    params=model.get_params(),feature_names=names)
        rows.append(dict(model=model_id,**spec,n_features=x.shape[1],seed=seed))
        print(f'Frozen {model_id}: {x.shape[1]} features, {len(x)} windows, seed {seed}',flush=True)
    historical=out/'original_frozen_3tx_model.joblib'
    shutil.copyfile(path,historical)
    manifest['historical']=dict(file=historical.name,sha256=sha(historical),
                  original_path=str(path.resolve()),
                  seeds={key:int(b['models_by_subset'][name].random_state)
                         for key,name in [('H1','2tx_outer'),('H2','3tx_all')]})
    pd.DataFrame(rows).to_csv(out/'model_definitions.csv',index=False)
    # Validate all saved bytes before committing the prospective manifest.
    for key,entry in manifest['artifacts'].items():
        saved=load_trusted(out/entry['file'])
        if saved.n_features_in_!=entry['n_features']:
            raise ValueError('Serialized feature dimension mismatch.')
    for name in a.future_sessions:
        if list((root/'data_3tx/raw'/name).glob('*_trial*.csv')):
            raise ValueError('Future-session data appeared during fitting. No prospective manifest committed.')
    manifest.update(status='frozen',frozen_utc=utc(),frozen_unix=time.time())
    write_json(out/'manifest.json',manifest)
    (out/'manifest_sha256.txt').write_text(sha(out/'manifest.json')+'\n',encoding='ascii')
    print('FREEZE PASS. All six models and original historical models saved; no future session was read.')
    print('Manifest:',out/'manifest.json')


def frozen_run(a):
    run=a.run_dir.resolve()
    manifest_path=run/'manifest.json'
    expected=(run/'manifest_sha256.txt').read_text(encoding='ascii').strip()
    if sha(manifest_path)!=expected:
        raise ValueError('Prospective manifest changed after freezing.')
    m=json.loads(manifest_path.read_text())
    if m.get('format')!='tricsi-six-v1' or m.get('status')!='frozen':
        raise ValueError('No completed frozen run found.')
    if set(m['artifacts'])!=set(SPECS) or m['specs']!=SPECS:
        raise ValueError('Six-model definitions differ from this toolkit.')
    for name,digest in m['toolkit_sha256'].items():
        if sha(HERE/name)!=digest:
            raise ValueError(f'Toolkit changed after freezing: {name}. Preserve version and investigate.')
    if environment()!=m['environment']:
        raise ValueError('Python/package versions differ from freezing environment. Restore that environment.')
    core=original(a.project_root)
    if ast_sha(a.project_root/'tricsi_3tx_modeling_bundle/tricsi_3tx_modeling.py')!=m['original_module_ast_sha256']:
        raise ValueError('Original extraction code changed after freezing.')
    for entry in list(m['artifacts'].values())+[m['historical']]:
        if sha(run/entry['file'])!=entry['sha256']:
            raise ValueError('A saved model file changed after freezing.')
    return run,m,core


def collect(a):
    run,m,core=frozen_run(a)
    planned={s['session']:s for s in m['future_sessions']}
    if a.session not in planned:
        raise ValueError('Session was not registered before freezing.')
    path=a.project_root/'collect_three_link_trials.py'
    if sha(path)!=m['collector']['sha256']:
        raise ValueError('Original collector changed after freezing.')
    raw=a.project_root/'data_3tx/raw'/a.session
    receipt=run/('collection_'+a.session+'.json')
    command=[sys.executable,str(path),'--session',a.session,'--seed',str(planned[a.session]['collection_seed']),
             '--raw-dir',str(a.project_root/'data_3tx/raw'),'--trials-per-zone','5','--seconds','8',
             '--empty-baseline-seconds','12','--settle','5','--preflight-seconds','30',
             '--preflight-min-packets','100']
    if a.port:
        command.extend(['--port',a.port])
    if raw.exists() and any(raw.iterdir()):
        if not a.resume or not receipt.exists():
            raise ValueError('Existing session data require --resume AND this run\'s original collection receipt.')
        prior=json.loads(receipt.read_text())
        if prior['command']!=command:
            raise ValueError('Resume collector options differ from original receipt.')
        print('Resuming ORIGINAL collector: it skips every existing trial file. Interrupted partial files are not silently overwritten.')
    elif a.resume:
        raise ValueError('Nothing to resume.')
    if not receipt.exists():
        write_json(receipt,dict(session=a.session,started_utc=utc(),started_unix=time.time(),
                               command=command,collector_sha256=sha(path),manifest_sha256=sha(run/'manifest.json')))
    print('Starting unchanged original collector. No furniture/antenna/layout changes requested.',flush=True)
    result=subprocess.run(command,cwd=a.project_root)
    if result.returncode:
        raise RuntimeError(f'Collector exited {result.returncode}; original data preserved.')
    # q may finish the process without all 50 trials. Audit catches this.
    audit_session(core,raw,run/'collection_audits'/(a.session+'_'+str(time.time_ns())))
    print('COLLECTION AUDIT PASS:',a.session)


def evaluate(a):
    run,m,core=frozen_run(a)
    planned=[s['session'] for s in m['future_sessions']]
    if set(a.sessions)!=set(planned) or len(a.sessions)!=len(planned):
        raise ValueError('Evaluate both distinct prospectively registered sessions together.')
    # Keep both planned collections complete before opening outcomes.
    for name in planned:
        receipt=run/('collection_'+name+'.json')
        directory=a.project_root/'data_3tx/raw'/name
        if not receipt.exists():
            raise ValueError(f'No prospective collection receipt for {name}. Use this toolkit collect command.')
        r=json.loads(receipt.read_text())
        if r['started_unix']<m['frozen_unix'] or r['manifest_sha256']!=sha(run/'manifest.json'):
            raise ValueError('Collection was not committed after this model freeze.')
        if len(list(directory.glob('*_trial*.csv')))!=50:
            raise ValueError(f'Complete both planned sessions before viewing results: {name} is incomplete.')
    output=run/'evaluations'
    output.mkdir(exist_ok=True)
    prepared=[]
    hashes=set(m['training_data_sha256'].values())
    trial_hashes={h for name in m['training_sessions'] for h in
                  json.loads((run/'training_audit'/name/'audit.json').read_text())['individual_trial_sha256'].values()}
    for name in a.sessions:
        target=output/name
        if target.exists():
            raise FileExistsError(f'{target} exists; original outputs are never overwritten.')
        temp=run/'audit_attempts'/(name+'_'+str(time.time_ns()))
        data,audit=read_features(core,a.project_root/'data_3tx/raw'/name,temp,m['feature_config'])
        if data.zone_coords!={k:tuple(v) for k,v in m['zone_coords'].items()}:
            raise ValueError('Test grid differs from frozen grid.')
        if audit['raw_csv_sha256'] in hashes:
            raise ValueError('Session duplicates training or another test session.')
        current=set(audit['individual_trial_sha256'].values())
        if current & trial_hashes:
            raise ValueError('At least one raw trial duplicates training or another test trial.')
        hashes.add(audit['raw_csv_sha256']);trial_hashes.update(current)
        prepared.append((name,target,data,audit))
    # Data validation finishes before any scoring starts.
    old=legacy_bundle(run/m['historical']['file'])
    for name,target,data,audit in prepared:
        target.mkdir()
        rows,frames=[],[]
        for model_id,spec in SPECS.items():
            model=load_trusted(run/m['artifacts'][model_id]['file'])
            row,pred,cm=score(core,model,data,spec['links'],spec['view'],model_id,'matched_seed_six')
            rows.append(row);frames.append(pred)
            pd.DataFrame(cm,index=CLASS_ORDER,columns=CLASS_ORDER).to_csv(target/(model_id+'_confusion.csv'))
        metrics,pred=historical_scores(core,old,data)
        rows.extend(metrics.to_dict('records'));frames.append(pred)
        pd.DataFrame(rows).to_csv(target/'metrics.csv',index=False)
        pd.concat(frames,ignore_index=True).to_csv(target/'trial_predictions.csv',index=False)
        write_json(target/'evaluation.json',dict(session=name,evaluated_utc=utc(),fitting_performed=False,
                   model_manifest_sha256=sha(run/'manifest.json'),raw_csv_sha256=audit['raw_csv_sha256'],
                   individual_trial_sha256=audit['individual_trial_sha256'],environment=environment(),
                   result_sha256={f:sha(target/f) for f in ['metrics.csv','trial_predictions.csv']}))
        print(name,pd.DataFrame(rows)[['model','trial_accuracy','occupied_trial_accuracy','empty_recall']].to_string(index=False))
    print('EVALUATION PASS. Six matched-seed models and two historical models; no fitting performed.')


def report(a):
    from study_analysis import make_report
    run,m,core=frozen_run(a)
    frames=[];metrics=[];seen=set()
    for s in m['future_sessions']:
        folder=run/'evaluations'/s['session']
        meta=json.loads((folder/'evaluation.json').read_text())
        if meta['model_manifest_sha256']!=sha(run/'manifest.json') or meta['raw_csv_sha256'] in seen:
            raise ValueError('Mismatched model provenance or duplicate session data.')
        for name,digest in meta['result_sha256'].items():
            if sha(folder/name)!=digest:
                raise ValueError('Saved prediction or metric CSV changed after evaluation.')
        seen.add(meta['raw_csv_sha256'])
        frames.append(pd.read_csv(folder/'trial_predictions.csv'))
        metrics.append(pd.read_csv(folder/'metrics.csv'))
    out=make_report(pd.concat(frames,ignore_index=True),pd.concat(metrics,ignore_index=True),
                    a.output if a.output else run/'report',iterations=5000)
    print('REPORT PASS:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('inspect');s.add_argument('--output',type=Path,default=Path('results/six_model_inspection_20261030'))
    s=sub.add_parser('reproduce');s.add_argument('--output',type=Path,default=Path('results/s3_historical_reproduction_20261030'))
    s=sub.add_parser('freeze');s.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    s.add_argument('--future-sessions',nargs=2,default=['3TX_S4','3TX_S5'])
    s=sub.add_parser('collect');s.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    s.add_argument('--session',required=True);s.add_argument('--port');s.add_argument('--resume',action='store_true')
    s=sub.add_parser('evaluate');s.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    s.add_argument('--sessions',nargs='+',default=['3TX_S4','3TX_S5'])
    s=sub.add_parser('report');s.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    s.add_argument('--output',type=Path)
    a=p.parse_args()
    a.project_root=a.project_root.resolve()
    if not a.project_root.is_dir():
        p.error('Project directory does not exist.')
    try:
        globals()[a.command](a)
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,KeyError) as e:
        raise SystemExit('STOP: '+str(e)) from e


if __name__=='__main__':
    main()
