"""Optional synthetic integration checks. NO scientific results are generated.

Uses the inspected original code, a synthetic historical bundle, and a mocked
serial-collection subprocess. Hardware/actual historical predictions are NOT
validated by these fixtures. Do not include fixture scores in the manuscript.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from study_core import ALL, CLASS_ORDER, HERE, SPECS, audit_session, import_path, matrix, original, sha


def make_session(root,name,seed):
    c=import_path('_collector_fixture',root/'collect_three_link_trials.py')
    directory=root/'data_3tx/raw'/name;directory.mkdir(parents=True)
    rng=np.random.default_rng(seed)
    for z_index,zone in enumerate(CLASS_ORDER):
        for trial in range(1,6):
            facing='NA' if zone=='EMPTY' else ['N','E','S','W','N'][trial-1]
            records=[]
            for link_index,(mac,link) in enumerate(c.MAC_TO_LINK.items()):
                for packet in range(66):
                    row={k:0 for k in c.CSV_HEADER}
                    row.update(host_time_s=1000+packet*.03+link_index*.001,session=name,
                               zone=zone,trial=trial,orientation=facing,link=link,mac=mac,
                               role='AP',rssi=-35-z_index*2-link_index*10+rng.normal(0,.3),
                               local_timestamp=packet*30000,len=[128,256,384][packet%3])
                    if zone!='EMPTY':row['x_m'],row['y_m']=c.ZONE_CENTERS[zone]
                    for i in range(128):row[f'iq_{i:03d}']=int(rng.integers(-25,26)+z_index)
                    records.append(row)
            pd.DataFrame(records,columns=c.CSV_HEADER).to_csv(
                directory/f'{zone}_trial{trial:02d}_{facing}.csv',index=False)
    (directory/'session_manifest.json').write_text(json.dumps(
        dict(session=name,mac_to_link=c.MAC_TO_LINK,zone_centers_m=c.ZONE_CENTERS)))
    return directory


def run_cli(root,*args,success=True):
    command=[sys.executable,str(HERE/'tricsi_day.py'),'--project-root',str(root),*map(str,args)]
    p=subprocess.run(command,capture_output=True,text=True)
    if (p.returncode==0)!=success:
        raise AssertionError('Unexpected CLI outcome: '+str(args)+'\n'+p.stdout+'\n'+p.stderr)
    return p


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path)
    a=p.parse_args()
    temp=tempfile.TemporaryDirectory(prefix='TriCSI fixture ') if a.work_dir is None else None
    root=Path(temp.name) if temp else a.work_dir.resolve()
    if not temp:root.mkdir(parents=True,exist_ok=False)
    bundle=root/'tricsi_3tx_modeling_bundle';bundle.mkdir()
    for name in ['tricsi_3tx_modeling.py','compare_and_freeze_3tx.py','evaluate_frozen_3tx.py']:
        shutil.copyfile(HERE/'reference_sources'/name,bundle/name)
    shutil.copyfile(HERE/'reference_sources/collect_three_link_trials.py',root/'collect_three_link_trials.py')
    core=original(root)
    for name,seed in [('3TX_S1',1),('3TX_S2',2),('3TX_S3',3)]:make_session(root,name,seed)
    d=core.extract_session_features(root/'data_3tx/raw/3TX_S1')
    assert np.array_equal(matrix(d,ALL,'full')[0],core.combine_links(d,ALL))
    assert matrix(d,ALL,'csi')[0].shape==(350,477)
    assert matrix(d,ALL,'rssi')[0].shape==(350,9)
    models={}
    for i,(name,links) in enumerate(core.LINK_SUBSETS.items()):
        m=core.make_classifier(seed=2026+i,n_estimators=8)
        m.fit(core.combine_links(d,links),d.metadata.zone);models[name]=m
    old=dict(format_version=2,models_by_subset=models,subset_links={k:list(v) for k,v in core.LINK_SUBSETS.items()},
             primary_subset='3tx_all',feature_config={'window_packets':64,'windows_per_trial':7},
             classes=CLASS_ORDER,zone_coords=d.zone_coords,training_sessions=['3TX_S1','3TX_S2'],
             training_data_sha256={s:core.session_data_sha256(root/'data_3tx/raw'/s) for s in ['3TX_S1','3TX_S2']},
             seed=2026,trees=8)
    model_path=root/'results/3tx_dev_comparison/frozen_3tx_model.joblib';model_path.parent.mkdir(parents=True)
    joblib.dump(old,model_path)
    before=sha(model_path)
    # Audit the exact collector's 160-column schema, valid declared lengths,
    # metadata-only damage, and a row excluded by the original feature code.
    f=root/'data_3tx/raw/3TX_S3/Z9_trial05_N.csv'
    frame=pd.read_csv(f)
    frame['len']=frame['len'].astype(object)
    frame.loc[0,'len']='384 EXTRA CSI TEXT [1 2 3]'
    frame['iq_000']=frame['iq_000'].astype(object)
    frame.loc[1,'iq_000']='NOT_AN_IQ_VALUE'
    frame.to_csv(f,index=False)
    audit=audit_session(core,f.parent,root/'metadata_audit')
    assert audit['issue_counts']['unused_metadata_warning']==1
    assert audit['rows_excluded_by_original_extractor']==1
    features=core.extract_session_features(f.parent)
    assert features.metadata.packets_back_left.min()==65
    print('PASS: original feature parity, dimensions, 160-column CSV, len metadata, required-row filtering')
    run_cli(root,'inspect','--output',root/'inspect')
    # The synthetic bundle is intentionally NOT engineered to attain 70/84.
    # Historical mismatch must stop, keep outputs, and never retune.
    run_cli(root,'reproduce','--output',root/'history_mismatch',success=False)
    assert (root/'history_mismatch/reproduction.json').is_file()
    assert not json.loads((root/'history_mismatch/reproduction.json').read_text())['historical_accuracy_matches']
    run=root/'study'
    run_cli(root,'freeze','--run-dir',run)
    assert sha(model_path)==before
    m=json.loads((run/'manifest.json').read_text())
    assert all(x['seed']==2026 for x in m['artifacts'].values())
    # A fixture replaces ONLY the hardware subprocess. All collection receipts,
    # commands, unchanged original-source checks, and audits execute normally.
    day=import_path('_tricsi_day_fixture',HERE/'tricsi_day.py')
    def fake_hardware(command,**kwargs):
        name=command[command.index('--session')+1]
        seed=int(command[command.index('--seed')+1])
        make_session(root,name,seed)
        return SimpleNamespace(returncode=0)
    with patch.object(day.subprocess,'run',side_effect=fake_hardware):
        for name in ['3TX_S4','3TX_S5']:
            day.collect(SimpleNamespace(project_root=root,run_dir=run,session=name,port=None,resume=False))
    run_cli(root,'evaluate','--run-dir',run)
    for name in ['3TX_S4','3TX_S5']:
        pred=pd.read_csv(run/'evaluations'/name/'trial_predictions.csv')
        assert len(pred)==400 and set(pred.model)==set(SPECS)|{'H1','H2'}
    run_cli(root,'report','--run-dir',run)
    assert len(pd.read_csv(run/'report/paired_contrasts.csv'))==21
    assert len(pd.read_csv(run/'report/all_trial_predictions.csv'))==800
    run_cli(root,'evaluate','--run-dir',run,success=False)  # no result overwrite
    run_cli(root,'freeze','--run-dir',root/'cannot_redeclare_old_data',success=False)
    saved=run/'models/M1.joblib';data=saved.read_bytes();saved.write_bytes(data+b'altered')
    run_cli(root,'report','--run-dir',run,success=False)
    saved.write_bytes(data)
    assert sha(model_path)==before
    print('PASS: inspect, historical mismatch diagnostic, six-model freezing, mocked collection, eight-model inference, stats, figures')
    print('PASS: result overwrite, prospective-data reuse, and model-tamper guards')
    print('SYNTHETIC SOFTWARE CHECKS ONLY. Actual historical accuracy and serial hardware were not tested.')
    if temp:temp.cleanup()
    else:print('Synthetic checks retained at',root)


if __name__=='__main__':main()
