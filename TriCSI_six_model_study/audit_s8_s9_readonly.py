"""Optional AI-assisted read-only acquisition inventory, not a new neural evaluator.

No PyTorch required. Does not retrain, change prototypes/thresholds, replace S9,
or certify that current files are the exact files used historically.
"""
import argparse
import ast
import csv
import json
import math
from collections import Counter
from pathlib import Path

from study_core import fresh, sha, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,default=Path('results/s8_s9_readonly_inventory_20261030'))
    a=p.parse_args();root=a.project_root.resolve();out=fresh(a.output)
    config=root/'project_config.py'
    symbols={}
    for node in ast.parse(config.read_text(encoding='utf-8-sig')).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('LEFT_MAC','RIGHT_MAC'):
                symbols[node.targets[0].id]=ast.literal_eval(node.value).upper()
    if len(symbols)!=2:raise ValueError('Could not establish MAC mapping from actual project_config.py.')
    expected={symbols['LEFT_MAC']:'left',symbols['RIGHT_MAC']:'right'}
    files=[];issues=[]
    for session in ['S8','S9']:
        directory=root/'data/raw'/session
        paths=sorted(directory.glob('*_trial*.csv'))
        if not paths:raise FileNotFoundError(directory)
        identities=[]
        for path in paths:
            counts=Counter();invalid=0;first=None;previous={};timing=0;rows=0
            with path.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f)
                required={'mac','rssi','zone','trial',*[f'iq_{i:03d}' for i in range(128)]}
                if not required.issubset(reader.fieldnames or []):
                    issues.append(dict(file=path.name,session=session,issue='Missing required saved columns'))
                    continue
                for row in reader:
                    if first is None:first=row.copy()
                    rows+=1
                    mac=row['mac'].strip().upper();counts[mac]+=1
                    if mac not in expected:
                        issues.append(dict(file=path.name,session=session,issue='Unknown MAC '+mac))
                    elif row.get('link',expected[mac]).lower()!=expected[mac]:
                        issues.append(dict(file=path.name,session=session,issue='MAC/link label mismatch'))
                    try:
                        if not all(math.isfinite(float(row[k])) for k in ['rssi',*[f'iq_{i:03d}' for i in range(128)]]):
                            invalid+=1
                    except (ValueError,TypeError):invalid+=1
                    if 'host_time_s' in row:
                        try:
                            t=float(row['host_time_s'])
                            if t<=previous.get(mac,-float('inf')):timing+=1
                            previous[mac]=t
                        except (ValueError,TypeError):timing+=1
            if first is None:
                issues.append(dict(file=path.name,session=session,issue='Empty trial file'));continue
            zone=first['zone'];trial=int(first['trial']);identities.append((zone,trial))
            scored=(trial>=2) if zone=='EMPTY' else (trial>=3 if session=='S8' else True)
            files.append(dict(session=session,file=path.name,zone=zone,trial=trial,
                              scored_by_inspected_original_protocol=scored,
                              background_calibration=(zone=='EMPTY' and trial==1),rows=rows,
                              invalid_iq_or_rssi_rows=invalid,nonincreasing_host_times=timing,
                              packets_by_mac=dict(counts),sha256=sha(path)))
        duplicates=[str(k) for k,v in Counter(identities).items() if v>1]
        if duplicates:issues.append(dict(session=session,issue='Duplicate zone/trial identities '+str(duplicates)))
        if ('EMPTY',1) not in identities:issues.append(dict(session=session,issue='Missing EMPTY trial 1'))
    checkpoint=root/'models/tricsi_anchor_s4_s5_s6_s7.pt'
    sources={}
    for s in ['S4','S5','S6','S7']:
        sources[s]={f.name:sha(f) for f in sorted((root/'data/raw'/s).glob('*_trial*.csv'))}
    code={}
    for name in ['project_config.py','csi_io.py','train_csi_cnn.py',
                 'train_tricsi_anchor_final.py','evaluate_frozen_anchor_confirmatory.py']:
        f=root/name;code[name]=sha(f) if f.exists() else None
    write_json(out/'inventory.json',dict(
        mapping=expected,files=files,issues=issues,current_source_session_hashes=sources,
        checkpoint_exists=checkpoint.is_file(),checkpoint_sha256=sha(checkpoint) if checkpoint.is_file() else None,
        source_code_sha256=code,
        inspected_protocol={'S8':'Occupied trials >=3; EMPTY >=2. Only EMPTY trial 1 calibrates target.',
                            'S9':'All occupied trials; EMPTY >=2. Only EMPTY trial 1 calibrates target.',
                            'prototypes':'Original evaluator recomputes prototypes from S4-S7 using frozen encoder.',
                            'occupancy_threshold':0.5},
        limitations=['Inventory only; no PyTorch inference or prototype/normalization equivalence test.',
                     'Compare current checkpoint/source hashes with contemporaneous records before asserting historical identity.',
                     'Different trial-selection rules are documented design differences, not automatically an error.',
                     'No low-accuracy session excluded or altered.']))
    print('Read-only inventory saved to',out)
    print('This does not establish a cause for S9 or justify removing it.')


if __name__=='__main__':main()
