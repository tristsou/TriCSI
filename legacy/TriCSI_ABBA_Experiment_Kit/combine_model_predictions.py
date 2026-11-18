#!/usr/bin/env python3
"""Join the user's existing frozen 2-link and 3-link prediction exports by exact trial key.

Required columns in each input: session,trial_id,true_label,predicted_label.
Never trains/invokes a classifier, only merges actual predictions.
"""
import argparse
from pathlib import Path
import pandas as pd

NEEDED=['session','trial_id','true_label','predicted_label']

def read(path,suffix):
    d=pd.read_csv(path,dtype=str,keep_default_na=False)
    missing=[x for x in NEEDED if x not in d.columns]
    if missing: raise ValueError(f'{path}: missing required columns {missing}')
    d=d[NEEDED].copy()
    for col in NEEDED: d[col]=d[col].astype(str).str.strip()
    if d[NEEDED].eq('').any().any(): raise ValueError(f'{path}: blank labels/fields')
    if d.duplicated(['session','trial_id']).any(): raise ValueError(f'{path}: duplicate session/trial_id')
    return d.rename(columns={'true_label':f'truth_{suffix}',
                              'predicted_label':f'pred_{suffix}'})

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--two',required=True,help='CSV from frozen outer-pair evaluator')
    pa.add_argument('--three',required=True,help='CSV from frozen all-3 evaluator')
    pa.add_argument('--template',required=True,help='prediction_template.csv supplied in this kit')
    pa.add_argument('--output',default='filled_predictions.csv')
    a=pa.parse_args()
    two=read(a.two,'2link'); three=read(a.three,'3link')
    if len(two)!=200 or len(three)!=200: raise ValueError('Both exports must have 200 trial rows')
    m=two.merge(three,on=['session','trial_id'],validate='one_to_one',how='outer',indicator=True)
    if not m['_merge'].eq('both').all():
        raise ValueError('The two models did not evaluate the exact same 200 trials')
    template=pd.read_csv(a.template,dtype=str,keep_default_na=False)
    if len(template)!=200: raise ValueError('Template must have 200 rows')
    m=template[['session','trial_id','true_label']].merge(m,on=['session','trial_id'],how='outer',validate='one_to_one',indicator='template_match')
    if not m.template_match.eq('both').all():
        raise ValueError('Model exports contain different trial keys from the 200-row template')
    for col in ['truth_2link','truth_3link']:
        if not (m.true_label.str.upper() == m[col].str.upper()).all():
            bad=m[m.true_label.str.upper()!=m[col].str.upper()]
            raise ValueError(f'Ground truth conflict between template and {col}:\n{bad.head().to_string(index=False)}')
    combined=template.merge(m[['session','trial_id','pred_2link','pred_3link']],on=['session','trial_id'],how='left',validate='one_to_one',suffixes=('_old',''))
    for col in ['pred_2link','pred_3link']:
        if col not in combined: raise ValueError(f'Internal merge error missing {col}')
    output=Path(a.output)
    combined[[*('session','condition','trial_id','true_label','orientation'), 'pred_2link','pred_3link']].to_csv(output,index=False)
    print(f'Combined all 200 paired trials into: {output}')

if __name__=='__main__': main()
