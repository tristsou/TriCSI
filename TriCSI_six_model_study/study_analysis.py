"""Paired, session-conditional analysis and deterministic Matplotlib figures.

AI-assisted code. Intervals are unadjusted descriptive trial bootstrap intervals;
they do not estimate future-day variability or independence of participants.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study_core import CLASS_ORDER, SPECS, fresh, write_json

CONTRASTS = {
    'historical_full_link_gain': {'H2':1,'H1':-1},
    'matched_full_link_gain': {'M2':1,'M1':-1},
    'csi_gain_link_gain': {'M4':1,'M3':-1},
    'rssi_link_gain': {'M6':1,'M5':-1},
    'csi_gain_minus_rssi_link_gain': {'M4':1,'M3':-1,'M6':-1,'M5':1},
    'three_full_minus_rssi': {'M2':1,'M6':-1},
    'outer_full_minus_rssi': {'M1':1,'M5':-1},
}


def table_check(predictions):
    required = {'session','trial_uid','true_zone','predicted_zone','model'}
    if not required.issubset(predictions):
        raise ValueError(f'Missing prediction fields: {required-set(predictions)}')
    if predictions.duplicated(['session','model','trial_uid']).any():
        raise ValueError('Duplicate trial/model rows.')
    identities = predictions.groupby(['session','trial_uid']).true_zone.nunique()
    if not (identities == 1).all():
        raise ValueError('Truth labels differ across model predictions.')
    for session, part in predictions.groupby('session'):
        models = set(part.model)
        if models != set(SPECS)|{'H1','H2'}:
            raise ValueError(f'{session}: expected six matched-seed plus two historical models.')
        ids = None
        for model, frame in part.groupby('model'):
            counts = frame.true_zone.value_counts()
            if set(counts.index) != set(CLASS_ORDER) or not (counts == 5).all():
                raise ValueError(f'{session}/{model}: expected five trials per class.')
            current = set(frame.trial_uid)
            if ids is not None and current != ids:
                raise ValueError('Models do not predict identical trials.')
            ids = current


def bootstrap_rows(frame, iterations=5000, seed=2026):
    """Resample whole trials within session and class, preserving ALL model pairs."""
    frame = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pieces = []
    for _, group in frame.groupby(['session','true_zone'], sort=True):
        ids = group.index.to_numpy()
        pieces.append(rng.choice(ids, size=(iterations,len(ids)), replace=True))
    return np.concatenate(pieces, axis=1)


def contrasts(predictions, iterations=5000):
    p = predictions.copy()
    p['correct'] = (p.true_zone == p.predicted_zone).astype(int)
    records, pairs = [], []
    scopes = [(name,group) for name,group in p.groupby('session',sort=True)]
    if len(scopes)>1:
        scopes.append(('pooled_observed_sessions_only',p))
    for scope, frame in scopes:
        wide = frame.pivot(index=['session','trial_uid','true_zone'], columns='model',values='correct').reset_index()
        indices = bootstrap_rows(wide,iterations)
        for name, weights in CONTRASTS.items():
            value = sum(weight*wide[model].to_numpy(dtype=float) for model,weight in weights.items())
            sims = value[indices].mean(axis=1)
            low,high = np.quantile(sims,[.025,.975])
            rec = dict(scope=scope,contrast=name,n_trials=len(wide),estimate_pp=100*value.mean(),
                       ci95_low_pp=100*low,ci95_high_pp=100*high,
                       interval_scope='within observed session/class trial strata; unadjusted')
            plus = [k for k,v in weights.items() if v==1]
            minus = [k for k,v in weights.items() if v==-1]
            if len(plus)==len(minus)==1:
                good,bad = wide[plus[0]].to_numpy(),wide[minus[0]].to_numpy()
                rec.update(improved_trials=int(((good==1)&(bad==0)).sum()),
                           harmed_trials=int(((good==0)&(bad==1)).sum()),
                           both_correct=int(((good==1)&(bad==1)).sum()),
                           both_wrong=int(((good==0)&(bad==0)).sum()))
            records.append(rec)
            if scope!='pooled_observed_sessions_only':
                temp=wide[['session','trial_uid','true_zone']].copy()
                temp['contrast']=name
                temp['trial_effect']=value
                pairs.append(temp)
    return pd.DataFrame(records),pd.concat(pairs,ignore_index=True)


def zone_score_contrasts(predictions,iterations=5000):
    p=predictions[predictions.true_zone!='EMPTY'].copy()
    rows=[]
    scopes=list(p.groupby('session',sort=True))
    if len(scopes)>1:scopes.append(('pooled_observed_sessions_only',p))
    for scope,frame in scopes:
        wide=frame.pivot(index=['session','trial_uid','true_zone'],columns='model',values='position_error_m').reset_index()
        idx=bootstrap_rows(wide,iterations,seed=2027)
        for name,before,after in [('historical_full','H1','H2'),('matched_full','M1','M2'),
                                  ('csi_gain','M3','M4'),('rssi','M5','M6')]:
            reduction=(wide[before]-wide[after]).to_numpy()
            lo,hi=np.quantile(reduction[idx].mean(axis=1),[.025,.975])
            rows.append(dict(scope=scope,feature_comparison=name,n_occupied_trials=len(wide),
                             zone_score_reduction_m=float(reduction.mean()),ci95_low_m=lo,ci95_high_m=hi,
                             note='Includes fixed occupied-to-EMPTY penalty; not continuous tracking accuracy.'))
    return pd.DataFrame(rows)


def export(fig,path):
    fig.text(.01,.008,'Created by the student researcher using ChatGPT-assisted Python and Matplotlib code, 2026.',fontsize=7)
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ('png','pdf','svg'):
        fig.savefig(path.with_suffix('.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)


def figures(pred,metrics,effects,out):
    sessions=sorted(pred.session.unique())
    fig,axes=plt.subplots(1,len(sessions),figsize=(6*len(sessions),4.5),squeeze=False)
    for ax,session in zip(axes[0],sessions):
        m=metrics[(metrics.session==session)&metrics.model.isin(SPECS)].set_index('model')
        x=np.arange(3)
        for offset,ids,label,color in [(-.18,['M1','M3','M5'],'Outer pair','#315c8a'),
                                       (.18,['M2','M4','M6'],'All three','#d47a24')]:
            values=100*m.loc[ids,'trial_accuracy'].to_numpy()
            ax.bar(x+offset,values,.36,label=label,color=color)
            for xi,y in zip(x+offset,values):ax.text(xi,y+1,f'{y:.0f}',ha='center',fontsize=9)
        ax.set_xticks(x,['CSI + gain\n+ RSSI','CSI + gain\nno RSSI','RSSI only'])
        ax.set(ylim=(0,106),ylabel='Exact whole-trial accuracy (%)',title=session+' / matched seed')
        ax.legend(fontsize=9,loc='lower right')
    export(fig,out/'six_model_accuracy')
    names=['historical_full_link_gain','matched_full_link_gain','csi_gain_link_gain','rssi_link_gain']
    labels=['Historical full features','Matched-seed full features','CSI + gain, no RSSI','RSSI only']
    fig,axes=plt.subplots(1,len(sessions),figsize=(6*len(sessions),4.3),squeeze=False)
    for ax,session in zip(axes[0],sessions):
        frame=effects[(effects.scope==session)&effects.contrast.isin(names)].set_index('contrast').loc[names]
        y=np.arange(len(names))
        ax.hlines(y,frame.ci95_low_pp,frame.ci95_high_pp,color='#315c8a',lw=2)
        ax.plot(frame.estimate_pp,y,'o',color='#315c8a')
        ax.axvline(0,color='gray',ls='--',lw=1)
        ax.set_yticks(y,labels);ax.invert_yaxis()
        ax.set(xlabel='Three-link minus outer-pair accuracy (percentage points)',title=session)
    export(fig,out/'paired_link_gains')
    fig,axes=plt.subplots(1,len(sessions),figsize=(6*len(sessions),4.3),squeeze=False)
    for ax,session in zip(axes[0],sessions):
        frame=effects[(effects.scope==session)&effects.contrast.isin(names)].set_index('contrast').loc[names]
        y=np.arange(len(names))
        ax.barh(y,frame.improved_trials,color='#43845b',label='Corrected trials')
        ax.barh(y,-frame.harmed_trials,color='#ae4949',label='New errors')
        ax.axvline(0,color='gray',lw=1);ax.set_yticks(y,labels);ax.invert_yaxis()
        ax.set(xlabel='Number of matched trials',title=session);ax.legend(fontsize=8)
    export(fig,out/'corrected_and_harmed_trials')
    for session,frame in pred[pred.model.isin(SPECS)].groupby('session'):
        fig,axes=plt.subplots(2,3,figsize=(15,10))
        for ax,model in zip(axes.flat,SPECS):
            f=frame[frame.model==model]
            cm=pd.crosstab(f.true_zone,f.predicted_zone).reindex(index=CLASS_ORDER,columns=CLASS_ORDER,fill_value=0)
            ax.imshow(cm,cmap='Blues',vmin=0,vmax=5)
            ax.set_xticks(range(10),CLASS_ORDER,rotation=60,fontsize=8)
            ax.set_yticks(range(10),CLASS_ORDER,fontsize=8)
            ax.set(xlabel='Predicted',ylabel='Actual',title=model+' / '+SPECS[model]['label'])
            for i in range(10):
                for j in range(10):
                    if cm.iloc[i,j]:ax.text(j,i,str(cm.iloc[i,j]),ha='center',va='center',fontsize=8)
        fig.suptitle(session+' / five trials per actual class')
        export(fig,out/(session+'_confusion_matrices'))


def make_report(predictions,metrics,out,iterations=5000):
    out=fresh(out)
    table_check(predictions)
    effects,pairs=contrasts(predictions,iterations)
    metrics.to_csv(out/'metrics_by_session.csv',index=False)
    effects.to_csv(out/'paired_contrasts.csv',index=False)
    zone_score_contrasts(predictions,iterations).to_csv(out/'paired_zone_error_score_reductions.csv',index=False)
    pairs.to_csv(out/'paired_trial_effects.csv',index=False)
    predictions.to_csv(out/'all_trial_predictions.csv',index=False)
    summary=metrics.groupby(['lane','model']).agg(
        sessions=('session','nunique'),mean_session_accuracy=('trial_accuracy','mean'),
        minimum_session_accuracy=('trial_accuracy','min'),maximum_session_accuracy=('trial_accuracy','max'),
        descriptive_session_sd=('trial_accuracy','std')).reset_index()
    summary.to_csv(out/'descriptive_session_summary.csv',index=False)
    feature_effect=effects[effects.contrast=='csi_gain_minus_rssi_link_gain']
    feature_effect.to_csv(out/'feature_link_interaction.csv',index=False)
    by_class=predictions.copy()
    by_class['correct']=(by_class.true_zone==by_class.predicted_zone).astype(int)
    by_class.groupby(['session','model','true_zone']).correct.agg(['count','sum','mean']).to_csv(out/'per_class_results.csv')
    by_class.groupby(['session','model','orientation']).correct.agg(['count','sum','mean']).to_csv(out/'orientation_diagnostics.csv')
    figures(predictions,metrics,effects,out)
    write_json(out/'interpretation_limits.json',{
        'session_count':int(predictions.session.nunique()),
        'bootstrap_iterations':iterations,
        'scope':'Same participant, same room, short-term recording-session replication.',
        'uncertainty':'Percentile paired trial bootstrap within session and class; unadjusted marginal intervals.',
        'pooled_interval':'Conditional on observed sessions, not a future-session confidence interval.',
        'mechanism':'CSI view includes gain; RSSI/CSI comparisons do not establish an isolated spectral or physical cause.',
        'novelty':'These experiments do not constitute reproduction or outperformance of a published adaptation method.',
    })
    return out
