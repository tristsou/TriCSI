"""Summarize per-trial paired two vs three-link effects, with honest session-level reporting."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd

CONFIGS=['full','csi','rssi']
def report(input_csv,outdir):
    df=pd.read_csv(input_csv,dtype={'session':str,'trial_id':str,'actual':str,'predicted':str,'subset':str,'features':str})
    required={'session','trial_id','actual','predicted','subset','features','seed','correct'}
    if not required.issubset(df):raise ValueError(f'Missing columns {required-set(df)}')
    if df.duplicated(['session','trial_id','subset','features','seed']).any():raise ValueError('Duplicate predictions')
    if not (df.correct.isin([0,1]).all()):raise ValueError('correct must be 0/1')
    ses=sorted(df.session.unique());seeds=sorted(df.seed.unique())
    n_tr={s:df[df.session==s].trial_id.nunique() for s in ses}
    if any(n!=50 for n in n_tr.values()):raise ValueError(f'Expected 50 physical trials per session: {n_tr}')
    if set(df.features)!=set(CONFIGS) or set(df.subset)!={'2tx_outer','3tx_all'}:
        raise ValueError('Expected full/csi/rssi and both link subsets')
    expected=sum(n_tr.values())*len(seeds)*2*3
    if len(df)!=expected:raise ValueError(f'Expected {expected} prediction rows; found {len(df)}')
    # Pair by physical trial + seed; fail instead of dropping missing values.
    pair=df.pivot(index=['session','trial_id','actual','features','seed'],columns='subset',values='correct')
    if pair.isnull().any().any():raise ValueError('Some paired trial predictions missing')
    pair=pair.reset_index()
    pair['delta']=(pair['3tx_all']-pair['2tx_outer']).astype(int)
    pair['corrected']=(pair.delta==1).astype(int)
    pair['harmed']=(pair.delta==-1).astype(int)
    stat=pair.groupby(['session','features','seed'],as_index=False).agg(
          accuracy_two=('2tx_outer','mean'),accuracy_three=('3tx_all','mean'),
          gain=('delta','mean'),corrected=('corrected','sum'),harmed=('harmed','sum'),n=('delta','size'))
    overall=stat.groupby(['session','features'],as_index=False).agg(
          accuracy_two=('accuracy_two','mean'),accuracy_three=('accuracy_three','mean'),
          gain=('gain','mean'), corrected=('corrected','mean'),harmed=('harmed','mean'))
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    stat.to_csv(out/'per_seed_per_session.csv',index=False)
    overall.to_csv(out/'per_session.csv',index=False)
    pair.to_csv(out/'paired_trial_results.csv',index=False)
    # Feature effect = (3-2 RSSI) - (3-2 CSI), by session.
    diff=overall.pivot(index='session',columns='features',values='gain')
    diff['RSSI_minus_CSI_additional_link_gain'] = diff['rssi']-diff['csi']
    diff.to_csv(out/'feature_mechanism_contrast.csv')
    # Bootstrap resamples paired trials (within class) for each session, seed2026 only.
    # Not a confidence interval for independent days, rooms, or people.
    rng=np.random.default_rng(2026); cis=[]
    for (s,f),part in pair[pair.seed==2026].groupby(['session','features']):
        values=[]
        by_label=[g.delta.to_numpy() for _,g in part.groupby('actual')]
        for _ in range(2000):
            values.append(np.mean(np.concatenate([rng.choice(v,len(v),replace=True) for v in by_label])))
        lo,hi=np.percentile(values,[2.5,97.5]);cis.append({'session':s,'features':f,'mean_gain':part.delta.mean(),
           'bootstrap_95_low':lo,'bootstrap_95_high':hi,'interpretation':'conditional on recorded same-session trials'})
    pd.DataFrame(cis).to_csv(out/'paired_trial_bootstrap.csv',index=False)
    s3=stat[(stat.session=='3TX_S3')&(stat.features=='full')&(stat.seed==2026)]
    checks=[]
    if len(s3)==1:
        checks.append('Historical S3 reproduction check (newly fitted models, seed=2026): '
            f'outer={100*s3.iloc[0].accuracy_two:.1f}%, all={100*s3.iloc[0].accuracy_three:.1f}%; '
            'original paper values were 70.0% and 84.0%. Differences mean this reconstructed pipeline is not equivalent.')
    newest=overall[overall.session.isin(['3TX_S4','3TX_S5'])]
    if len(newest)>0:
        newest[ ['session','features','accuracy_two','accuracy_three','gain'] ].to_csv(out/'new_heldout_only.csv',index=False)
    fig,axes=plt.subplots(figsize=(9,5))
    full=overall[overall.features=='full'].sort_values('session')
    x=np.arange(len(full));ww=0.34
    axes.bar(x-ww/2,100*full.accuracy_two,width=ww,label='2 links')
    axes.bar(x+ww/2,100*full.accuracy_three,width=ww,label='3 links')
    axes.set_xticks(x,full.session);axes.set_ylabel('Exact whole-trial accuracy (%)')
    axes.set_ylim(0,100);axes.set_title('Full-feature model: same-trial link comparison')
    axes.legend();fig.tight_layout();fig.savefig(out/'fig1_full_accuracy.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for key in CONFIGS:
        d=overall[overall.features==key].sort_values('session')
        ax.plot(d.session,d.gain*100,marker='o',label=key.upper())
    ax.axhline(0,linestyle='--',linewidth=1);ax.set_ylabel('Three-link gain (percentage points)')
    ax.set_title('Does the extra link help via CSI or RSSI?');ax.legend()
    fig.tight_layout();fig.savefig(out/'fig2_feature_link_gains.png',dpi=180);plt.close(fig)
    lines=['# TriCSI post-review analysis (new models)','',
      'All accuracy scores are **whole-trial** results. Multiple random seeds do not create new sessions.','',
      *checks,'', '## Per-session results','',overall.to_string(index=False,float_format=lambda x:f'{x:.3f}'),'',
      '## CSI-versus-RSSI extra-link contrast','',diff.to_string(float_format=lambda x:f'{x:.3f}'),'',
      'Paired bootstrap intervals treat recorded trials as resampling units. They do NOT measure generalization to new rooms/days/people.','',
      'S3 has already been analyzed historically. Newly reconstructed comparison variants are post-review analyses. Do not count S3 as a fresh prospective validation.','',
      'New S4/S5 scores are prospective only if models and feature choices were fixed before their labels were analyzed.',
      'This study does not measure continuous tracking, gait, fall detection, or positional centimeter accuracy.']
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(checks));print(overall.to_string(index=False));print('Saved results:',out)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--predictions',required=True);a.add_argument('--out',required=True)
    v=a.parse_args();report(v.predictions,v.out)
