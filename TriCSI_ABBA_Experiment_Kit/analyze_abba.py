#!/usr/bin/env python3
"""Analyze matched frozen-model predictions for the TriCSI A-B-B-A study.

Run: python analyze_abba.py --input prediction_template.csv --output output
Only analyses predictions supplied by the user; never trains models or invents results.
"""
import argparse
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLAN = {'3TX_S4':'A_normal','3TX_S5':'B_changed',
        '3TX_S6':'B_changed','3TX_S7':'A_normal'}
SESSIONS = list(PLAN)
CLASSES = ['EMPTY'] + [f'Z{i}' for i in range(1,10)]
COORDS = {}
for zi in range(1, 10):
    r,c = divmod(zi-1,3)
    COORDS[f'Z{zi}'] = (0.4064+0.7112*c, 1.5240-0.6096*r)
EMPTY_MISS_PENALTY_M = 1.873  # Source-paper scoring rule, NOT continuous localization error.


def zone_error(true, predicted):
    if true == 'EMPTY':
        return np.nan
    if predicted == 'EMPTY':
        return EMPTY_MISS_PENALTY_M
    x1,y1 = COORDS[true]
    x2,y2 = COORDS[predicted]
    return math.hypot(x2-x1,y2-y1)


def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else float('nan')


def assess_group(d):
    n = len(d)
    out = {'n': n}
    for suffix in ['2link', '3link']:
        correct = d[f'correct_{suffix}']
        occupied = d['true_label'] != 'EMPTY'
        empty = ~occupied
        out[f'acc_{suffix}_pct'] = pct(correct.sum(), n)
        out[f'occupied_acc_{suffix}_pct'] = pct((correct & occupied).sum(), occupied.sum())
        out[f'empty_recall_{suffix}_pct'] = pct((correct & empty).sum(), empty.sum())
        out[f'occupied_pred_empty_{suffix}'] = int((occupied & (d[f'pred_{suffix}'] == 'EMPTY')).sum())
        out[f'empty_pred_occupied_{suffix}'] = int((empty & (d[f'pred_{suffix}'] != 'EMPTY')).sum())
        out[f'zone_center_error_{suffix}_m'] = round(float(d[f'error_{suffix}_m'].mean()), 4)
    out['gain_3_minus_2_pp'] = out['acc_3link_pct'] - out['acc_2link_pct']
    out['corrected_by_3link'] = int((~d.correct_2link & d.correct_3link).sum())
    out['harmed_by_3link'] = int((d.correct_2link & ~d.correct_3link).sum())
    out['both_wrong'] = int((~d.correct_2link & ~d.correct_3link).sum())
    out['both_correct'] = int((d.correct_2link & d.correct_3link).sum())
    return out


def require_valid(d):
    needed = ['session','condition','trial_id','true_label','orientation','pred_2link','pred_3link']
    missing = [k for k in needed if k not in d.columns]
    if missing: raise ValueError(f'Missing CSV columns: {missing}')
    if len(d) != 200: raise ValueError(f'Expected 200 trials, found {len(d)}. Do not silently exclude trials.')
    if d[needed].isna().any().any() or d[needed].eq('').any().any():
        bad = d.loc[d[needed].isna().any(axis=1) | d[needed].eq('').any(axis=1), ['session','trial_id']].head(8)
        raise ValueError('Missing prediction(s) or metadata. Fill pred_2link and pred_3link with actual frozen-model outputs.\n' + bad.to_string(index=False))
    for col in ['session','condition','trial_id','true_label','orientation','pred_2link','pred_3link']:
        d[col] = d[col].astype(str).str.strip().str.upper() if col in ['true_label','pred_2link','pred_3link','orientation'] else d[col].astype(str).str.strip()
    if set(d.session) != set(SESSIONS):
        raise ValueError(f'Sessions must be exactly {SESSIONS}. Found {sorted(set(d.session))}')
    if any(PLAN[s] != c for s,c in zip(d.session,d.condition)):
        raise ValueError('Session condition does not match predefined A-B-B-A design')
    if d.duplicated(['session','trial_id']).any():
        raise ValueError('Repeated trial_id within a session')
    for s in SESSIONS:
        a = d[d.session == s]
        if len(a)!=50:
            raise ValueError(f'{s} has {len(a)} trials; expected exactly 50')
        counts = a.true_label.value_counts()
        for c in CLASSES:
            if counts.get(c,0)!=5:
                raise ValueError(f'{s} has {counts.get(c,0)} trials labeled {c}, expected 5')
    for col in ['true_label','pred_2link','pred_3link']:
        invalid = sorted(set(d[col]) - set(CLASSES))
        if invalid: raise ValueError(f'Invalid classes in {col}: {invalid}. Valid: {CLASSES}')
    if (d.trial_id == '').any():
        raise ValueError('Empty trial_id')
    # Validate that true labels agree with canonical trial IDs.
    for s, t, cls in zip(d.session, d.trial_id, d.true_label):
        if not t.upper().startswith(cls + '_'):
            raise ValueError(f'Trial name and label disagree: {s} / {t} / {cls}')
    return d


def bootstrap_intervals(d, rng, reps=3000):
    """Conditional matched-trial resampling; not inference to other rooms or people."""
    sessions = {}
    for s in SESSIONS:
        a = d[d.session == s]
        sessions[s] = {cl: a[a.true_label == cl][['correct_2link','correct_3link']].to_numpy(dtype=float)
                        for cl in CLASSES}
    results = {s: [] for s in SESSIONS}
    results['difference_in_differences'] = []
    for _ in range(reps):
        gains={}
        for s in SESSIONS:
            sampled=[]
            for cl in CLASSES:
                block=sessions[s][cl]
                sampled.append(block[rng.integers(0,len(block),len(block))])
            m=np.vstack(sampled).mean(axis=0)
            gains[s]=100*(m[1]-m[0])
            results[s].append(gains[s])
        a=(gains['3TX_S4']+gains['3TX_S7'])/2
        b=(gains['3TX_S5']+gains['3TX_S6'])/2
        results['difference_in_differences'].append(b-a)
    return {name:(float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))) for name,vals in results.items()}


def save_plots(per, condition_summary, out):
    labels=['S4\nnormal','S5\nchanged','S6\nchanged','S7\nrestored']
    x=np.arange(4); w=.38
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.bar(x-w/2, per.acc_2link_pct, width=w,label='2 links')
    ax.bar(x+w/2, per.acc_3link_pct, width=w,label='3 links')
    ax.set_xticks(x,labels); ax.set_ylabel('Whole-trial accuracy (%)')
    ax.set_ylim(0,100); ax.set_title('Frozen-model performance across A–B–B–A sessions')
    ax.legend(loc='lower left'); fig.tight_layout()
    fig.savefig(out/'figure_accuracy_by_session.png',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    gains=per.gain_3_minus_2_pp.to_numpy()
    ax.bar(x,gains)
    ax.axhline(0,linewidth=1,color='black')
    ax.set_xticks(x,labels)
    ax.set_ylabel('Three-link minus two-link accuracy (percentage points)')
    ax.set_title('Additional-link gain in each independent recording session')
    for xi,yi in zip(x,gains):
        ax.annotate(f'{yi:+.1f}',(xi,yi),xytext=(0,5 if yi>=0 else -14),
                    textcoords='offset points',ha='center',fontsize=10)
    fig.tight_layout()
    fig.savefig(out/'figure_link_gain_by_session.png',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    x=np.arange(2)
    ax.plot(x,condition_summary.acc_2link_pct,marker='o',label='2 links')
    ax.plot(x,condition_summary.acc_3link_pct,marker='o',label='3 links')
    ax.set_xticks(x,['Normal / restored','Chair changed'])
    ax.set_ylabel('Mean whole-trial accuracy (%)')
    ax.set_ylim(0,100); ax.set_title('Within-day environmental-change comparison')
    ax.legend(); fig.tight_layout()
    fig.savefig(out/'figure_condition_comparison.png',dpi=220)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True,help='200-row filled prediction CSV')
    ap.add_argument('--output',default='abba_results',help='Output directory')
    ap.add_argument('--bootstrap-reps',type=int,default=3000)
    args=ap.parse_args()
    ip=Path(args.input).resolve(); output=Path(args.output).resolve()
    df=require_valid(pd.read_csv(ip,dtype=str,keep_default_na=False))
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ['2link','3link']:
        df[f'correct_{suffix}']=(df.true_label==df[f'pred_{suffix}'])
        df[f'error_{suffix}_m']=[zone_error(a,b) for a,b in zip(df.true_label,df[f'pred_{suffix}'])]
    df['correction_status']=np.select([
        ~df.correct_2link & df.correct_3link,
        df.correct_2link & ~df.correct_3link,
        df.correct_2link & df.correct_3link
    ],['corrected_by_3link','harmed_by_3link','both_correct'],default='both_wrong')
    df.to_csv(output/'trial_outcomes.csv',index=False)
    per=pd.DataFrame([{'session':s,'condition':PLAN[s],**assess_group(df[df.session==s])} for s in SESSIONS])
    per.to_csv(output/'per_session_metrics.csv',index=False,float_format='%.4f')
    condition_rows=[]
    for label,ss in [('A_normal',['3TX_S4','3TX_S7']),('B_changed',['3TX_S5','3TX_S6'])]:
        grp=per[per.session.isin(ss)]
        condition_rows.append({'condition':label,'sessions':','.join(ss),
            'acc_2link_pct':float(grp.acc_2link_pct.mean()),
            'acc_3link_pct':float(grp.acc_3link_pct.mean()),
            'gain_pp':float(grp.gain_3_minus_2_pp.mean())})
    condition=pd.DataFrame(condition_rows)
    condition.to_csv(output/'condition_metrics.csv',index=False,float_format='%.4f')
    a,b=condition.iloc[0],condition.iloc[1]
    changed_2=b.acc_2link_pct-a.acc_2link_pct
    changed_3=b.acc_3link_pct-a.acc_3link_pct
    interaction=changed_3-changed_2
    rng=np.random.default_rng(2026)
    cis=bootstrap_intervals(df,rng,reps=args.bootstrap_reps)
    cirows=[{'comparison':s,'low_pp':lo,'high_pp':hi} for s,(lo,hi) in cis.items()]
    pd.DataFrame(cirows).to_csv(output/'paired_bootstrap_conditional.csv',index=False,float_format='%.4f')
    class_rows=[]
    for s in SESSIONS:
        for c in CLASSES:
            v=df[(df.session==s)&(df.true_label==c)]
            class_rows.append({'session':s,'class':c,'n':len(v),
                'acc_2link_pct':pct(v.correct_2link.sum(),len(v)),
                'acc_3link_pct':pct(v.correct_3link.sum(),len(v))})
    pd.DataFrame(class_rows).to_csv(output/'per_class_metrics.csv',index=False,float_format='%.4f')
    save_plots(per,condition,output)
    h=hashlib.sha256(ip.read_bytes()).hexdigest()
    lines=[
        '# TriCSI A–B–B–A analysis (post-collection)', '',
        f'Input CSV SHA-256: `{h}`',
        'All 200 rows (50 per session, five per class) were validated and included.',
        'The same actual trial is paired between the two-link and three-link predictions.',
        '',
        '## Accuracy by session', '',
        '| Session | Condition | 2 links | 3 links | Gain | Corrected | Harmed |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for _,r in per.iterrows():
        lines.append(f'| {r.session} | {r.condition} | {r.acc_2link_pct:.1f}% | {r.acc_3link_pct:.1f}% | {r.gain_3_minus_2_pp:+.1f} pp | {r.corrected_by_3link:d} | {r.harmed_by_3link:d} |')
    lines += ['', '## Condition summaries (descriptive, same room / participant)', '',
              '| Condition | Two-link accuracy | Three-link accuracy | Link gain |',
              '|---|---:|---:|---:|',
              f'| A: normal/restored | {a.acc_2link_pct:.1f}% | {a.acc_3link_pct:.1f}% | {a.gain_pp:+.1f} pp |',
              f'| B: chair changed | {b.acc_2link_pct:.1f}% | {b.acc_3link_pct:.1f}% | {b.gain_pp:+.1f} pp |',
              '', '## Predeclared environmental robustness contrast', '',
              'Accuracy change from A to B = B accuracy minus A accuracy (negative implies deterioration).',
              f'- Two-link accuracy change: **{changed_2:+.1f} percentage points**',
              f'- Three-link accuracy change: **{changed_3:+.1f} percentage points**',
              f'- Difference of changes, (B−A) three-link minus (B−A) two-link: **{interaction:+.1f} percentage points**',
              f'- Equivalently, added-link gain in B minus added-link gain in A: **{b.gain_pp-a.gain_pp:+.1f} percentage points**',
              '', '## Conditional paired bootstrap (not generalizable uncertainty)', '',
              'The 95% intervals below resample **complete trials within each fixed recorded session**, stratified by ground-truth class and paired across model predictions. They do **not** account for variability across future rooms, people, days, or different environmental changes.',
              'Do not interpret these as confidence intervals for real-world robustness.', '']
    for s in SESSIONS:
        lo,hi=cis[s]
        lines.append(f'- {s} added-link gain interval: [{lo:+.1f}, {hi:+.1f}] pp')
    lo,hi=cis['difference_in_differences']
    lines += [f'- Conditional interval for environmental robustness contrast: [{lo:+.1f}, {hi:+.1f}] pp',
              '', '## Limits to state in the paper', '',
              'One person, one room, two sessions per room condition, same-day order A–B–B–A. Changing a chair does not reproduce different homes or long-term environmental drift.',
              'A restoration effect is suggestive but not definitive causal evidence because order and time were not randomized.',
              'Zone-center error uses discrete class assignments and a 1.873 m occupied-as-EMPTY penalty; it is NOT continuously measured positioning error.',
              'Do not add these 200 trials to training or retune the previously frozen model after looking at results.',
              '', '## Output files', '',
              '- `figure_accuracy_by_session.png`: main performance figure',
              '- `figure_link_gain_by_session.png`: each new-session model difference',
              '- `figure_condition_comparison.png`: descriptive A/B response',
              '- `per_session_metrics.csv`, `condition_metrics.csv`, `per_class_metrics.csv`: tables',
              '- `trial_outcomes.csv`: audit of every prediction and error',
              '- `paired_bootstrap_conditional.csv`: within-recording paired-resampling intervals',
              '']
    (output/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:25]))
    print(f'\nSaved analysis to: {output}')

if __name__=='__main__':
    main()
