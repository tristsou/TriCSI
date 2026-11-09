"""Rebuild TriCSI paper plots from saved CSV/JSON files, without training models.

Run: python make_verified_tricsi_figures.py --project-root PATH --output PATH
Requires numpy, pandas, matplotlib, python-docx. No uploaded Python is executed.
"""
from pathlib import Path
import argparse, ast, hashlib, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

LABELS = ['EMPTY'] + [f'Z{i}' for i in range(1, 10)]
ORDER = ['1tx_back_left','1tx_middle_back','1tx_back_right','2tx_left_middle',
         '2tx_middle_right','2tx_outer','3tx_all']
NAMES = ['Left only','Middle only','Right only','Left + middle','Middle + right',
         'Left + right (outer pair)','All three']
BLUE, ORANGE, TEAL, GRAY = '#28678C','#B15F24','#267566','#737C85'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'axes.labelsize':10,'legend.fontsize':9,'savefig.dpi':350,'svg.fonttype':'none',
 'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).parent/'inputs')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output')
    args=parser.parse_args()
    root=args.project_root.resolve(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    sources={}; checks=[]; sections=[]; supplements=[]
    def source(rel):
        p=root/rel
        if not p.is_file(): raise FileNotFoundError(p)
        sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    def csv(rel): return pd.read_csv(source('results/'+rel),keep_default_na=True)
    def js(rel): return json.loads(source('results/'+rel).read_text(encoding='utf-8-sig'))
    def check(name,a,b,tol=1e-9):
        if not np.allclose(a,b,rtol=tol,atol=tol,equal_nan=True):
            raise ValueError(f'{name}: {a!r} != {b!r}')
        checks.append(name)
    def save(fig,stem):
        for ext in ['png','svg','pdf']:
            fig.savefig(out/f'{stem}.{ext}',bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def style(ax):
        ax.set_axisbelow(True); ax.grid(axis='y',color='#E0E3E6',linewidth=.7)
        ax.set_ylim(0,105); ax.set_yticks(range(0,101,20))
    def export(df,stem): df.to_csv(out/f'{stem}.csv',index=False)
    def acc(frame,true,pred):
        assert frame[true].notna().all() and frame[pred].notna().all()
        assert set(frame[true]).issubset(LABELS) and set(frame[pred]).issubset(LABELS)
        return (frame[true]==frame[pred]).astype(int)
    credit='Source: Tristan Tsou experimental results, 2026. Plotted with Python and Matplotlib using ChatGPT-assisted code.'

    # R1. Consistent whole-trial aggregation, not best-window or majority metrics.
    records=[]
    configs=[
      ('Random Forest','Within S1','crossval_s1/crossval_trial_predictions.csv','crossval_s1/crossval_summary.json','aggregate_oof_trial_mean_probability'),
      ('Random Forest','Within S2','crossval_s2/crossval_trial_predictions.csv','crossval_s2/crossval_summary.json','aggregate_oof_trial_mean_probability'),
      ('CNN','Within S1','cnn_5fold_s1/oof_trial_predictions.csv','cnn_5fold_s1/summary.json','aggregate_oof_trial_mean_probability'),
      ('CNN','Within S2','cnn_5fold_s2/oof_trial_predictions.csv','cnn_5fold_s2/summary.json','aggregate_oof_trial_mean_probability'),
      ('Random Forest','S1 to S2','cross_session_s1_to_s2/trial_predictions.csv','cross_session_s1_to_s2/summary.json','trial_mean_probability_metrics'),
      ('Random Forest','S2 to S1','cross_session_s2_to_s1/trial_predictions.csv','cross_session_s2_to_s1/summary.json','trial_mean_probability_metrics'),
      ('CNN','S1 to S2','cnn_cross_session_s1_to_s2/trial_predictions.csv','cnn_cross_session_s1_to_s2/summary.json','trial_mean_probability_metrics')]
    for model,test,fp,jp,key in configs:
        d=csv(fp); correct=acc(d,'zone_true','pred_mean_probability')
        assert not d.trial_group.duplicated().any()
        check(fp,correct.mean(),js(jp)[key]['accuracy'])
        records.append(dict(model=model,test=test,correct=int(correct.sum()),n_trials=len(d),accuracy=correct.mean(),source=fp))
    early=pd.DataFrame(records); export(early,'R1_early_models_data')
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.6),sharey=True)
    for ax,tests,title in zip(axes,[['Within S1','Within S2'],['S1 to S2','S2 to S1']],
                             ['A  New trials in the same session','B  Testing on a different session']):
        for j,(model,color) in enumerate([('Random Forest',BLUE),('CNN',ORANGE)]):
            for i,test in enumerate(tests):
                row=early[(early.model==model)&(early.test==test)]
                x=i+(j-.5)*.36
                if row.empty:
                    ax.text(x,5,'Not\navailable',ha='center',va='bottom',fontsize=8,color=GRAY)
                    continue
                row=row.iloc[0]; val=100*row.accuracy
                ax.bar(x,val,.34,color=color,label=model if i==0 else None)
                ax.text(x,val+2,f'{val:.0f}%',ha='center',fontsize=9)
        ax.set_xticks([0,1],tests); ax.set_xlim(-.6,1.6)
        ax.set_title(title,loc='left');style(ax)
    axes[0].set_ylabel('Correct trial predictions (%)')
    axes[0].legend(frameon=False,loc='upper right')
    fig.tight_layout(w_pad=1.4); save(fig,'R1_early_models')
    sections.append(('R1_early_models','Early models and testing across sessions',
      'Within-session testing used five folds, keeping each full trial together. Random Forest correctly classified 38 of 50 S1 trials (76%) and 30 of 50 S2 trials (60%). The CNN reached 70% and 62%, respectively. When trained on S1 and tested on S2, accuracy fell to 30% for Random Forest and 8% for the CNN. Random Forest reached 28% in the reverse direction. These results show that performance on new trials from a familiar session did not reliably predict performance in another recording session.',
      'Whole-trial accuracy for the early two-link models. A: five-fold testing within each session. B: training on one entire session and testing on the other. Each available bar represents 50 trial predictions. Every trial decision uses the highest mean class probability across its windows. The CNN reverse-direction result was not present in the supplied outputs; it is not treated as zero. The two panels use different training/testing arrangements.'))

    # R2. Preserve all held-out development sessions and variant seed differences.
    ab=csv('tricsi_anchor_ablation/ablation_folds.csv')
    summary=js('tricsi_anchor_ablation/ablation_summary.json')
    variants=['FULL','NO_BACKGROUND','NO_ORIENTATION_INVARIANCE']
    pretty=['Full model','Without empty-room\nsubtraction','Without orientation\nalignment']
    means=[]
    for v in variants:
        sub=ab[ab.variant==v]; assert len(sub)==4 and set(sub.test_session)=={'S4','S5','S6','S7'}
        assert (sub.n_trials==49).all()
        check(v,sub.exact_accuracy.mean(),summary[v]['mean_exact_accuracy'])
        means.append(sub.exact_accuracy.mean()*100)
    fig,ax=plt.subplots(figsize=(7.5,4.1))
    ax.bar(np.arange(3),means,width=.63,color='#D9E4EA')
    for s,offset,color,marker in zip(['S4','S5','S6','S7'],[-.18,-.06,.06,.18],[BLUE,TEAL,GRAY,ORANGE],['o','s','^','D']):
        vals=[100*float(ab[(ab.variant==v)&(ab.test_session==s)].exact_accuracy.iloc[0]) for v in variants]
        ax.scatter(np.arange(3)+offset,vals,color=color,marker=marker,s=42,label=f'Test {s}',zorder=3)
    for i,val in enumerate(means):ax.text(i,6,f'Mean {val:.1f}%',ha='center',weight='bold',fontsize=9)
    ax.set_xticks(range(3),pretty);ax.set_ylabel('Correct trial predictions (%)');style(ax)
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.16),frameon=False)
    fig.tight_layout();save(fig,'R2_model_components');export(ab,'R2_model_components_data')
    sections.append(('R2_model_components','Tests of the two-link model components',
      'The component tests used S4–S7 as development data. Each run trained on three sessions and tested on the remaining session. Mean trial accuracy was 78.6% for the full model, 74.5% without empty-room subtraction, and 75.5% without the training term that aligns different body orientations. S7 was the hardest session for all three versions. These tests are consistent with both components being useful, but their effects were not uniform across sessions. Because the versions also used different random training seeds, the differences cannot be attributed entirely to the removed component.',
      'Bars show mean accuracy across four held-out development sessions; symbols show the individual session scores. Each run scores 49 trials, excluding EMPTY trial 1. Background subtraction removes an empty-room reference from the CSI features. Orientation alignment encourages recordings of the same zone to remain similar when the person faces different directions. This is an exploratory component comparison, not a repeated-seed significance test.'))

    # R3. Recompute S8/S9 metrics, matrices, and rates directly from predictions.
    two=[]; confs=[]; zone_counts=[]
    for session in ['S8','S9']:
        d=csv(f'tricsi_anchor_confirmatory/{session}_trial_predictions.csv')
        m=js(f'tricsi_anchor_confirmatory/{session}_metrics.json')
        c=acc(d,'true_zone','pred_zone');assert not d.group.duplicated().any()
        check(session+' correct',c,d.correct);check(session+' accuracy',c.mean(),m['exact_accuracy'])
        check(session+' count',len(d),m['n_trials']);check(session+' successes',c.sum(),m['correct_trials'])
        cm=pd.crosstab(d.true_zone,d.pred_zone).reindex(index=LABELS,columns=LABELS,fill_value=0).fillna(0).astype(int)
        check(session+' confusion',cm.values,m['confusion_matrix']);confs.append(cm)
        check(session+' median error',d.weighted_xy_error_m.median(),m['median_weighted_xy_error_m'])
        two.append({'session':session,'correct':int(c.sum()),'n_trials':len(d),'accuracy':c.mean()})
        zone_counts.append([int(((d.true_zone==z)&(d.pred_zone==z)).sum()) for z in LABELS])
        assert (d.groupby('true_zone').size().reindex(LABELS)==4).all()
    pooled=js('tricsi_anchor_confirmatory/pooled_confirmatory_summary.json')
    check('pooled two-link',sum(x['correct'] for x in two)/sum(x['n_trials'] for x in two),pooled['pooled_exact_accuracy'])
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8.2,3.4),gridspec_kw={'width_ratios':[1,1.8]})
    ax.bar([0,1],[100*x['accuracy'] for x in two],color=[BLUE,ORANGE],width=.6)
    for i,row in enumerate(two):ax.text(i,100*row['accuracy']+2,f"{row['accuracy']*100:g}%\n{row['correct']}/{row['n_trials']}",ha='center',fontsize=9)
    ax.set_xticks([0,1],['S8','S9']);ax.set_ylim(0,118);ax.set_yticks(range(0,101,20));ax.set_ylabel('Correct trial predictions (%)');ax.set_title('A  Frozen model by session',loc='left');ax.grid(axis='y',color='#E0E3E6');ax.set_axisbelow(True)
    bx.imshow(zone_counts,cmap='Blues',vmin=0,vmax=4,aspect='auto')
    bx.set_xticks(range(10),LABELS,rotation=45,ha='right');bx.set_yticks([0,1],['S8','S9'])
    for i in range(2):
        for j in range(10):bx.text(j,i,f'{zone_counts[i][j]}/4',ha='center',va='center',fontsize=9,color='white' if zone_counts[i][j]>=3 else 'black')
    bx.set_xlabel('Actual zone');bx.set_title('B  Correct predictions out of four trials',loc='left')
    fig.tight_layout(w_pad=1.5);save(fig,'R3_frozen_two_link_sessions');export(pd.DataFrame(two),'R3_frozen_two_link_data')
    sections.append(('R3_frozen_two_link_sessions','The frozen two-link model on later sessions',
      'Using the saved model without retraining, the confirmatory evaluator correctly classified 38 of 40 S8 trials (95%) but only 21 of 40 S9 trials (52.5%). Across the two sessions, it classified 59 of 80 trials correctly (73.75%). S8 had two errors, while S9 had errors in several zones, including all four Z3 trials. The model therefore performed well in S8 but did not maintain that accuracy in S9. The remaining session-to-session variation is an important limitation of this two-link system.',
      'A: trial accuracy for the same saved two-link model. B: the number of correct predictions for each class, with four test trials per class per session. The evaluator uses only EMPTY trial 1 for target-session background calibration; no labeled occupied test-session examples enter its prototypes. S8 scores occupied trials 3–6 and EMPTY trials 2–5; S9 scores occupied trials 1–4 and EMPTY trials 2–5.'))

    # R4. Compare seven subsets; recompute all displayed metrics from trial rows.
    dev=csv('3tx_dev_comparison/trial_predictions.csv'); final=csv('3tx_s3_final/s3_trial_predictions.csv')
    devsum=csv('3tx_dev_comparison/subset_summary.csv').set_index('subset')
    finalsum=csv('3tx_s3_final/s3_subset_metrics.csv').set_index('subset')
    devreport=js('3tx_dev_comparison/development_report.json'); finalreport=js('3tx_s3_final/s3_final_metrics.json')
    assert finalreport['test_session']=='3TX_S3' and set(final.session)=={'3TX_S3'}
    assert set(dev.subset)==set(final.subset)==set(ORDER)
    geometry=None
    for node in ast.parse(source('project_config.py').read_text(encoding='utf-8-sig')).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ZONE_CENTERS' for t in node.targets):geometry=ast.literal_eval(node.value)
    assert geometry is not None
    pts=np.array(list(geometry.values())); penalty=float(np.linalg.norm(pts.max(axis=0)-pts.min(axis=0)))
    combined=[]
    for frame,label,summary,metric in [(dev,'Development',devsum,'mean_trial_accuracy'),(final,'Final S3',finalsum,'trial_accuracy')]:
        check(label+' correct flags',acc(frame,'true_zone','predicted_zone'),frame.correct)
        assert not frame.duplicated(['subset','trial_uid']).any()
        for row in frame.itertuples():
            err=np.nan if row.true_zone=='EMPTY' else penalty if row.predicted_zone=='EMPTY' else float(np.linalg.norm(np.array(geometry[row.true_zone])-geometry[row.predicted_zone]))
            if not np.isclose(err,row.position_error_m,equal_nan=True,atol=1e-8):raise ValueError('Position error mismatch')
        for subset in ORDER:
            d=frame[frame.subset==subset];correct=d.correct.mean();error=d.position_error_m.mean()
            check(label+subset+' accuracy',correct,summary.loc[subset,metric]);check(label+subset+' error',error,summary.loc[subset,'mean_position_error_m'])
            assert len(d)==(100 if label=='Development' else 50)
            combined.append(dict(stage=label,subset=subset,n_trials=len(d),correct=int(d.correct.sum()),accuracy=correct,mean_position_error_m=error))
    combo=pd.DataFrame(combined);export(combo,'R4_link_comparison_data')
    fig,axes=plt.subplots(1,2,figsize=(9.2,5.2),sharey=True)
    for ax,key,mult,title,limit in zip(axes,['accuracy','mean_position_error_m'],[100,1],['A  Correct trial predictions (%)','B  Mean occupied-trial error score (m)'],[105,1.0]):
        for j,(stage,color) in enumerate([('Development',BLUE),('Final S3',TEAL)]):
            d=combo[combo.stage==stage].set_index('subset').loc[ORDER];vals=d[key].to_numpy()*mult
            bars=ax.barh(np.arange(7)+(j-.5)*.35,vals,height=.32,color=color,label='S1/S2 development' if j==0 else 'Final 3TX_S3 test')
            for bar,val in zip(bars,vals):ax.text(val+limit*.012,bar.get_y()+bar.get_height()/2,f'{val:.0f}%' if mult==100 else f'{val:.3f}',va='center',fontsize=8.5)
        ax.set_xlim(0,limit);ax.grid(axis='x',color='#E0E3E6');ax.set_axisbelow(True);ax.set_title(title,loc='left')
        ax.axhspan(5.65,6.4,color='#EDF4F0',zorder=-1)
    axes[0].set_yticks(range(7),NAMES);axes[0].invert_yaxis();axes[0].legend(frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(-.65,1.10))
    axes[1].tick_params(axis='y',left=False);fig.tight_layout(w_pad=1.7);save(fig,'R4_three_link_comparison')
    sections.append(('R4_three_link_comparison','Comparing one, two, and three transmitters',
      'A separate experiment compared all seven transmitter subsets using the same three-link recordings and an Extra Trees classifier. During S1/S2 development, all three links reached 79% mean trial accuracy and the outer pair reached 77%. On the later S3 test, all three reached 84% (42/50), compared with 70% (35/50) for the outer pair. The left-plus-middle pair reached 78%, and the three individual links ranged from 48% to 68%. The full three-link model had been selected during development, before evaluating S3. These results are separate from the two-link neural-network experiment and should not be compared directly with its S8 accuracy.',
      'All seven fixed link subsets are shown in the same order. Development averages the two whole-session tests, S1 to S2 and S2 to S1; each contains 50 trials. Final models were fitted on S1+S2 and evaluated on the same 50 S3 trials. All three links form the selected primary model. Error is the distance between actual and predicted zone centers for occupied trials. An occupied trial predicted as EMPTY receives a fixed '+f'{penalty:.3f}'+' m penalty, the diagonal of the zone-center bounds. EMPTY trials are excluded from this error average. This score is not a measure of continuous tracking precision.'))

    # R5. Trial-level confusion counts, independently reconciled with saved matrix.
    primary=final[final.subset==finalreport['primary_subset']]
    cm=pd.crosstab(primary.true_zone,primary.predicted_zone).reindex(index=LABELS,columns=LABELS,fill_value=0).fillna(0).astype(int)
    saved=csv('3tx_s3_final/s3_primary_confusion_matrix.csv').set_index('Unnamed: 0')
    check('S3 saved confusion',cm.values,saved.loc[LABELS,LABELS].values)
    check('S3 primary accuracy',np.trace(cm)/cm.values.sum(),finalreport['primary_metrics']['trial_accuracy'])
    def confusion(ax,matrix,title,maxcount):
        im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=maxcount)
        for i in range(10):
            ax.add_patch(Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor=TEAL,lw=1.0))
            for j in range(10):
                val=matrix[i,j]
                if val:ax.text(j,i,str(val),ha='center',va='center',color='white' if val>=maxcount*.6 else 'black',fontsize=10)
        ax.set_xticks(range(10),LABELS,rotation=45,ha='right');ax.set_yticks(range(10),LABELS)
        ax.set_xlabel('Predicted zone');ax.set_ylabel('Actual zone');ax.set_title(title,loc='left');return im
    fig,ax=plt.subplots(figsize=(6.6,5.4));im=confusion(ax,cm.values,'All three links on S3: 42 of 50 trials correct',5)
    fig.colorbar(im,ax=ax,label='Number of trials',ticks=range(6),shrink=.8);fig.tight_layout();save(fig,'R5_s3_confusion_matrix')
    cm.to_csv(out/'R5_s3_confusion_counts.csv')
    sections.append(('R5_s3_confusion_matrix','Where the final model made mistakes',
      'The three-link model correctly recognized all five EMPTY trials and 37 of 45 occupied trials (82.2%). Four occupied trials were incorrectly labeled EMPTY: three in Z3 and one in Z5. The other four errors were Z4 to Z6, Z7 to Z4, Z8 to Z5, and Z9 to Z6. Thus, recognizing an empty room perfectly did not mean detecting every occupied trial. The mean occupied-trial error score was 0.239 m for all three links and 0.463 m for the outer pair, including the predefined penalties for missed occupancy.',
      'Each row contains five actual trials. Columns show the predicted class. Numbers on the outlined diagonal are correct predictions; numbers elsewhere are errors. Blank cells contain zero trials. The top row measures EMPTY recall, while entries in the first column below it are missed occupied trials.'))

    # R6. Validate the stored paired comparison and reproduce its bootstrap.
    intervals=[]; outcomes=[]
    for stage,path,report,key,frame in [('Development','3tx_dev_comparison/paired_outer_vs_3tx.csv',devreport,'middle_link_test',dev),('Final S3','3tx_s3_final/s3_paired_outer_vs_3tx.csv',finalreport,'s3_middle_link_contrast',final)]:
        paired=csv(path)
        a=frame[frame.subset=='2tx_outer'].set_index('trial_uid');b=frame[frame.subset=='3tx_all'].set_index('trial_uid');p=paired.set_index('trial_uid')
        assert set(p.index)==set(a.index)==set(b.index) and not p.index.duplicated().any()
        check(stage+' pair outer',p.correct_outer,a.loc[p.index,'correct']);check(stage+' pair all',p.correct_3tx,b.loc[p.index,'correct'])
        check(stage+' gain rows',p.accuracy_gain_3tx,p.correct_3tx-p.correct_outer)
        check(stage+' error gain rows',p.error_reduction_3tx_m,a.loc[p.index,'position_error_m']-b.loc[p.index,'position_error_m'])
        for col,metric,seed in [('accuracy_gain_3tx','accuracy_gain',2026),('error_reduction_3tx_m','position_error_reduction_m',2027)]:
            groups=[g[col].dropna().to_numpy(float) for _,g in paired.groupby('true_zone')];groups=[g for g in groups if len(g)]
            rng=np.random.default_rng(seed);draws=np.array([np.mean(np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])) for _ in range(5000)])
            val=np.mean(np.concatenate(groups));lo,hi=np.percentile(draws,[2.5,97.5]);expected=report[key][metric]
            check(stage+metric+' interval',[val,lo,hi],[expected['estimate'],expected['ci95_low'],expected['ci95_high']])
            intervals.append(dict(stage=stage,metric=metric,estimate=val,ci95_low=lo,ci95_high=hi))
        counts=paired.groupby(['correct_outer','correct_3tx']).size()
        outcomes.append(dict(stage=stage,both_correct=int(counts.get((1,1),0)),only_three_correct=int(counts.get((0,1),0)),only_outer_correct=int(counts.get((1,0),0)),both_wrong=int(counts.get((0,0),0))))
    interval=pd.DataFrame(intervals);export(interval,'R6_paired_intervals_data');export(pd.DataFrame(outcomes),'R6_paired_outcomes_data')
    fig,ax=plt.subplots(figsize=(7.8,2.8));ax.axvline(0,color=GRAY,ls='--',lw=1)
    for y,stage,color in [(1,'Development',BLUE),(0,'Final S3',TEAL)]:
        r=interval[(interval.stage==stage)&(interval.metric=='accuracy_gain')].iloc[0];v,lo,hi=np.array([r.estimate,r.ci95_low,r.ci95_high])*100
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=5,ms=7,lw=2)
        ax.text(hi+1.2,y,f'{v:+.0f} points\n95% interval: {lo:+.0f} to {hi:+.0f}',va='center',fontsize=9)
    ax.set_yticks([1,0],['S1/S2 development','Final S3']);ax.set_ylim(-.65,1.65);ax.set_xlim(-10,39)
    ax.set_xticks([-10,0,10,20,30]);ax.set_xlabel('Accuracy difference: all three minus outer pair (percentage points)')
    ax.grid(axis='x',color='#E0E3E6');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.tight_layout();save(fig,'R6_middle_link_gain')
    sections.append(('R6_middle_link_gain','How much the middle link helped',
      'On S3, both configurations were correct on 35 trials and wrong on eight. Adding the middle link corrected seven additional trials and did not turn any correct outer-pair prediction into an error. This produced a 14-percentage-point accuracy gain. The saved paired resampling analysis gave a 95% interval of +6 to +22 points. During development, the gain was only two points, with an interval of −6 to +10 points, which included no improvement. The S3 result supports a benefit in this recorded session, but one final session does not establish how large the benefit would be in other rooms or on other days.',
      'Dots show accuracy differences and horizontal lines show 95% paired bootstrap intervals, reproduced with 5,000 resamples and the saved random seed. Trials are resampled within their actual class, keeping the two model predictions paired. A line crossing zero includes no benefit. These intervals reflect resampling trials from the recorded sessions, not variation across new rooms, people, or future sessions.'))

    # Supplemental A: the exact north-facing/window numbers used in the old draft.
    north=csv('n_only_diagnostic/n_only_results.csv');nr=north[north.model=='RF'].set_index('comparison')
    keys=['S1_N_repeatability_MEAN','S2_N_repeatability_MEAN','S1_N_to_S2_N','S2_N_to_S1_N']
    fig,ax=plt.subplots(figsize=(7.5,3.5))
    for j,(col,color,label) in enumerate([('window_accuracy',BLUE,'Window accuracy'),('trial_mean_probability_accuracy',ORANGE,'Whole-trial accuracy')]):
        vals=100*nr.loc[keys,col].to_numpy();bars=ax.bar(np.arange(4)+(j-.5)*.36,vals,.34,color=color,label=label)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+1.5,f'{v:.1f}%',ha='center',fontsize=8.5)
    ax.set_xticks(range(4),['Within S1\nN1 vs N5','Within S2\nN1 vs N5','S1 to S2\nNorth only','S2 to S1\nNorth only']);style(ax)
    ax.set_ylabel('Correct predictions (%)');ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.16));fig.tight_layout();save(fig,'A1_north_facing_diagnostic')
    export(nr.loc[keys].reset_index(),'A1_north_facing_data')
    supplements.append(('A1_north_facing_diagnostic','North-facing Random Forest diagnostic. The first two comparisons average N1-to-N5 and N5-to-N1 directions. Window and trial accuracy are deliberately separated. The 71.5%, 83.6%, and approximately 39% values in the previous draft were window scores for this restricted diagnostic, not general whole-trial scores.'))
    # Supplemental B: paired session confusion matrices, same scale.
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8))
    for ax,matrix,s in zip(axes,confs,['S8','S9']):confusion(ax,matrix.values,s+' frozen two-link predictions',4)
    fig.tight_layout(w_pad=2);save(fig,'A2_two_link_confusion_matrices')
    supplements.append(('A2_two_link_confusion_matrices','Two-link confusion counts for S8 and S9. Both panels use a 0–4 color scale and four trials per actual class. The diagonal marks correct predictions; blank cells are zero.'))
    # Supplemental C: distinct development tests, never claimed as a controlled gain.
    devrows=[]
    for label,fp,jp in [('Early RF on S3','rf_s1_s2_to_s3/trial_predictions.csv','rf_s1_s2_to_s3/summary.json'),('Standardized RF on S6','rf_s4_s5_to_s6/trial_predictions.csv','rf_s4_s5_to_s6/summary.json')]:
        d=csv(fp);v=acc(d,'zone_true','pred_mean_probability').mean();check(label,v,js(jp)['trial_mean_probability_metrics']['accuracy'])
        devrows.append(dict(test=label,accuracy=v,n_trials=len(d)))
    d=csv('tricsi_adapt_final/final_S7_trial_predictions.csv');m=js('tricsi_adapt_final/final_S7_metrics.json')
    for label,col,key in [('TriCSI-Adapt S7\nSource references','pred_zero','zero_shot_trial_accuracy'),('TriCSI-Adapt S7\nOne target example per class','pred_one','one_shot_trial_accuracy')]:
        v=acc(d,'true_zone',col).mean();check(label,v,m[key]);devrows.append(dict(test=label,accuracy=v,n_trials=len(d)))
    fig,ax=plt.subplots(figsize=(7.8,3.6));vals=[100*x['accuracy'] for x in devrows];ax.bar(range(4),vals,color=[GRAY,BLUE,TEAL,ORANGE],width=.62)
    for i,row in enumerate(devrows):ax.text(i,100*row['accuracy']+2,f"{100*row['accuracy']:g}%\nn = {row['n_trials']}",ha='center',fontsize=9)
    ax.set_xticks(range(4),[r['test'].replace(' RF on ',' RF\non ') for r in devrows],fontsize=9);style(ax);ax.set_ylabel('Correct trial predictions (%)')
    fig.tight_layout();save(fig,'A3_intermediate_development_tests');export(pd.DataFrame(devrows),'A3_intermediate_data')
    supplements.append(('A3_intermediate_development_tests','Selected intermediate two-link tests. The early S3 and standardized S6 Random Forest runs each score 50 trials. The two S7 TriCSI-Adapt variants score the same 40 evaluation trials. These different sessions and model versions document development history and do not isolate the effect of any single change. Two-link S3 is not the later three-link session 3TX_S3.'))

    # Deliverable text, exact input hashes, and minimal data required to reproduce.
    for item in sections: assert (out/(item[0]+'.png')).is_file()
    payload=[dict(stem=s,title=t,text=b,caption=c+' '+credit) for s,t,b,c in sections]
    (out/'captions_and_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    md=['# TriCSI results\n']
    for item in payload:md += ['## '+item['title'],item['text'],f"Figure {item['stem'].split('_')[0]}. "+item['caption'],'']
    md += ['## Optional supplemental figures']
    for stem,caption in supplements:md += [stem,caption+' '+credit,'']
    (out/'Results_and_Captions.md').write_text('\n\n'.join(md),encoding='utf-8')
    audit={'source_root':str(root),'checks_passed':checks,'inputs_sha256':sources,'no_model_training':True,'no_image_generation':True,'s3_empty_prediction_penalty_m':penalty}
    (out/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    copyroot=out/'inputs'
    for rel in sources:
        dest=copyroot/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if (root/rel).resolve()!=dest.resolve():shutil.copy2(root/rel,dest)
    # Include the exact relevant analysis code for methods verification, not execution.
    for rel in ['train_crossval.py','train_csi_cnn.py','evaluate_frozen_anchor_confirmatory.py','train_tricsi_anchor_final.py','run_tricsi_anchor_ablations.py','tricsi_3tx_modeling_bundle/tricsi_3tx_modeling.py','tricsi_3tx_modeling_bundle/compare_and_freeze_3tx.py','tricsi_3tx_modeling_bundle/evaluate_frozen_3tx.py']:
        p=root/rel
        if p.is_file():
            target=out/'reference_code'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    # Accessible, editable results section: one figure with its explanation per page.
    from docx import Document
    from docx.shared import Inches,Pt,RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc=Document();sec=doc.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=sec.bottom_margin=Inches(.7);sec.left_margin=sec.right_margin=Inches(.8)
    for name in ['Normal','Title','Heading 1','Caption']:
        st=doc.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
    doc.styles['Normal'].font.size=Pt(11);doc.styles['Normal'].paragraph_format.space_after=Pt(8)
    doc.styles['Normal'].paragraph_format.line_spacing=1.12
    doc.styles['Heading 1'].font.size=Pt(15);doc.styles['Title'].font.size=Pt(20)
    doc.styles['Caption'].font.size=Pt(9.5);doc.styles['Caption'].font.italic=False
    doc.styles['Caption'].font.bold=False
    for i,item in enumerate(payload):
        if i:doc.add_page_break()
        else:doc.add_paragraph('TriCSI Results and Figures','Title')
        doc.add_paragraph(item['title'],'Heading 1');doc.add_paragraph(item['text'])
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        run=p.add_run();run.add_picture(str(out/(item['stem']+'.png')),width=Inches(6.7))
        inline=run._r.xpath('.//wp:docPr')[0];inline.set('descr',item['caption'])
        p=doc.add_paragraph(style='Caption')
        p.add_run(f"Figure R{i+1}. ").bold=True
        p.add_run(item['caption'])
    doc.save(out/'TriCSI_Results_and_Figures.docx')
    print(json.dumps({'output':str(out),'main_figures':len(sections),'supplemental_figures':len(supplements),'checks_passed':len(checks),'sources':len(sources)},indent=2))

if __name__=='__main__':main()
