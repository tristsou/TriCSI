"""AI-assisted read-only analysis of the supplied transfer diagnostic archive.

Creates descriptive EMPTY-profile summaries and figures, not new model fits.
Requires the extracted diagnostic ZIP's directories as --input. Original files
are never edited. All calculated comparisons concern supplied recordings only.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SESSIONS=[f'3TX_S{i}' for i in range(1,6)]
LINKS=['back_left','middle_back','back_right']
ACTIVE=list(range(6,32))+list(range(33,59))
GUARD=sorted(set(range(64))-set(ACTIVE))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.input;out=a.output
    out.mkdir(parents=True,exist_ok=False)
    rows=[];profiles={};profile_rows=[]
    source_hashes={}
    for session in SESSIONS:
        path=root/session/'EMPTY_trial01_NA.csv'
        source_hashes[str(path.relative_to(root))]=sha(path)
        f=pd.read_csv(path,low_memory=False)
        for link,g in f.groupby('link'):
            raw=g[[f'iq_{i:03d}' for i in range(128)]].to_numpy(float)
            amp=np.hypot(raw[:,0::2],raw[:,1::2])
            log=np.log1p(amp[:,ACTIVE])
            gain=np.median(log,axis=1)
            shape=log-gain[:,None]
            profile=np.median(shape,axis=0);profiles[(session,link)]=profile
            times=g.host_time_s.to_numpy()
            rows.append(dict(session=session,link=link,packets=len(g),
                             raw_iq_nonfinite=int((~np.isfinite(raw)).sum()),
                             rssi_median_dbm=float(g.rssi.median()),
                             gain_log_median=float(np.median(gain)),
                             median_active_amplitude=float(np.median(amp[:,ACTIVE])),
                             zero_active_fraction=float(np.mean(amp[:,ACTIVE]==0)),
                             nonzero_guard_fraction=float(np.mean(amp[:,GUARD]!=0)),
                             rate_hz=float((len(g)-1)/(times.max()-times.min())),
                             mcs_median=float(g.mcs.median()),sgi_fraction=float(g.sgi.mean()),
                             bandwidth_1_fraction=float(g.bandwidth.mean()),
                             channel_values=str(sorted(g.channel.unique().tolist())),
                             first_host_utc=datetime.fromtimestamp(times.min(),timezone.utc).isoformat()))
            profile_rows.extend(dict(session=session,link=link,subcarrier_index=sc,
                                     median_packet_centered_log_amplitude=float(value))
                                for sc,value in zip(ACTIVE,profile))
    pd.DataFrame(rows).to_csv(out/'empty_link_summary.csv',index=False)
    pd.DataFrame(profile_rows).to_csv(out/'empty_csi_profiles.csv',index=False)
    distances=[]
    for link in LINKS:
        for session in SESSIONS[1:]:
            x=profiles[('3TX_S1',link)];y=profiles[(session,link)]
            distances.append(dict(reference='3TX_S1',session=session,link=link,
                                  across_subcarrier_profile_correlation=float(np.corrcoef(x,y)[0,1]),
                                  profile_rmse=float(np.sqrt(np.mean((x-y)**2)))))
    pd.DataFrame(distances).to_csv(out/'empty_profile_comparison_to_s1.csv',index=False)
    all_pred=[];checks=[]
    for session in SESSIONS[3:]:
        directory=root/'evaluations'/session
        f=pd.read_csv(directory/'trial_predictions.csv')
        all_pred.append(f)
        meta=json.loads((directory/'evaluation.json').read_text())
        cols=['p_EMPTY']+[f'p_Z{i}' for i in range(1,10)]
        checks.append(dict(session=session,
                           result_hashes_match=all(sha(directory/k)==v for k,v in meta['result_sha256'].items()),
                           probability_sum_max_error=float(abs(f[cols].sum(axis=1)-1).max()),
                           decisions_match_argmax=bool((f[cols].idxmax(axis=1).str.replace('p_','',regex=False)==f.predicted_zone).all()),
                           prediction_rows=len(f),empty_prediction_count=int((f.predicted_zone=='EMPTY').sum())))
    predictions=pd.concat(all_pred,ignore_index=True)
    predictions.groupby(['session','model','predicted_zone']).size().rename('trial_count').to_csv(out/'prediction_class_counts.csv')
    predictions[predictions.true_zone=='EMPTY'][['session','model','trial_uid','predicted_zone','p_EMPTY','confidence']].to_csv(out/'actual_empty_predictions.csv',index=False)
    m=json.loads((root/'manifest.json').read_text())
    reproduction=json.loads((root/'s3_historical_reproduction_20261030/reproduction.json').read_text())
    checks_report=dict(prediction_checks=checks,extractor_sha256=sha(root/'tricsi_3tx_modeling.py'),
                       extractor_matches_frozen_manifest=sha(root/'tricsi_3tx_modeling.py')==m['original_module_sha256'],
                       historical_model_hash_matches_manifest=reproduction['original_model_sha256']==m['historical']['sha256'],
                       raw_s3_reproduction_passed=reproduction['historical_accuracy_matches'],
                       raw_s3_reproduction_metrics=reproduction['observed'],
                       source_empty_csv_sha256=source_hashes,
                       limits='Only EMPTY trial 1 raw packets supplied per session. Full occupied raw data/model bytes not included in diagnostic ZIP.')
    (out/'consistency_checks.json').write_text(json.dumps(checks_report,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),sharey=True)
    colors=['#a8b6ca','#6a86ab','#254f85','#d57a1b','#b43b45']
    for ax,link in zip(axes,LINKS):
        for session,color in zip(SESSIONS,colors):
            ax.plot(ACTIVE,profiles[(session,link)],label=session.replace('3TX_',''),color=color,
                    lw=2 if session in SESSIONS[3:] else 1.4)
        ax.axhline(0,color='#bbbbbb',lw=.7)
        ax.set(title=link.replace('_',' ').title(),xlabel='Saved subcarrier index')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Median packet-centered log amplitude')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False)
    fig.suptitle('EMPTY-state CSI profiles by recording session',y=1.01)
    fig.text(.01,.015,'Graph generated by ChatGPT using Python and Matplotlib from student-collected data, 2026.',fontsize=7)
    fig.tight_layout(rect=(0,.06,1,.85))
    for ext in ['png','pdf','svg']:fig.savefig(out/('empty_csi_profiles.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(pd.DataFrame(rows)[['session','link','packets','rssi_median_dbm','rate_hz']].round(3).to_string(index=False))
    print('Consistency checks:',checks_report['extractor_matches_frozen_manifest'],checks_report['raw_s3_reproduction_passed'])


if __name__=='__main__':main()
