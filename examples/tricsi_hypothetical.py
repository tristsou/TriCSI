"""INVENTED planning illustration. No measurements, predictions, or upper bounds.
Run: python tricsi_hypothetical.py (requires numpy and matplotlib).
All 196 cells are invented, including any coincidental matches to real results.
"""
from pathlib import Path
from itertools import combinations
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
CONFIGS = ['L','C','R','LC','LR','CR','LCR']
# Hand-selected optimistic scenario. Even percentages correspond to integer
# correct counts out of 50 trials. These are NOT generated CSI or model outputs.
VALUES = {
'S1': [[48,54,56,64,64,62,72],[50,56,54,66,62,64,74],[46,52,54,62,60,62,70],
       [56,64,62,76,72,72,84],[54,62,60,74,70,72,82],[56,64,60,76,72,74,84],
       [60,68,66,82,78,80,90]],
'S2': [[50,52,60,64,68,64,74],[52,56,58,68,66,66,76],[48,54,56,64,64,62,72],
       [58,62,66,76,76,74,86],[56,62,64,74,74,74,84],[56,64,64,76,74,76,86],
       [62,68,70,82,80,82,92]],
'S3': [[44,56,50,62,62,60,72],[46,54,52,64,64,62,72],[48,58,50,66,62,64,74],
       [50,66,56,76,72,70,86],[52,68,58,78,72,74,86],[52,66,58,76,74,72,84],
       [58,72,64,84,80,80,92]],
'S4': [[46,50,52,60,60,58,68],[48,52,54,62,62,60,70],[48,54,52,64,62,62,72],
       [54,60,58,72,68,68,80],[54,62,60,74,70,70,82],[56,62,60,74,70,72,82],
       [60,66,64,80,76,78,88]]}
rows=[]
for test in VALUES:
    others=[s for s in VALUES if s != test]
    splits=[c for n in (1,2,3) for c in combinations(others,n)]
    for train,values in zip(splits,VALUES[test]):
        assert test not in train and all(0<=v<=100 and v%2==0 for v in values)
        rows.append({'status':'INVENTED','test':test,'train':'+'.join(train),
                     'n_sessions':len(train),**dict(zip(CONFIGS,values))})
assert len(rows)==28
with (OUT/'tricsi_hypothetical.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
md=['# INVENTED optimistic scenario — not experimental results',
    'All 196 accuracies are invented. Not forecasts, measured findings, or an upper bound. L/C/R = left/center/right. All values are percentages.',
    '| Test | Train | L | C | R | LC | LR | CR | LCR |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    md.append('| '+' | '.join([r['test'],r['train']]+[str(r[c]) for c in CONFIGS])+' |')
(OUT/'tricsi_hypothetical_table.md').write_text('\n\n'.join(md[:2])+'\n\n'+'\n'.join(md[2:]))
# Average training subsets WITHIN test sessions first, then weight four tests equally.
means=np.array([[np.mean([[r[c] for c in CONFIGS] for r in rows
                 if r['test']==s and r['n_sessions']==n],axis=0)
                 for n in (1,2,3)] for s in VALUES])
grand=means.mean(axis=0)
fig,axes=plt.subplots(1,3,figsize=(15,5.5),gridspec_kw={'width_ratios':[1.1,1,1]})
fig.suptitle('INVENTED optimistic scenario — not experimental results',fontsize=18,fontweight='bold',y=.98)
colors=['#b2a477','#7f9c9a','#a39dc5','#80b0cc','#236a9b','#6da08a','#c55737']
for i,c in enumerate(CONFIGS):
    axes[0].plot([1,2,3],grand[:,i],'-o',label=c,color=colors[i],lw=2.7 if c in ('LR','LCR') else 1.4)
axes[0].set(title='A. Accuracy by configuration',ylabel='Invented accuracy (%)',ylim=(40,100))
axes[0].legend(ncol=4,fontsize=9,loc='upper left',frameon=False)
for s,v in zip(VALUES,means):
    axes[1].plot([1,2,3],v[:,6]-v[:,4],'-o',alpha=.45,label=s)
gap=grand[:,6]-grand[:,4]
axes[1].plot([1,2,3],gap,'k-o',lw=3,label='Mean')
axes[1].set(title='B. Three-link advantage over LR',ylabel='Invented difference (percentage points)',ylim=(0,20))
axes[1].legend(ncol=3,fontsize=9,frameon=False)
groups=[[0,1,2],[3,4,5],[6]]
heat=np.array([grand[:,g].mean(axis=1) for g in groups])
im=axes[2].imshow(heat,vmin=40,vmax=100,cmap='YlGnBu',aspect='auto')
for i in range(3):
    for j in range(3):
        axes[2].text(j,i,f'{heat[i,j]:.1f}%',ha='center',va='center',color='white' if heat[i,j]>73 else '#17323e',fontsize=14)
axes[2].set(title='C. Two dimensions of improvement',xticks=[0,1,2],xticklabels=[1,2,3],yticks=[0,1,2],yticklabels=['1 TX (L, C, R)','2 TX (LC, LR, CR)','3 TX (LCR)'])
for ax in axes[:2]:
    ax.set_xticks([1,2,3]); ax.grid(alpha=.15); ax.spines[['top','right']].set_visible(False)
for ax in axes: ax.set_xlabel('Number of training sessions')
fig.text(.5,.025,'Hand-selected values illustrate a favorable hypothesis; no uncertainty or significance is inferred.\n50 / 100 / 150 training trials: session diversity and training-set size remain confounded. Equal weighting of test sessions and subsets.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.11,1,.92])
fig.savefig(OUT/'tricsi_hypothetical.png',dpi=200)
print('Mean accuracies (rows: 1, 2, 3 training sessions; columns: L C R LC LR CR LCR):')
print(np.round(grand,2)); print('LCR - LR:',gap)
