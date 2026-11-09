"""Tests pipeline ONLY on synthetic, invented data. NEVER use outputs as science."""
import csv, json, tempfile
from pathlib import Path
import numpy as np
from common import load_config, load_packet_groups, feature_rows,CLASSES,LINKS
from importlib.machinery import SourceFileLoader


def main():
  base=Path(__file__).parent
  with tempfile.TemporaryDirectory(prefix='TRICSI_SYNTHETIC_DO_NOT_REPORT_') as td:
    root=Path(td);d=root/'3TX_S1';d.mkdir()
    mc=['aa:bb:cc:dd:ee:01','aa:bb:cc:dd:ee:02','aa:bb:cc:dd:ee:03']
    cfg=json.loads((base/'config.example.json').read_text());cfg['data_root']=str(root)
    cfg['macs']={k:m for k,m in zip(LINKS,mc)}
    cp=root/'config.json';cp.write_text(json.dumps(cfg))
    pkt=d/'Z1_trial01_N.csv'
    with pkt.open('w',newline='') as f:
      w=csv.writer(f);w.writerow('type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data'.split(','))
      for i in range(80):
        for j,mac in enumerate(mc):
          iq=[int((i%4)+j+k%7) for k in range(128)]
          w.writerow(['CSI_DATA','0',mac,str(-45-j*8),'11','1','0','0','0','0','0','0','0','0','-93','0','6','0',str(i*30000),'0','0','0','128','0','['+','.join(map(str,iq))+']'])
    groups,_=load_packet_groups(pkt,load_config(cp),'3TX_S1')
    z=feature_rows(groups,load_config(cp));assert z.shape==(7,486),z.shape
    print('Raw ESP32 CSV parsing and feature dimensions: PASS')
    rng=np.random.default_rng(100); cfg_fp='demo-only-synthetic-fingerprint'
    training=[]; testing=[]
    for s in ['3TX_S1','3TX_S2','3TX_S4','3TX_S5']:
      for cls in CLASSES:
        for k in range(1,6):
          for win in range(7):
            vec=rng.normal(size=486).astype('float32')+int(cls[1:])*.03 if cls!='EMPTY' else rng.normal(size=486).astype('float32')
            (training if s in ['3TX_S1','3TX_S2'] else testing).append((vec,cls,s,f'{cls}_trial{k:02d}_N.csv',win))
    def save(rows,path):
      np.savez_compressed(path,X=np.stack([a[0] for a in rows]),y=np.array([a[1] for a in rows]),session=np.array([a[2] for a in rows]),trial=np.array([a[3] for a in rows]),window=np.array([a[4] for a in rows],dtype=np.int8),config_fingerprint=np.array(cfg_fp))
    tp=root/'training.npz';ep=root/'heldout.npz';save(training,tp);save(testing,ep)
    trainer=SourceFileLoader('trainer',str(base/'02_train_freeze.py')).load_module()
    evaluator=SourceFileLoader('evaluator',str(base/'03_evaluate_frozen.py')).load_module()
    report=SourceFileLoader('report',str(base/'04_analyze_and_plot.py')).load_module()
    trainer.train(str(tp),str(root/'frozen'),[2026])
    evaluator.evaluate(str(root/'frozen'),str(ep),str(root/'pred.csv'),['3TX_S4','3TX_S5'])
    report.report(str(root/'pred.csv'),str(root/'results'))
    assert (root/'results'/'SUMMARY.md').exists()
    print('Training, frozen evaluation, pair validation, CSV output and figures: PASS')
    print('ALL TEST RESULTS DISCARDED (synthetic; not TriCSI evidence).')

if __name__=='__main__':main()
