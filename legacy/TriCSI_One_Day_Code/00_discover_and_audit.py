"""Discover raw trial files and physical transmitter MACs, audit S8/S9 raw-data QC."""
import argparse, csv, json, statistics
from pathlib import Path
from collections import Counter, defaultdict
from common import trials, trial_info, raw_rows


def audit(root, sessions, output):
    entries=[]
    for session in sessions:
        files=trials(root,session)
        print(f'\n=== {session}: {len(files)} trial CSV files ===')
        inv=Counter(trial_info(p)[0] for p in files)
        print('Trial inventory:',dict(inv))
        if len(files)!=50 or any(inv[c]!=5 for c in ['EMPTY']+[f'Z{i}' for i in range(1,10)]):
            print('WARNING: session not a standard 50-trial/5-per-class session (S8 differs by protocol)')
        session_macs=Counter()
        link_to_macs=defaultdict(Counter)
        for p in files:
            packs=list(raw_rows(p))
            with p.open('r',encoding='utf-8-sig',errors='replace',newline='') as original:
                record_rows=list(csv.reader(original))
            # Headered trial CSV: first line is header, following lines are packets.
            # Legacy serial CSV: count actual CSI_DATA records, not the header.
            header=[c.strip().lower() for c in record_rows[0]] if record_rows else []
            if 'iq_000' in header and 'mac' in header:
                raw_CSI_lines=sum(bool(row) for row in record_rows[1:])
            else:
                raw_CSI_lines=sum(bool(row) and row[0].strip().upper()=='CSI_DATA' for row in record_rows)
            if not packs:
                print('WARNING: no recognized valid CSI packets:',p)
            mac_counts=Counter(x['mac'] for x in packs)
            session_macs.update(mac_counts)
            for rec in packs:
                if rec.get('link'):
                    link_to_macs[rec['link']][rec['mac']] += 1
            counts = {mac:{'n':n,
                  'median_rssi':round(statistics.median(x['rssi'] for x in packs if x['mac']==mac),2)}
                for mac,n in mac_counts.items()}
            lengths=Counter(x['csi_len'] for x in packs)
            modes=Counter(x['sig_mode'] for x in packs if x['sig_mode'])
            channels=Counter(x['channel'] for x in packs if x['channel'])
            tstamp={}
            for mac in mac_counts:
                t=[x['timestamp'] for x in packs if x['mac']==mac and x['timestamp']==x['timestamp']]
                tstamp[mac]={'timestamp_nonmonotonic':sum(b<a for a,b in zip(t,t[1:])),
                              'approx_packet_rate_hz':round((len(t)-1)*1e6/(t[-1]-t[0]),2) if len(t)>1 and t[-1]>t[0] else None}
            entries.append({'session':session,'filename':p.name,'class':trial_info(p)[0],
               'total_valid_packets':len(packs),'raw_CSI_lines':raw_CSI_lines,
               'valid_CSI_ratio':round(len(packs)/raw_CSI_lines,4) if raw_CSI_lines else None,'per_mac':json.dumps(counts),'csi_lengths':json.dumps(lengths),
               'sig_modes':json.dumps(modes),'channels':json.dumps(channels),'timestamp_qc':json.dumps(tstamp)})
            if raw_CSI_lines and len(packs)<.98*raw_CSI_lines:
                print(f'PARSER WARNING: {p.name} {len(packs)}/{raw_CSI_lines} valid packet lines (check CSV format or malformed records)')
            if len(packs) and min(mac_counts.values())<128:
                print(f'LOW PACKET COUNT: {p.name} {dict(mac_counts)}')
        print('MAC totals:',dict(session_macs))
        print('Link -> MAC from CSV (verified as collector-provided labels):',
              {link:dict(macs) for link,macs in sorted(link_to_macs.items())})
        for link,macs in link_to_macs.items():
            if len(macs)!=1:
                print(f'WARNING: {session} link {link} uses {len(macs)} different MAC addresses; verify hardware/source mapping')
        if not session_macs:
            print('PARSER WARNING: No records recognized; do not interpret this as failed acquisition.')
        print('NOTE: MAC identities are NOT inferred from RSSI; consult existing source mapping.')
    if output:
        out=Path(output); out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(entries[0]) if entries else ['session']);w.writeheader();w.writerows(entries)
        print('Saved QC:',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True,help='Directory containing 3TX_S1, ... OR S8, S9 subfolders')
    ap.add_argument('--sessions',nargs='+',default=['3TX_S1','3TX_S2'])
    ap.add_argument('--output',default='audit_trial_qc.csv')
    args=ap.parse_args();audit(args.root,args.sessions,args.output)
