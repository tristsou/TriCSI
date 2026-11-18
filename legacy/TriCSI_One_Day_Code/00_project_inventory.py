"""Read-only inventory of your local TriCSI project. Never executes discovered scripts."""
import argparse,hashlib,re
from pathlib import Path

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',default='.')
    a.add_argument('--output',default='tricsi_project_inventory.txt')
    v=a.parse_args();root=Path(v.root).resolve()
    keywords=re.compile(r'3TX_S3|ExtraTreesClassifier|predict_proba|joblib\.load|pickle\.load|serial\.Serial|921600|CSI_DATA',re.I)
    lines=[f'READ-ONLY TRI-CSI INVENTORY: {root}','']
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '.venv' in p.parts or 'site-packages' in p.parts:continue
        rel=p.relative_to(root)
        if p.suffix.lower() in ('.joblib','.pkl','.pickle','.pt','.pth'):
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f'MODEL  {rel}  {p.stat().st_size} bytes SHA256={h}')
        if p.suffix.lower()=='.py' and p.stat().st_size<750_000:
            txt=p.read_text(encoding='utf-8',errors='replace')
            matches=[(i,x.strip()[:170]) for i,x in enumerate(txt.splitlines(),1) if keywords.search(x)]
            if matches:
                lines.append(f'CODE   {rel}')
                lines += [f'       L{i}: {line}' for i,line in matches[:12]]
    out=Path(v.output);out.write_text('\n'.join(lines),encoding='utf-8')
    print(f'Wrote project inventory to {out}. Review before sharing; it may contain local paths.')
