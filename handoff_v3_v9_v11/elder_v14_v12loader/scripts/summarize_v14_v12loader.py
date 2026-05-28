#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--root', default='outputs/elder_v14_v12loader')
args=ap.parse_args()
root=Path(args.root)
rows=[]
for d in sorted(root.glob('*')):
    if not d.is_dir():
        continue
    m=d/'oof_metrics.json'
    dist=d/'distribution_report.json'
    if not m.exists():
        continue
    r={'run': d.name}
    try: r.update(json.loads(m.read_text()))
    except Exception: pass
    if dist.exists():
        try:
            x=json.loads(dist.read_text())
            r['test_binary']=str(x.get('binary'))
            r['test_ternary']=str(x.get('ternary'))
            r['test_severe']=str(x.get('severe'))
            r['test_inconsistent']=x.get('inconsistent')
        except Exception: pass
    rows.append(r)
if not rows:
    print('[WARN] no expert metrics found under', root)
    raise SystemExit(0)
df=pd.DataFrame(rows)
cols=['run','binary_acc','binary_macro_f1','binary_kappa','ternary_acc','ternary_macro_f1','ternary_kappa','phq_ccc','phq_mae','test_binary','test_ternary','test_severe','test_inconsistent']
cols=[c for c in cols if c in df.columns]
print(df[cols].sort_values(['ternary_macro_f1','ternary_kappa','binary_macro_f1'], ascending=False).to_string(index=False))
out=root/'expert_summary.csv'
df.to_csv(out,index=False)
print('[OK] saved', out)
