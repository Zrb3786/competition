#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=${ROOT_DIR:-/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite}
cd "$ROOT_DIR"
python - <<'PY'
from pathlib import Path
import pandas as pd, json, zipfile

candidates=[]
# Known useful candidates from prior runs + v14 experts.
patterns=[
 'outputs/elder_v3_lite/v9_no_cross_raw_motion_5x1/predictions_normal',
 'outputs/elder_v3_lite/v9_no_cross_raw_motion_5x3/predictions_cls_top6_normal_btfix',
 'outputs/elder_v3_lite/v9_no_cross_raw_motion_5x3/predictions_cls_top8_normal_btfix',
 'outputs/elder_v3_lite/v9_no_cross_raw_motion_5x3/predictions_normal',
 'outputs/elder_v14_v12loader/*/predictions_normal',
]
for pat in patterns:
    candidates += list(Path('.').glob(pat))
seen=set()
rows=[]
for d in candidates:
    d=d.resolve()
    if d in seen: continue
    seen.add(d)
    b=d/'binary.csv'; t=d/'ternary.csv'; z=d/'submission.zip'
    if not (b.exists() and t.exists()):
        continue
    try:
        db=pd.read_csv(b); dt=pd.read_csv(t)
        bp='binary_pred' if 'binary_pred' in db.columns else db.columns[-1]
        tp='ternary_pred' if 'ternary_pred' in dt.columns else dt.columns[-1]
        binary=db[bp].astype(int).value_counts().sort_index().to_dict()
        ternary=dt[tp].astype(int).value_counts().sort_index().to_dict()
        severe=dt.loc[dt[tp].astype(int).eq(2), dt.columns[0]].astype(int).tolist()
        inc=int(((db[bp].to_numpy()==0)!=(dt[tp].to_numpy()==0)).sum())
        ok=(inc==0 and binary.get(1,0) in range(10,14) and ternary.get(2,0) in range(2,5) and ternary.get(0,0)>=10)
        rows.append({'path':str(d), 'binary':binary, 'ternary':ternary, 'severe':severe, 'inconsistent':inc, 'safe_shape':ok, 'zip':str(z) if z.exists() else ''})
    except Exception as e:
        rows.append({'path':str(d), 'error':repr(e)})
if not rows:
    print('[WARN] no candidates found')
else:
    df=pd.DataFrame(rows)
    print(df.to_string(index=False))
    print('\n[SAFE-SHAPE CANDIDATES]')
    if 'safe_shape' in df.columns:
        print(df[df['safe_shape'].eq(True)][['path','zip']].to_string(index=False))
PY
