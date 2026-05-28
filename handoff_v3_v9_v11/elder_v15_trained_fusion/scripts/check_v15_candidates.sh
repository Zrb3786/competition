#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite}
cd "$ROOT"
OUT_ROOT=${OUT_ROOT:-outputs/elder_v15_trained_fusion}
python - <<'PY'
from pathlib import Path
import pandas as pd, json
root = Path(__import__('os').environ.get('OUT_ROOT','outputs/elder_v15_trained_fusion'))
for sub in sorted(root.glob('*/predictions_model_*')):
    b = sub/'binary.csv'; t = sub/'ternary.csv'; meta = sub/'distribution_report.json'
    if not b.exists() or not t.exists():
        continue
    bd = pd.read_csv(b); td = pd.read_csv(t)
    bcol = 'binary_pred' if 'binary_pred' in bd.columns else bd.columns[-1]
    tcol = 'ternary_pred' if 'ternary_pred' in td.columns else td.columns[-1]
    idcol = 'id' if 'id' in td.columns else td.columns[0]
    bb = bd[bcol].astype(int).to_numpy(); tt = td[tcol].astype(int).to_numpy()
    dist = {
        'binary': pd.Series(bb).value_counts().sort_index().to_dict(),
        'ternary': pd.Series(tt).value_counts().sort_index().to_dict(),
        'severe': td.loc[td[tcol].astype(int).eq(2), idcol].astype(int).tolist(),
        'inconsistent': int(((bb==0)!=(tt==0)).sum()),
    }
    safe = dist['inconsistent']==0 and 10 <= dist['binary'].get(1,0) <= 13 and 2 <= dist['ternary'].get(2,0) <= 4 and dist['ternary'].get(0,0) >= 10
    print('\n==', sub)
    print(dist, 'SAFE=', safe)
    print('zip=', sub/'submission.zip')
PY
