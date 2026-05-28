#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/configs/elder_paths.env"

python - <<'PY'
from pathlib import Path
import pandas as pd
ROOT=Path("/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite/outputs/elder_v3_lite")
cands=[
 ("v9_best","v9_no_cross_raw_motion_5x1","predictions_normal"),
 ("v10_B","v10_B_pprior_motion_5x3","predictions_normal"),
 ("v10_C","v10_C_audio_pprior_motion_5x3","predictions_normal"),
]
for name, exp, pred in cands:
    p=ROOT/exp/pred
    print("\n==========", name, exp, pred, "==========")
    if not (p/'binary.csv').exists():
        print('[MISSING]', p); continue
    b=pd.read_csv(p/'binary.csv')
    t=pd.read_csv(p/'ternary.csv')
    m=b[['id','binary_pred','phq9_pred']].merge(t[['id','ternary_pred']], on='id').sort_values('id')
    bad=m[((m.binary_pred==0)&(m.ternary_pred>0))|((m.binary_pred==1)&(m.ternary_pred==0))]
    print('binary:', m.binary_pred.value_counts().sort_index().to_dict())
    print('ternary:', m.ternary_pred.value_counts().sort_index().to_dict())
    print('inconsistent:', len(bad), bad.id.astype(int).tolist())
    print('severe:', m[m.ternary_pred==2].id.astype(int).tolist())
    cv=ROOT/exp/'cv_summary.csv'
    if cv.exists():
        df=pd.read_csv(cv)
        cols=[c for c in ['best_score','binary_macro_f1','binary_kappa','ternary_macro_f1','ternary_kappa','phq_ccc'] if c in df.columns]
        print('cv mean:')
        print(df[cols].mean())
    print(m.to_string(index=False))
PY
