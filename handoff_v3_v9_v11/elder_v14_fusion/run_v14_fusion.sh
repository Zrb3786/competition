#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd /remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite
#   bash elder_v14_fusion/run_v14_fusion.sh
#
# Optional:
#   EXPERTS=p,audio_controlled,gait,video,audio_big
#   SCORE_MODE=cls
#   MAX_POSITIVE=13 MIN_NORMAL=10 MAX_SEVERE=4

ROOT_DIR=${ROOT_DIR:-/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite}
FUSION_DIR=${FUSION_DIR:-$ROOT_DIR/elder_v14_fusion}
EXPERT_ROOT=${EXPERT_ROOT:-$ROOT_DIR/outputs/elder_v14_v12loader}
OUT_DIR=${OUT_DIR:-$ROOT_DIR/outputs/elder_v14_fusion/v14_fusion_v1}

TRAIN_SPLIT_CSV=${TRAIN_SPLIT_CSV:-/remote-home/yangmz/zhangruibo/MPDD-AVG-2026/MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/split_labels_train.csv}

EXPERTS=${EXPERTS:-p,audio_controlled,gait,video,audio_big}
SCORE_MODE=${SCORE_MODE:-balanced}

cd "$ROOT_DIR"

python -u "$FUSION_DIR/v14_fusion.py" \
  --expert_root "$EXPERT_ROOT" \
  --output_dir "$OUT_DIR" \
  --experts "$EXPERTS" \
  --train_split_csv "$TRAIN_SPLIT_CSV" \
  --score_mode "$SCORE_MODE" \
  --max_weight_candidates "${MAX_WEIGHT_CANDIDATES:-3500}" \
  --save_top_k "${SAVE_TOP_K:-20}" \
  --min_positive "${MIN_POSITIVE:-10}" \
  --max_positive "${MAX_POSITIVE:-13}" \
  --min_severe "${MIN_SEVERE:-2}" \
  --max_severe "${MAX_SEVERE:-4}" \
  --min_normal "${MIN_NORMAL:-10}"

echo
echo "[INFO] Candidate submissions:"
find "$OUT_DIR" -path "*/submission.zip" -print | sort

echo
echo "[INFO] Top saved candidates:"
if [[ -f "$OUT_DIR/saved_candidates.csv" ]]; then
  python - <<PY
import pandas as pd
p="$OUT_DIR/saved_candidates.csv"
df=pd.read_csv(p)
print(df.head(20).to_string(index=False))
PY
fi
