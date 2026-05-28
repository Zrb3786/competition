#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite"
TRAIN_ROOT="/remote-home/yangmz/zhangruibo/MPDD-AVG-2026/MPDD-AVG2026/MPDD-AVG2026-trainval/Elder"
TEST_ROOT="/remote-home/yangmz/zhangruibo/MPDD-AVG-2026/MPDD-AVG2026/MPDD-AVG2026-test/Elder"
TRAIN_SPLIT_CSV="$TRAIN_ROOT/split_labels_train.csv"

OUT_DIR="$CODE_DIR/outputs/elder_v3_lite"
FEATURE_DIR="$OUT_DIR/features"

TRAIN_DESC_CSV="$TRAIN_ROOT/descriptions.csv"
TEST_DESC_CSV="$TEST_ROOT/descriptions.csv"
if [ ! -f "$TEST_DESC_CSV" ]; then
  TEST_DESC_CSV="$TRAIN_DESC_CSV"
fi

P_EMBED_NPY="$TRAIN_ROOT/descriptions_embeddings_with_ids.npy"
TRAIN_MOTION_NPZ="$FEATURE_DIR/elder_raw_video_motion_train.npz"
TEST_ID_CSV="$FEATURE_DIR/test_ids_from_official_imu.csv"
TEST_MOTION_NPZ="$FEATURE_DIR/elder_raw_video_motion_test_official_ids_train_fallback.npz"

mkdir -p "$FEATURE_DIR" "$OUT_DIR/logs"
cd "$CODE_DIR"

python scripts/patch_v9_depformerv2.py
python -m py_compile mpdd_elder_v3_lite.py

python -u "$CODE_DIR/mpdd_elder_v3_lite.py" parse-desc \
  --train_desc "$TRAIN_DESC_CSV" \
  --test_desc "$TEST_DESC_CSV" \
  --output_dir "$FEATURE_DIR"

predict_topk () {
  local EXP_DIR="$1"
  local K="$2"
  local MODE="$3"

  local OUT_NAME
  if [ "$MODE" = "best" ]; then
    OUT_NAME="predictions_top${K}_normal"
    SCORE_EXPR='cv["select_score"] = cv["best_score"]'
  else
    OUT_NAME="predictions_cls_top${K}_normal"
    SCORE_EXPR='cv["select_score"] = 0.30*cv["binary_macro_f1"] + 0.30*cv["ternary_macro_f1"] + 0.20*cv["binary_kappa"] + 0.20*cv["ternary_kappa"]'
  fi

  python - <<PY
from pathlib import Path
import pandas as pd

exp = Path("$EXP_DIR")
cv = pd.read_csv(exp / "cv_summary.csv")
$SCORE_EXPR
sel = cv.sort_values("select_score", ascending=False).head(int("$K"))
out = exp / "${MODE}_top${K}_ckpts.txt"
out.write_text(",".join(sel["checkpoint"].astype(str).tolist()))
print(sel[["fold","seed","best_score","select_score","binary_macro_f1","binary_kappa","ternary_macro_f1","ternary_kappa","phq_ccc"]].to_string(index=False))
print("[OK]", out)
PY

  CKPTS=$(cat "$EXP_DIR/${MODE}_top${K}_ckpts.txt")

  python -u "$CODE_DIR/mpdd_elder_v3_lite.py" predict \
    --test_data_root "$TEST_ROOT" \
    --test_split_csv "$TEST_ID_CSV" \
    --p_struct_test_csv "$FEATURE_DIR/elder_descriptions_struct_test.csv" \
    --p_embed_npy "$P_EMBED_NPY" \
    --motion_test_npz "$TEST_MOTION_NPZ" \
    --checkpoints "$CKPTS" \
    --output_dir "$EXP_DIR/$OUT_NAME" \
    --batch_size 16 \
    --device cuda
}

run_exp () {
  local NAME="$1"
  local ARCH="$2"
  local SEEDS="$3"
  local MOTION_TRAIN="$4"
  local MOTION_TEST="$5"
  local EPOCHS="$6"
  local PATIENCE="$7"
  local TOPK="$8"

  EXP_DIR="$OUT_DIR/$NAME"

  echo
  echo "============================================================"
  echo "RUN $NAME | arch=$ARCH | seeds=$SEEDS"
  echo "============================================================"

  python -u "$CODE_DIR/mpdd_elder_v3_lite.py" train \
    --train_data_root "$TRAIN_ROOT" \
    --train_split_csv "$TRAIN_SPLIT_CSV" \
    --p_struct_train_csv "$FEATURE_DIR/elder_descriptions_struct_train.csv" \
    --p_embed_npy "$P_EMBED_NPY" \
    --motion_train_npz "$MOTION_TRAIN" \
    --output_dir "$EXP_DIR" \
    --model_arch "$ARCH" \
    --audio_features wav2vec,opensmile \
    --official_video_features "" \
    --use_gait 1 \
    --folds 5 \
    --seeds "$SEEDS" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size 8 \
    --hidden_dim 96 \
    --p_embed_bottleneck 48 \
    --dropout 0.45 \
    --modality_dropout 0.18 \
    --reg_weight 0.10 \
    --ccc_weight 0.03 \
    --binary_weight 1.20 \
    --consistency_weight 0.05 \
    --ordinal_weight 0.05 \
    --soft_f1_weight 0.25 \
    --kappa_weight 0.15 \
    --class_weight_power 0.75 \
    --lr 5e-4 \
    --weight_decay 3e-3 \
    --device cuda

  python -u "$CODE_DIR/mpdd_elder_v3_lite.py" predict \
    --test_data_root "$TEST_ROOT" \
    --test_split_csv "$TEST_ID_CSV" \
    --p_struct_test_csv "$FEATURE_DIR/elder_descriptions_struct_test.csv" \
    --p_embed_npy "$P_EMBED_NPY" \
    --motion_test_npz "$MOTION_TEST" \
    --checkpoint_dir "$EXP_DIR/checkpoints" \
    --output_dir "$EXP_DIR/predictions_normal" \
    --batch_size 16 \
    --device cuda

  if [ "$TOPK" != "0" ]; then
    predict_topk "$EXP_DIR" "$TOPK" "best"
    predict_topk "$EXP_DIR" 6 "cls"
    predict_topk "$EXP_DIR" 8 "cls"
  fi

  echo "[DONE] $NAME"
  echo "normal: $EXP_DIR/predictions_normal/submission.zip"
  if [ "$TOPK" != "0" ]; then
    echo "top${TOPK}: $EXP_DIR/predictions_top${TOPK}_normal/submission.zip"
    echo "cls_top6: $EXP_DIR/predictions_cls_top6_normal/submission.zip"
    echo "cls_top8: $EXP_DIR/predictions_cls_top8_normal/submission.zip"
  fi
}

run_exp "v9_depformerv2_raw_motion_5x3" "v9_depformerv2" "42,43,44" "$TRAIN_MOTION_NPZ" "$TEST_MOTION_NPZ" 80 14 8
run_exp "v9_no_pguide_raw_motion_5x1" "v9_no_pguide" "42" "$TRAIN_MOTION_NPZ" "$TEST_MOTION_NPZ" 70 12 0
run_exp "v9_no_cross_raw_motion_5x1" "v9_no_cross" "42" "$TRAIN_MOTION_NPZ" "$TEST_MOTION_NPZ" 70 12 0
run_exp "v9_no_sp_raw_motion_5x1" "v9_no_sp" "42" "$TRAIN_MOTION_NPZ" "$TEST_MOTION_NPZ" 70 12 0
run_exp "v9_depformerv2_no_motion_5x1" "v9_depformerv2" "42" "" "" 70 12 0

echo
echo "========== ALL DONE =========="
echo "Main top8:     $OUT_DIR/v9_depformerv2_raw_motion_5x3/predictions_top8_normal/submission.zip"
echo "Main cls-top6: $OUT_DIR/v9_depformerv2_raw_motion_5x3/predictions_cls_top6_normal/submission.zip"
echo "Main cls-top8: $OUT_DIR/v9_depformerv2_raw_motion_5x3/predictions_cls_top8_normal/submission.zip"
echo "No P guide:    $OUT_DIR/v9_no_pguide_raw_motion_5x1/predictions_normal/submission.zip"
echo "No cross:      $OUT_DIR/v9_no_cross_raw_motion_5x1/predictions_normal/submission.zip"
echo "No shared/private: $OUT_DIR/v9_no_sp_raw_motion_5x1/predictions_normal/submission.zip"
echo "No motion:     $OUT_DIR/v9_depformerv2_no_motion_5x1/predictions_normal/submission.zip"
