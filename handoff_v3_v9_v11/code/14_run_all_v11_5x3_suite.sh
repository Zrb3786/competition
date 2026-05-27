#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# V11 5x3 suite
# Main first:
#   allA + fullVG + no_cross
#
# Feature sets:
#   newA = A_big only, no official wav2vec/opensmile
#   allA = official wav2vec/opensmile + A_big
#
# VG sets:
#   fullVG    = rawG + GUnit + VBeh
#   noVBeh   = rawG + GUnit, no VBeh
#   noGUnit  = rawG + VBeh, no GUnit
#   rawGOnly = rawG only, no GUnit/VBeh
#   GUnitOnly= GUnit only, no rawG/VBeh
#
# Fusion:
#   no_cross = inherit v9_no_cross setting
#   cross    = optional pair-level cross
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 你可以在外面 export ELDER_PATHS_ENV=... 覆盖
ELDER_PATHS_ENV="${ELDER_PATHS_ENV:-/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite/mpdd_v10_feature_scripts/configs/elder_paths.env}"

if [ ! -f "$ELDER_PATHS_ENV" ]; then
  echo "[ERROR] cannot find ELDER_PATHS_ENV=$ELDER_PATHS_ENV"
  exit 2
fi

set -a
source "$ELDER_PATHS_ENV"
set +a

CODE="$PKG_DIR/src/mpdd_elder_v11_audio_vg.py"
if [ ! -f "$CODE" ]; then
  echo "[ERROR] cannot find v11 code: $CODE"
  exit 2
fi

# 输出与日志
STATUS_TXT="$OUT_DIR/run_status_v11_5x3.txt"
RUN_LOG_DIR="$OUT_DIR/logs/v11_5x3_suite"
mkdir -p "$RUN_LOG_DIR" "$OUT_DIR" "$FEATURE_DIR"

# v11 feature paths
AUDIO_BIG_PCA_TRAIN="$FEATURE_DIR/elder_audio_big_v10_pca256_train.npz"
AUDIO_BIG_PCA_TEST="$FEATURE_DIR/elder_audio_big_v10_pca256_test.npz"

MOTION_BEHAVIOR_TRAIN="$FEATURE_DIR/elder_motion_behavior_v10_train.npz"
MOTION_BEHAVIOR_TEST="$FEATURE_DIR/elder_motion_behavior_v10_test.npz"

GAIT_UNIT_TRAIN="$FEATURE_DIR/elder_gait_unit_v10_train.npz"
GAIT_UNIT_TEST="$FEATURE_DIR/elder_gait_unit_v10_test.npz"

# P clean：优先用未混入 MG 的 P；没有就退回当前 P
P_EXTRA_TRAIN="${P_V10_TRAIN:-$FEATURE_DIR/elder_p_v10_train.csv}"
P_EXTRA_TEST="${P_V10_TEST:-$FEATURE_DIR/elder_p_v10_test.csv}"

if [ -f "$FEATURE_DIR/elder_p_v10_train.base_no_mg.csv" ] && [ -f "$FEATURE_DIR/elder_p_v10_test.base_no_mg.csv" ]; then
  P_EXTRA_TRAIN="$FEATURE_DIR/elder_p_v10_train.base_no_mg.csv"
  P_EXTRA_TEST="$FEATURE_DIR/elder_p_v10_test.base_no_mg.csv"
fi

# structured P
P_STRUCT_TRAIN="$FEATURE_DIR/elder_descriptions_struct_train.csv"
P_STRUCT_TEST="$FEATURE_DIR/elder_descriptions_struct_test.csv"

# 如果 v10 P 已经包含原始 structured P，v11 仍然单独传 structured + extra
# P_EMBED_NPY 来自官方 descriptions_embeddings_with_ids.npy

# 默认超参：继承 v9_no_cross 分类优先设置
FOLDS="${FOLDS:-5}"
SEEDS="${SEEDS:-42,43,44}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-14}"
BATCH_SIZE="${BATCH_SIZE:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-96}"
P_EMBED_BOTTLENECK="${P_EMBED_BOTTLENECK:-48}"
DROPOUT="${DROPOUT:-0.50}"
MODALITY_DROPOUT="${MODALITY_DROPOUT:-0.25}"
LR="${LR:-4e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-3}"
DEVICE="${DEVICE:-cuda}"

BINARY_WEIGHT="${BINARY_WEIGHT:-1.20}"
SOFT_F1_WEIGHT="${SOFT_F1_WEIGHT:-0.25}"
KAPPA_WEIGHT="${KAPPA_WEIGHT:-0.15}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.05}"
ORDINAL_WEIGHT="${ORDINAL_WEIGHT:-0.05}"
REG_WEIGHT="${REG_WEIGHT:-0.06}"
CCC_WEIGHT="${CCC_WEIGHT:-0.02}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.75}"

TOPK="${TOPK:-8}"

# arch 名称自动兼容
HELP_TXT="$RUN_LOG_DIR/train_help.txt"
python "$CODE" train -h > "$HELP_TXT" 2>&1 || true

if grep -q "v11_no_cross" "$HELP_TXT"; then
  ARCH_NO_CROSS="v11_no_cross"
elif grep -q "v11_audio_vg_no_cross" "$HELP_TXT"; then
  ARCH_NO_CROSS="v11_audio_vg_no_cross"
else
  ARCH_NO_CROSS="${ARCH_NO_CROSS:-v11_no_cross}"
fi

if grep -q "v11_cross" "$HELP_TXT"; then
  ARCH_CROSS="v11_cross"
elif grep -q "v11_audio_vg_cross" "$HELP_TXT"; then
  ARCH_CROSS="v11_audio_vg_cross"
else
  ARCH_CROSS="${ARCH_CROSS:-v11_cross}"
fi

echo "========== V11 5x3 suite ==========" | tee -a "$STATUS_TXT"
echo "time=$(date)" | tee -a "$STATUS_TXT"
echo "CODE=$CODE" | tee -a "$STATUS_TXT"
echo "ELDER_PATHS_ENV=$ELDER_PATHS_ENV" | tee -a "$STATUS_TXT"
echo "ARCH mapping: newA_no_cross=v11_newA_no_cross allA_no_cross=v11_allA_no_cross newA_cross=v11_newA_cross allA_cross=v11_allA_cross" | tee -a "$STATUS_TXT"
echo "STATUS_TXT=$STATUS_TXT" | tee -a "$STATUS_TXT"
echo | tee -a "$STATUS_TXT"

require_file () {
  local p="$1"
  local name="$2"
  if [ ! -f "$p" ]; then
    echo "[ERROR] missing $name: $p" | tee -a "$STATUS_TXT"
    exit 2
  fi
}

echo "========== Check required features =========="
require_file "$TRAIN_SPLIT_CSV" "TRAIN_SPLIT_CSV"
require_file "$TEST_ID_CSV" "TEST_ID_CSV"
require_file "$P_STRUCT_TRAIN" "P_STRUCT_TRAIN"
require_file "$P_STRUCT_TEST" "P_STRUCT_TEST"
require_file "$P_EXTRA_TRAIN" "P_EXTRA_TRAIN"
require_file "$P_EXTRA_TEST" "P_EXTRA_TEST"
require_file "$P_EMBED_NPY" "P_EMBED_NPY"
require_file "$TRAIN_MOTION_NPZ" "TRAIN_MOTION_NPZ"
require_file "$TEST_MOTION_NPZ" "TEST_MOTION_NPZ"
require_file "$AUDIO_BIG_PCA_TRAIN" "AUDIO_BIG_PCA_TRAIN"
require_file "$AUDIO_BIG_PCA_TEST" "AUDIO_BIG_PCA_TEST"
require_file "$MOTION_BEHAVIOR_TRAIN" "MOTION_BEHAVIOR_TRAIN"
require_file "$MOTION_BEHAVIOR_TEST" "MOTION_BEHAVIOR_TEST"
require_file "$GAIT_UNIT_TRAIN" "GAIT_UNIT_TRAIN"
require_file "$GAIT_UNIT_TEST" "GAIT_UNIT_TEST"

echo "[OK] required feature files exist."

append_done () {
  local exp="$1"
  local outdir="$2"
  echo -e "$(date '+%F %T')\tDONE\t$exp\t$outdir" >> "$STATUS_TXT"
}

append_fail () {
  local exp="$1"
  local outdir="$2"
  echo -e "$(date '+%F %T')\tFAILED\t$exp\t$outdir" >> "$STATUS_TXT"
}

is_done () {
  local exp="$1"
  grep -q $'\tDONE\t'"$exp"$'\t' "$STATUS_TXT" 2>/dev/null
}

predict_topk () {
  local exp_dir="$1"
  local k="$2"
  local mode="$3"

  local score_expr
  local out_name
  local ckpt_txt

  if [ "$mode" = "best" ]; then
    out_name="predictions_top${k}_normal"
    ckpt_txt="best_top${k}_ckpts.txt"
    score_expr='cv["select_score"] = cv["best_score"]'
  else
    out_name="predictions_cls_top${k}_normal"
    ckpt_txt="cls_top${k}_ckpts.txt"
    score_expr='cv["select_score"] = 0.30*cv["binary_macro_f1"] + 0.30*cv["ternary_macro_f1"] + 0.20*cv["binary_kappa"] + 0.20*cv["ternary_kappa"]'
  fi

  python - <<PY
from pathlib import Path
import pandas as pd

exp = Path("$exp_dir")
cv_path = exp / "cv_summary.csv"
if not cv_path.exists():
    raise SystemExit(f"missing {cv_path}")
cv = pd.read_csv(cv_path)
$score_expr
sel = cv.sort_values("select_score", ascending=False).head(int("$k"))
out = exp / "$ckpt_txt"
out.write_text(",".join(sel["checkpoint"].astype(str).tolist()))
print("\\n[SELECT] $exp_dir $mode top$k")
cols = [c for c in ["fold","seed","best_score","select_score","binary_macro_f1","binary_kappa","ternary_macro_f1","ternary_kappa","phq_ccc","checkpoint"] if c in sel.columns]
print(sel[cols].to_string(index=False))
print("[OK]", out)
PY

  local ckpts
  ckpts="$(cat "$exp_dir/$ckpt_txt")"

  python -u "$CODE" predict \
    --test_data_root "$TEST_ROOT" \
    --test_split_csv "$TEST_ID_CSV" \
    --p_struct_test_csv "$P_STRUCT_TEST" \
    --p_extra_csv "$P_EXTRA_TEST" \
    --p_embed_npy "$P_EMBED_NPY" \
    --motion_test_npz "$TEST_MOTION_NPZ" \
    --audio_big_npz "$AUDIO_BIG_PCA_TEST" \
    --motion_extra_npz "$motion_behavior_test" \
    --gait_extra_npz "$gait_unit_test" \
    --checkpoints "$ckpts" \
    --output_dir "$exp_dir/$out_name" \
    --batch_size 16 \
    --device "$DEVICE"
}

run_analysis () {
  echo
  echo "========== Analyze current candidates =========="
  if [ -f "$PKG_DIR/scripts/13_analyze_v11_candidates.py" ]; then
    python "$PKG_DIR/scripts/13_analyze_v11_candidates.py" | tee "$OUT_DIR/v11_candidate_report_latest.txt" || true
  else
    echo "[WARN] no scripts/13_analyze_v11_candidates.py found, skip analysis."
  fi
}

run_exp () {
  local exp_name="$1"
  local fusion="$2"          # no_cross / cross
  local audio_mode="$3"      # newA / allA
  local vg_mode="$4"         # fullVG / noVBeh / noGUnit / rawGOnly / GUnitOnly

  local exp_dir="$OUT_DIR/$exp_name"
  local log_file="$RUN_LOG_DIR/${exp_name}.log"

  if is_done "$exp_name" && [ -f "$exp_dir/predictions_normal/submission.zip" ]; then
    echo "[SKIP DONE] $exp_name"
    return 0
  fi

  mkdir -p "$exp_dir"

  local arch=""
  case "${audio_mode}_${fusion}" in
    newA_no_cross)
      arch="v11_newA_no_cross"
      ;;
    allA_no_cross)
      arch="v11_allA_no_cross"
      ;;
    newA_cross)
      arch="v11_newA_cross"
      ;;
    allA_cross)
      arch="v11_allA_cross"
      ;;
    *)
      echo "[ERROR] unsupported audio_mode/fusion combination: audio_mode=$audio_mode fusion=$fusion"
      exit 2
      ;;
  esac

  local audio_features=""
  if [ "$audio_mode" = "allA" ]; then
    audio_features="wav2vec,opensmile"
  elif [ "$audio_mode" = "newA" ]; then
    audio_features=""
  else
    echo "[ERROR] bad audio_mode=$audio_mode"
    exit 2
  fi

  local use_gait=1
  local gait_unit_train="$GAIT_UNIT_TRAIN"
  local gait_unit_test="$GAIT_UNIT_TEST"
  local motion_behavior_train="$MOTION_BEHAVIOR_TRAIN"
  local motion_behavior_test="$MOTION_BEHAVIOR_TEST"

  case "$vg_mode" in
    fullVG)
      use_gait=1
      ;;
    noVBeh)
      motion_behavior_train=""
      motion_behavior_test=""
      use_gait=1
      ;;
    noGUnit)
      gait_unit_train=""
      gait_unit_test=""
      use_gait=1
      ;;
    rawGOnly)
      gait_unit_train=""
      gait_unit_test=""
      motion_behavior_train=""
      motion_behavior_test=""
      use_gait=1
      ;;
    GUnitOnly)
      # 不使用原始 gait sequence，只用 GUnit token
      use_gait=0
      motion_behavior_train=""
      motion_behavior_test=""
      ;;
    *)
      echo "[ERROR] bad vg_mode=$vg_mode"
      exit 2
      ;;
  esac

  echo | tee -a "$STATUS_TXT"
  echo "============================================================" | tee -a "$STATUS_TXT"
  echo "START $exp_name" | tee -a "$STATUS_TXT"
  echo "fusion=$fusion arch=$arch audio_mode=$audio_mode audio_features=$audio_features vg_mode=$vg_mode use_gait=$use_gait" | tee -a "$STATUS_TXT"
  echo "exp_dir=$exp_dir" | tee -a "$STATUS_TXT"
  echo "log=$log_file" | tee -a "$STATUS_TXT"
  echo "============================================================" | tee -a "$STATUS_TXT"

  (
    set -x

    python -u "$CODE" train \
      --train_data_root "$TRAIN_ROOT" \
      --train_split_csv "$TRAIN_SPLIT_CSV" \
      --p_struct_train_csv "$P_STRUCT_TRAIN" \
      --p_extra_csv "$P_EXTRA_TRAIN" \
      --p_embed_npy "$P_EMBED_NPY" \
      --motion_train_npz "$TRAIN_MOTION_NPZ" \
      --audio_big_npz "$AUDIO_BIG_PCA_TRAIN" \
      --motion_extra_npz "$motion_behavior_train" \
      --gait_extra_npz "$gait_unit_train" \
      --output_dir "$exp_dir" \
      --model_arch "$arch" \
      --audio_features "$audio_features" \
      --official_video_features "" \
      --use_gait "$use_gait" \
      --folds "$FOLDS" \
      --seeds "$SEEDS" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --batch_size "$BATCH_SIZE" \
      --hidden_dim "$HIDDEN_DIM" \
      --p_embed_bottleneck "$P_EMBED_BOTTLENECK" \
      --dropout "$DROPOUT" \
      --modality_dropout "$MODALITY_DROPOUT" \
      --reg_weight "$REG_WEIGHT" \
      --ccc_weight "$CCC_WEIGHT" \
      --binary_weight "$BINARY_WEIGHT" \
      --consistency_weight "$CONSISTENCY_WEIGHT" \
      --ordinal_weight "$ORDINAL_WEIGHT" \
      --soft_f1_weight "$SOFT_F1_WEIGHT" \
      --kappa_weight "$KAPPA_WEIGHT" \
      --class_weight_power "$CLASS_WEIGHT_POWER" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --device "$DEVICE"

    python -u "$CODE" predict \
      --test_data_root "$TEST_ROOT" \
      --test_split_csv "$TEST_ID_CSV" \
      --p_struct_test_csv "$P_STRUCT_TEST" \
      --p_extra_csv "$P_EXTRA_TEST" \
      --p_embed_npy "$P_EMBED_NPY" \
      --motion_test_npz "$TEST_MOTION_NPZ" \
      --audio_big_npz "$AUDIO_BIG_PCA_TEST" \
      --motion_extra_npz "$motion_behavior_test" \
      --gait_extra_npz "$gait_unit_test" \
      --checkpoint_dir "$exp_dir/checkpoints" \
      --output_dir "$exp_dir/predictions_normal" \
      --batch_size 16 \
      --device "$DEVICE"

    predict_topk "$exp_dir" "$TOPK" "best"
    predict_topk "$exp_dir" 6 "cls"
    predict_topk "$exp_dir" 8 "cls"

  ) 2>&1 | tee "$log_file"

  if [ -f "$exp_dir/predictions_normal/submission.zip" ]; then
    append_done "$exp_name" "$exp_dir"
    echo "[DONE] $exp_name" | tee -a "$STATUS_TXT"
    echo "normal:   $exp_dir/predictions_normal/submission.zip" | tee -a "$STATUS_TXT"
    echo "top${TOPK}:     $exp_dir/predictions_top${TOPK}_normal/submission.zip" | tee -a "$STATUS_TXT"
    echo "cls_top6: $exp_dir/predictions_cls_top6_normal/submission.zip" | tee -a "$STATUS_TXT"
    echo "cls_top8: $exp_dir/predictions_cls_top8_normal/submission.zip" | tee -a "$STATUS_TXT"
    run_analysis
  else
    append_fail "$exp_name" "$exp_dir"
    echo "[FAILED] $exp_name, see $log_file" | tee -a "$STATUS_TXT"
    exit 1
  fi
}

# ============================================================
# Run order
# 主配置放第一个：allA + fullVG + no_cross
# ============================================================

run_exp "v11_allA_no_cross_fullVG_5x3"  "no_cross" "allA" "fullVG"

# 第二优先：只用新 A，验证 A_big 是否独立有效
run_exp "v11_newA_no_cross_fullVG_5x3"  "no_cross" "newA" "fullVG"

# 第三、四：cross 对照
run_exp "v11_allA_cross_fullVG_5x3"     "cross"    "allA" "fullVG"
run_exp "v11_newA_cross_fullVG_5x3"     "cross"    "newA" "fullVG"

# Ablations：默认都跑 no_cross + cross
run_exp "v11_allA_no_cross_noVBeh_5x3"   "no_cross" "allA" "noVBeh"
run_exp "v11_allA_cross_noVBeh_5x3"      "cross"    "allA" "noVBeh"

run_exp "v11_allA_no_cross_noGUnit_5x3"  "no_cross" "allA" "noGUnit"
run_exp "v11_allA_cross_noGUnit_5x3"     "cross"    "allA" "noGUnit"

run_exp "v11_allA_no_cross_rawGOnly_5x3" "no_cross" "allA" "rawGOnly"
run_exp "v11_allA_cross_rawGOnly_5x3"    "cross"    "allA" "rawGOnly"

run_exp "v11_allA_no_cross_GUnitOnly_5x3" "no_cross" "allA" "GUnitOnly"
run_exp "v11_allA_cross_GUnitOnly_5x3"    "cross"    "allA" "GUnitOnly"

echo
echo "========== ALL V11 5x3 SUITE DONE =========="
echo "status: $STATUS_TXT"
run_analysis
