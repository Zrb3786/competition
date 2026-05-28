#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/configs/elder_paths.env"
CODE="$CODE_DIR/mpdd_elder_v10_audio_pprior.py"
python -m py_compile "$CODE"

for ARCH in v10_p_prior v10_audio_p_prior; do
  echo "========== smoke $ARCH =========="
  python -u "$CODE" train \
    --train_data_root "$TRAIN_ROOT" \
    --train_split_csv "$TRAIN_SPLIT_CSV" \
    --p_struct_train_csv "$FEATURE_DIR/elder_descriptions_struct_train.csv" \
    --p_embed_npy "$P_EMBED_NPY" \
    --motion_train_npz "$TRAIN_MOTION_NPZ" \
    --audio_big_npz "$AUDIO_BIG_TRAIN_NPZ" \
    --p_extra_csv "$P_EXTRA_TRAIN_CSV" \
    --output_dir "$OUT_DIR/smoke_${ARCH}" \
    --model_arch "$ARCH" \
    --audio_features wav2vec,opensmile \
    --official_video_features "" \
    --use_gait 1 \
    --folds 2 \
    --seeds 42 \
    --epochs 1 \
    --patience 1 \
    --batch_size 4 \
    --hidden_dim 96 \
    --p_embed_bottleneck 48 \
    --dropout 0.45 \
    --modality_dropout 0.18 \
    --reg_weight 0.08 \
    --ccc_weight 0.02 \
    --binary_weight 1.20 \
    --consistency_weight 0.05 \
    --ordinal_weight 0.05 \
    --soft_f1_weight 0.25 \
    --kappa_weight 0.15 \
    --class_weight_power 0.75 \
    --device cuda
 done
