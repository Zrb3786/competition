#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/configs/elder_paths.env"
CODE="$CODE_DIR/mpdd_elder_v10_audio_pprior.py"
EXP_DIR="$OUT_DIR/v10_C_audio_pprior_motion_5x3"

python -u "$CODE" train \
  --train_data_root "$TRAIN_ROOT" \
  --train_split_csv "$TRAIN_SPLIT_CSV" \
  --p_struct_train_csv "$FEATURE_DIR/elder_descriptions_struct_train.csv" \
  --p_embed_npy "$P_EMBED_NPY" \
  --motion_train_npz "$TRAIN_MOTION_NPZ" \
  --audio_big_npz "$AUDIO_BIG_TRAIN_NPZ" \
  --p_extra_csv "$P_EXTRA_TRAIN_CSV" \
  --output_dir "$EXP_DIR" \
  --model_arch v10_audio_p_prior \
  --audio_features wav2vec,opensmile \
  --official_video_features "" \
  --use_gait 1 \
  --folds 5 \
  --seeds 42,43,44 \
  --epochs 80 \
  --patience 14 \
  --batch_size 8 \
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
  --lr 5e-4 \
  --weight_decay 3e-3 \
  --device cuda

python -u "$CODE" predict \
  --test_data_root "$TEST_ROOT" \
  --test_split_csv "$TEST_ID_CSV" \
  --p_struct_test_csv "$FEATURE_DIR/elder_descriptions_struct_test.csv" \
  --p_embed_npy "$P_EMBED_NPY" \
  --motion_test_npz "$TEST_MOTION_NPZ" \
  --audio_big_npz "$AUDIO_BIG_TEST_NPZ" \
  --p_extra_csv "$P_EXTRA_TEST_CSV" \
  --checkpoint_dir "$EXP_DIR/checkpoints" \
  --output_dir "$EXP_DIR/predictions_normal" \
  --batch_size 16 \
  --device cuda

echo "[DONE] $EXP_DIR/predictions_normal/submission.zip"
