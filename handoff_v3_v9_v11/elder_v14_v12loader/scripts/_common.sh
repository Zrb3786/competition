#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-elder_v14_v12loader/configs/elder_v14_v12loader_paths.env}
if [[ ! -f "$CFG" ]]; then
  echo "[ERR] config not found: $CFG" >&2
  exit 1
fi
source "$CFG"

pick_existing() {
  local a="$1"
  local b="$2"
  if [[ -f "$a" ]]; then echo "$a"; else echo "$b"; fi
}

P_EXTRA_TRAIN=$(pick_existing "$P_EXTRA_TRAIN_CLEAN" "$P_EXTRA_TRAIN_FALLBACK")
P_EXTRA_TEST=$(pick_existing "$P_EXTRA_TEST_CLEAN" "$P_EXTRA_TEST_FALLBACK")

COMMON_ARGS=(
  --train_data_root "$TRAIN_ROOT"
  --test_data_root "$TEST_ROOT"
  --train_split_csv "$TRAIN_SPLIT_CSV"
  --test_split_csv "$TEST_SPLIT_CSV"
  --p_struct_train_csv "$P_STRUCT_TRAIN"
  --p_struct_test_csv "$P_STRUCT_TEST"
  --p_embed_npy "$P_EMBED_TRAIN"
  --p_embed_test_npy "$P_EMBED_TEST"
  --motion_train_npz "$MOTION_TRAIN"
  --motion_test_npz "$MOTION_TEST"
  --audio_big_train_npz "$A_BIG_TRAIN"
  --audio_big_test_npz "$A_BIG_TEST"
  --motion_extra_train_npz "$VBEH_TRAIN"
  --motion_extra_test_npz "$VBEH_TEST"
  --gait_extra_train_npz "$GUNIT_TRAIN"
  --gait_extra_test_npz "$GUNIT_TEST"
  --p_extra_train_csv "$P_EXTRA_TRAIN"
  --p_extra_test_csv "$P_EXTRA_TEST"
  --audio_features "$AUDIO_FEATURES"
  --official_video_features "$OFFICIAL_VIDEO_FEATURES"
  --use_gait "$USE_GAIT"
  --target_t 128
  --num_workers 0
)
