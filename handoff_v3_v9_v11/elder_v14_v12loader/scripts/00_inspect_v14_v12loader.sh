#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=${ROOT_DIR:-/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite}
PROJ_DIR=${PROJ_DIR:-$ROOT_DIR/elder_v14_v12loader}
CFG=${1:-$PROJ_DIR/configs/elder_v14_v12loader_paths.env}
cd "$ROOT_DIR"
source "$PROJ_DIR/scripts/_common.sh" "$CFG"
OUT=${OUT:-$OUT_BASE/inspect_v14_v12loader}
mkdir -p "$OUT"
python -u "$PROJ_DIR/mpdd_elder_v14_v12loader.py" inspect \
  "${COMMON_ARGS[@]}" \
  --output_dir "$OUT" \
  --expert_list "${EXPERT_LIST:-audio_big,audio_official,audio,audio_controlled,video,gait,p}" \
  ${SMOKE:+--smoke}
