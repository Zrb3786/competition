#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=${ROOT_DIR:-/remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite}
PROJ_DIR=${PROJ_DIR:-$ROOT_DIR/elder_v14_v12loader}
CFG=${1:-$PROJ_DIR/configs/elder_v14_v12loader_paths.env}
cd "$ROOT_DIR"
source "$PROJ_DIR/scripts/_common.sh" "$CFG"
EXPERT_LIST=${EXPERT_LIST:-audio_big,audio_official,audio,audio_controlled,video,gait,p}
FAIL_FAST=${FAIL_FAST:-0}
LOG_DIR=${LOG_DIR:-$ROOT_DIR/logs_v14_v12loader_train}
mkdir -p "$LOG_DIR"
IFS=',' read -ra EXPERTS <<< "$EXPERT_LIST"
OK=(); FAIL=()
echo "[V14 V12LOADER TRAIN 5x1] CFG=$CFG EXPERT_LIST=$EXPERT_LIST CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
for EXPERT in "${EXPERTS[@]}"; do
  EXPERT="$(echo "$EXPERT" | xargs)"
  [[ -z "$EXPERT" ]] && continue
  RUN_NAME="${RUN_NAME_PREFIX:-v14v12}_${EXPERT}_5x1"
  OUT="$OUT_BASE/$RUN_NAME"
  LOG="$LOG_DIR/$RUN_NAME.log"
  echo "================ TRAIN $EXPERT ================"
  set +e
  python -u "$PROJ_DIR/mpdd_elder_v14_v12loader.py" train \
    "${COMMON_ARGS[@]}" \
    --expert "$EXPERT" \
    --output_dir "$OUT" \
    --device cuda \
    --folds "${FOLDS:-5}" \
    --seed "${SEED:-42}" \
    --epochs "${EPOCHS:-80}" \
    --patience "${PATIENCE:-15}" \
    --batch_size "${BS:-8}" \
    --hidden_dim "${HIDDEN:-96}" \
    --dropout "${DROPOUT:-0.35}" \
    --lr "${LR:-1e-4}" \
    --weight_decay "${WD:-5e-4}" \
    --binary_weight "${BINW:-0.6}" \
    --soft_f1_weight "${F1W:-0.15}" \
    --reg_weight "${PHQW:-0.20}" \
    --ccc_weight "${CCCW:-0.10}" \
    --phq_resid_scale "${PHQ_SCALE:-2.5}" \
    --label_smoothing "${LS:-0.03}" \
    --class_weight_power "${CWP:-0.5}" \
    --num_workers 0 \
    2>&1 | tee "$LOG"
  STATUS=${PIPESTATUS[0]}
  set -e
  if [[ "$STATUS" -eq 0 ]]; then
    echo "[OK] $EXPERT"
    OK+=("$EXPERT")
  else
    echo "[FAIL] $EXPERT status=$STATUS log=$LOG"
    FAIL+=("$EXPERT")
    [[ "$FAIL_FAST" == "1" ]] && exit "$STATUS"
  fi
done
echo "[SUMMARY] OK=${OK[*]:-none} FAIL=${FAIL[*]:-none}"
python "$PROJ_DIR/scripts/summarize_v14_v12loader.py" --root "$OUT_BASE" || true
[[ "${#FAIL[@]}" -eq 0 ]]
