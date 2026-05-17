#!/usr/bin/env bash
set -euo pipefail
DEVICE=${DEVICE:-cpu} \
EPOCHS=${EPOCHS:-1} \
CV_REPEATS=${CV_REPEATS:-1} \
MAX_FOLDS=${MAX_FOLDS:-1} \
MIN_SELECT_EPOCH=${MIN_SELECT_EPOCH:-1} \
EXPERIMENT_NAME=${EXPERIMENT_NAME:-track2_avgp_targeted_baseline_smoke} \
bash scripts/Track2/A-V-G+P/run_targeted_baseline_cv.sh
