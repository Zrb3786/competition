#!/usr/bin/env bash
set -euo pipefail
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$(pwd)}"
cd "$REPO_DIR"

if [[ ! -f dataset.py || ! -f models/torchcat_baseline.py ]]; then
  echo "ERROR: run this from the MPDD-AVG-2026 repo root, or pass repo path as first argument." >&2
  exit 1
fi

mkdir -p scripts/Track2/A-V-G+P tools
cp "$PATCH_DIR/train_targeted_baseline_cv.py" ./train_targeted_baseline_cv.py
cp "$PATCH_DIR/tools/verify_split_and_predictions.py" ./tools/verify_split_and_predictions.py
cp "$PATCH_DIR/tools/inspect_split_csv.py" ./tools/inspect_split_csv.py
cp "$PATCH_DIR/scripts/Track2/A-V-G+P/run_targeted_baseline_cv.sh" ./scripts/Track2/A-V-G+P/run_targeted_baseline_cv.sh
cp "$PATCH_DIR/scripts/Track2/A-V-G+P/run_targeted_baseline_smoke.sh" ./scripts/Track2/A-V-G+P/run_targeted_baseline_smoke.sh
chmod +x ./scripts/Track2/A-V-G+P/run_targeted_baseline_cv.sh ./scripts/Track2/A-V-G+P/run_targeted_baseline_smoke.sh

echo "Installed targeted baseline CV files into: $REPO_DIR"
echo "Now run: python -m py_compile train_targeted_baseline_cv.py tools/verify_split_and_predictions.py tools/inspect_split_csv.py"
