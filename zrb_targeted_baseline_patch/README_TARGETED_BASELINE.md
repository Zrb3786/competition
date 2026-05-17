# ZRB targeted baseline CV patch

This patch keeps the official MPDD-AVG-2026 `TorchcatBaseline` architecture and official `MPDDElderDataset` feature loading, but replaces the training split/loss/metric loop with a strict CV pipeline for Track2 ternary classification.

## Why this patch exists

Previous custom logs showed massive mismatch between `ID -> label3 -> PHQ-9` in prediction CSVs and `split_labels_train.csv`. The likely cause is using official split helpers that may append the test counterpart CSV. This patch reads only the physical `split_labels_train.csv` rows with `split == train` for CV and checks every dataset/batch/prediction against the CSV.

## Files

- `train_targeted_baseline_cv.py`: main CV training entry.
- `scripts/Track2/A-V-G+P/run_targeted_baseline_cv.sh`: full CV launcher.
- `scripts/Track2/A-V-G+P/run_targeted_baseline_smoke.sh`: 1-fold smoke test launcher.
- `tools/inspect_split_csv.py`: verifies label3/PHQ consistency in the train CSV.
- `tools/verify_split_and_predictions.py`: verifies prediction logs align with train CSV and prints metrics.

## Training objective

The model architecture remains official baseline:

```text
A/V/G/P features -> official TorchcatBaseline encoders/fusion -> 3-way classifier + PHQ regression head
```

The loss is targeted to imbalanced ordinal ternary classification:

```text
total_loss =
  CE(label3)
+ ORD_LOSS_WEIGHT * [BCE(P(class>=1), label3>=1) + GE10_LOSS_WEIGHT * BCE(P(class>=2), label3>=2)]
+ PHQ_LOSS_WEIGHT * SmoothL1(log1p(PHQ))
```

`P(class>=1)` and `P(class>=2)` are derived from the same official 3-way classifier logits, so no new model head is introduced.

## Default knobs

```text
CLASS_WEIGHT_MODE=sqrt
SAMPLER_MODE=sqrt
BOUNDARY_POS_WEIGHT_MODE=sqrt
ORD_LOSS_WEIGHT=0.5
GE10_LOSS_WEIGHT=1.2
PHQ_LOSS_WEIGHT=0.0
LABEL_SMOOTHING=0.05
PRED_MODE=argmax
MIN_SELECT_EPOCH=8
```

Best checkpoint selection protects both middle and severe classes:

```text
score = macro_f1 + 0.1*kappa + 0.05*recall_class1 + 0.05*recall_class2
        - 0.1 if class1/class2 recall is zero
```

## Apply

```bash
cd /path/to/MPDD-AVG-2026
unzip zrb_targeted_baseline_patch.zip
bash zrb_targeted_baseline_patch/apply_to_repo.sh
python -m py_compile train_targeted_baseline_cv.py tools/verify_split_and_predictions.py tools/inspect_split_csv.py
```

## Check the split CSV first

```bash
python tools/inspect_split_csv.py \
  --split_csv MPDD-AVG2026/MPDD-AVG2026-trainval/Young/split_labels_train.csv
```

Expected for your current train CSV:

```text
class 0: 45, PHQ 0-4
class 1: 33, PHQ 5-9
class 2: 10, PHQ 10-17
inconsistent rows: empty
```

## Smoke test

```bash
DEVICE=cpu EPOCHS=1 MAX_FOLDS=1 MIN_SELECT_EPOCH=1 \
  bash scripts/Track2/A-V-G+P/run_targeted_baseline_smoke.sh
```

## Full run

```bash
CV_REPEATS=3 EPOCHS=60 \
  bash scripts/Track2/A-V-G+P/run_targeted_baseline_cv.sh
```

## Verify the output identity alignment

Replace `<timestamp>` with the new output directory name printed at the end.

```bash
python tools/verify_split_and_predictions.py \
  --split_csv MPDD-AVG2026/MPDD-AVG2026-trainval/Young/split_labels_train.csv \
  --log_dir logs/Track2/A-V-G+P/ternary/track2_avgp_targeted_baseline_cv5x3/<timestamp> \
  --pred_col pred_main \
  --out_json verify_pred_main.json
```

This must report:

```text
mismatch_rows: 0
extra_prediction_ids_not_in_split: []
```

If mismatch is not zero, stop and send the verification JSON/logs before tuning anything.
