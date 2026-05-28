# elder_v14_fusion

This is a no-retraining OOF-level fusion tool for v14 expert probes.

It reads expert outputs from:

```text
outputs/elder_v14_v12loader/<expert_run>/
  oof_predictions.csv
  test_predictions.csv
```

and searches weighted probability fusions with safety constraints.

Default experts:

```text
p,audio_controlled,gait,video,audio_big
```

Recommended run:

```bash
cd /remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite

bash elder_v14_fusion/run_v14_fusion.sh 2>&1 | tee logs_v14_fusion.txt
```

Classification policy:

```text
binary = ternary > 0
```

so inconsistent is always zero.

Safety constraints:

```text
positive count: 10~13
severe count: 2~4
normal count >= 10
```

Outputs:

```text
outputs/elder_v14_fusion/v14_fusion_v1/
  fusion_search_summary.csv
  saved_candidates.csv
  candidate_*/binary.csv
  candidate_*/ternary.csv
  candidate_*/submission.zip
  candidate_*/meta.json
```

Try classification-only score:

```bash
SCORE_MODE=cls OUT_DIR=outputs/elder_v14_fusion/v14_fusion_cls \
bash elder_v14_fusion/run_v14_fusion.sh
```

Try stricter distribution:

```bash
MAX_POSITIVE=12 MIN_NORMAL=11 OUT_DIR=outputs/elder_v14_fusion/v14_fusion_strict \
bash elder_v14_fusion/run_v14_fusion.sh
```
