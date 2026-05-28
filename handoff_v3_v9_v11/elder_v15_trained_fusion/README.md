# Elder V15 Trained Fusion

This is a trained fusion model, not pure weight search. It learns a sample-wise
expert gate on already trained v14 expert OOF/test predictions.

Default experts:
- P expert: main evidence
- audio_controlled: distribution guard
- gait: PHQ/severity evidence
- video: normal / anti-positive evidence
- audio_big: audio complementary evidence

Run:

```bash
cd /remote-home/yangmz/zhangruibo/mpdd_elder_v3_lite
unzip elder_v15_trained_fusion.zip
CUDA_VISIBLE_DEVICES=2 bash elder_v15_trained_fusion/scripts/run_v15_trained_fusion.sh 2>&1 | tee logs_v15_trained_fusion.txt
bash elder_v15_trained_fusion/scripts/check_v15_candidates.sh
```

Outputs:

```text
outputs/elder_v15_trained_fusion/v15_gated_cls/
  baseline_expert_oof_metrics.csv
  fold_metrics.csv
  oof_predictions_v15.csv
  oof_metrics_argmax.json
  threshold_search_best.json
  test_predictions_v15_argmax.csv
  run_summary.json
  predictions_model_argmax/submission.zip
  predictions_model_threshold/submission.zip
```

Decision rule:
- Do not submit if OOF is worse than P expert and test distribution is not safer.
- Safe distribution: inconsistent=0, binary positive 10-13, severe 2-4, normal >=10.
