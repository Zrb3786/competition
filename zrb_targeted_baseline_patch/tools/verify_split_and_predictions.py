#!/usr/bin/env python3
"""Verify that prediction CSV files are identity-aligned with split_labels_train.csv.

Usage:
  python tools/verify_split_and_predictions.py \
    --split_csv MPDD-AVG2026/MPDD-AVG2026-trainval/Young/split_labels_train.csv \
    --log_dir logs/Track2/A-V-G+P/ternary/track2_avgp_targeted_baseline_cv5x3/<timestamp>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score


def load_split(split_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    df = df[df["split"].astype(str).str.lower().eq("train")].copy()
    df["ID"] = df["ID"].astype(int)
    df["label3"] = df["label3"].astype(int)
    df["PHQ-9"] = df["PHQ-9"].astype(float)
    return df


def load_predictions(log_dir: str | Path) -> pd.DataFrame:
    log_dir = Path(log_dir)
    files = sorted(log_dir.glob("predictions_rep*_fold*.csv"))
    if not files and (log_dir / "oof_predictions.csv").exists():
        files = [log_dir / "oof_predictions.csv"]
    if not files:
        raise FileNotFoundError(f"No predictions_rep*_fold*.csv found in {log_dir}")
    frames = []
    for p in files:
        df = pd.read_csv(p)
        df["source_file"] = p.name
        frames.append(df)
    pred = pd.concat(frames, ignore_index=True)
    pred["ID"] = pred["ID"].astype(int)
    return pred


def verify(split_csv: str | Path, log_dir: str | Path, pred_col: str = "pred_main", out_json: str | None = None) -> dict[str, Any]:
    split = load_split(split_csv)
    pred = load_predictions(log_dir)
    need = {"ID", "label3", "PHQ-9"}
    missing = need - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction files missing columns: {sorted(missing)}")
    m = pred.merge(split[["ID", "label3", "PHQ-9"]], on="ID", how="left", suffixes=("_predfile", "_splitcsv"))
    mismatch = m[
        m["label3_splitcsv"].isna()
        | (m["label3_predfile"].astype(float) != m["label3_splitcsv"].astype(float))
        | (np.abs(m["PHQ-9_predfile"].astype(float) - m["PHQ-9_splitcsv"].astype(float)) > 1e-3)
    ].copy()

    report: dict[str, Any] = {
        "split_rows_train": int(len(split)),
        "prediction_rows": int(len(pred)),
        "unique_prediction_ids": int(pred["ID"].nunique()),
        "mismatch_rows": int(len(mismatch)),
        "missing_ids_from_split_in_predictions": sorted(set(split["ID"].astype(int)) - set(pred["ID"].astype(int))),
        "extra_prediction_ids_not_in_split": sorted(set(pred["ID"].astype(int)) - set(split["ID"].astype(int))),
        "prediction_label_counts": {str(k): int(v) for k, v in pred["label3"].astype(int).value_counts().sort_index().items()},
    }
    if len(mismatch):
        cols = ["ID", "source_file", "repeat", "fold", "label3_predfile", "label3_splitcsv", "PHQ-9_predfile", "PHQ-9_splitcsv"]
        cols = [c for c in cols if c in mismatch.columns]
        report["first_mismatches"] = mismatch[cols].head(50).to_dict(orient="records")
    if pred_col in pred.columns:
        y = pred["label3"].astype(int).to_numpy()
        yp = pred[pred_col].astype(int).to_numpy()
        report[f"metrics_{pred_col}"] = {
            "macro_f1": float(f1_score(y, yp, average="macro", labels=[0,1,2], zero_division=0)),
            "acc": float(accuracy_score(y, yp)),
            "kappa": float(cohen_kappa_score(y, yp, labels=[0,1,2])),
            "confusion_matrix": confusion_matrix(y, yp, labels=[0,1,2]).astype(int).tolist(),
            "classification_report": classification_report(y, yp, labels=[0,1,2], zero_division=0, output_dict=True),
        }
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--pred_col", default="pred_main")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()
    report = verify(args.split_csv, args.log_dir, args.pred_col, args.out_json or None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["mismatch_rows"] != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
