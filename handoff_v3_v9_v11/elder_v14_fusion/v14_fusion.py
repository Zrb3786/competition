#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v14_fusion.py

OOF-level fusion for MPDD Elder v14 expert probes.

It reads each expert's:
  - oof_predictions.csv
  - test_predictions.csv
and searches small, safe probability-level fusions.

Design:
  - binary is always derived from ternary: binary = (ternary > 0)
  - PHQ is calibrated as expected class PHQ plus optional blended expert PHQ
  - candidates are filtered by test distribution safety
  - no retraining is required

Recommended after running all v14_v12loader experts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error, mean_squared_error


DEFAULT_EXPERTS = ["p", "audio_controlled", "gait", "video", "audio_big"]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def ccc_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    mt, mp = np.mean(y_true), np.mean(y_pred)
    vt, vp = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mt) * (y_pred - mp))
    den = vt + vp + (mt - mp) ** 2
    if den <= 1e-12:
        return 0.0
    return float(2.0 * cov / den)


def metric_dict(y2, pred2, y3, pred3, phq=None, phq_pred=None) -> Dict[str, float]:
    out = {
        "binary_acc": float(accuracy_score(y2, pred2)),
        "binary_macro_f1": float(f1_score(y2, pred2, average="macro", zero_division=0)),
        "binary_kappa": float(cohen_kappa_score(y2, pred2)),
        "ternary_acc": float(accuracy_score(y3, pred3)),
        "ternary_macro_f1": float(f1_score(y3, pred3, average="macro", zero_division=0)),
        "ternary_kappa": float(cohen_kappa_score(y3, pred3)),
        "inconsistent": float(np.sum((np.asarray(pred2) == 0) != (np.asarray(pred3) == 0))),
    }
    if phq is not None and phq_pred is not None:
        out["phq_ccc"] = ccc_np(phq, phq_pred)
        out["phq_mae"] = float(mean_absolute_error(phq, phq_pred))
        out["phq_rmse"] = float(math.sqrt(mean_squared_error(phq, phq_pred)))
    else:
        out["phq_ccc"] = 0.0
        out["phq_mae"] = 0.0
        out["phq_rmse"] = 0.0
    return out


def overall_score(m: Dict[str, float], mode: str = "balanced") -> float:
    if mode == "cls":
        return (
            0.30 * m["binary_macro_f1"]
            + 0.30 * m["ternary_macro_f1"]
            + 0.20 * m["binary_kappa"]
            + 0.20 * m["ternary_kappa"]
        )
    return (
        0.20 * m["binary_macro_f1"]
        + 0.25 * m["ternary_macro_f1"]
        + 0.20 * m["binary_kappa"]
        + 0.25 * m["ternary_kappa"]
        + 0.10 * max(-1.0, min(1.0, m.get("phq_ccc", 0.0)))
    )


def value_counts_dict(x: np.ndarray) -> Dict[str, int]:
    s = pd.Series(np.asarray(x, dtype=int)).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in s.items()}


def distribution_report(ids: np.ndarray, pred3: np.ndarray) -> Dict[str, Any]:
    pred3 = np.asarray(pred3, dtype=int)
    pred2 = (pred3 > 0).astype(int)
    return {
        "n": int(len(pred3)),
        "binary": value_counts_dict(pred2),
        "ternary": value_counts_dict(pred3),
        "severe": [int(i) for i in ids[pred3 == 2]],
        "positive": [int(i) for i in ids[pred3 > 0]],
        "inconsistent": 0,
    }


def is_safe_distribution(dist: Dict[str, Any], min_positive: int, max_positive: int, min_severe: int, max_severe: int, min_normal: int) -> bool:
    bin_counts = {int(k): int(v) for k, v in dist["binary"].items()}
    ter_counts = {int(k): int(v) for k, v in dist["ternary"].items()}
    pos = bin_counts.get(1, 0)
    norm = ter_counts.get(0, 0)
    sev = ter_counts.get(2, 0)
    return (
        dist.get("inconsistent", 0) == 0
        and min_positive <= pos <= max_positive
        and min_severe <= sev <= max_severe
        and norm >= min_normal
    )


def normalize_id_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" not in df.columns:
        if "ID" in df.columns:
            df = df.rename(columns={"ID": "id"})
        else:
            df = df.rename(columns={df.columns[0]: "id"})
    df["id"] = df["id"].astype(int)
    return df


def find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def extract_prob3(df: pd.DataFrame) -> np.ndarray:
    cols_candidates = [
        ["prob3_0", "prob3_1", "prob3_2"],
        ["p3_0", "p3_1", "p3_2"],
        ["ternary_prob_0", "ternary_prob_1", "ternary_prob_2"],
        ["class0", "class1", "class2"],
    ]
    for cols in cols_candidates:
        if all(c in df.columns for c in cols):
            p = df[cols].to_numpy(dtype=np.float64)
            p = np.clip(p, 1e-8, None)
            p /= p.sum(axis=1, keepdims=True)
            return p

    pred_col = find_col(df, ["pred3", "ternary_pred", "label3_pred"])
    if pred_col is None:
        raise KeyError(f"Cannot find ternary probabilities or pred3 columns. columns={list(df.columns)}")
    pred = df[pred_col].to_numpy(dtype=int)
    p = np.zeros((len(pred), 3), dtype=np.float64)
    p[np.arange(len(pred)), np.clip(pred, 0, 2)] = 1.0
    return p


def extract_prob2(df: pd.DataFrame, prob3: Optional[np.ndarray] = None) -> np.ndarray:
    for cols in [["prob2_0", "prob2_1"], ["p2_0", "p2_1"], ["binary_prob_0", "binary_prob_1"]]:
        if all(c in df.columns for c in cols):
            p = df[cols].to_numpy(dtype=np.float64)
            p = np.clip(p, 1e-8, None)
            p /= p.sum(axis=1, keepdims=True)
            return p
    if prob3 is not None:
        pos = prob3[:, 1] + prob3[:, 2]
        return np.stack([1.0 - pos, pos], axis=1)
    pred_col = find_col(df, ["pred2", "binary_pred", "label2_pred"])
    if pred_col is None:
        raise KeyError(f"Cannot find binary probabilities or pred2 columns. columns={list(df.columns)}")
    pred = df[pred_col].to_numpy(dtype=int)
    p = np.zeros((len(pred), 2), dtype=np.float64)
    p[np.arange(len(pred)), np.clip(pred, 0, 1)] = 1.0
    return p


def extract_phq_pred(df: pd.DataFrame) -> Optional[np.ndarray]:
    col = find_col(df, ["phq_pred", "PHQ_pred", "phq9_pred", "pred_phq"])
    if col is None:
        return None
    return df[col].to_numpy(dtype=np.float64)


def extract_labels(oof: pd.DataFrame, train_split_csv: Optional[Path]) -> pd.DataFrame:
    oof = normalize_id_col(oof)
    y2_col = find_col(oof, ["y2", "label2", "binary_true", "binary_label"])
    y3_col = find_col(oof, ["y3", "label3", "ternary_true", "ternary_label"])
    phq_col = find_col(oof, ["phq", "PHQ-9", "phq9", "PHQ9"])

    if y2_col and y3_col and phq_col:
        return oof[["id", y2_col, y3_col, phq_col]].rename(columns={y2_col: "y2", y3_col: "y3", phq_col: "phq"})

    if train_split_csv is None or not train_split_csv.exists():
        raise ValueError("OOF has no labels and --train_split_csv is not provided or missing.")

    lab = pd.read_csv(train_split_csv)
    lab = normalize_id_col(lab)
    if "label2" not in lab.columns:
        if "PHQ-9" in lab.columns:
            lab["label2"] = (lab["PHQ-9"].astype(float) >= 5).astype(int)
        elif "phq" in lab.columns:
            lab["label2"] = (lab["phq"].astype(float) >= 5).astype(int)
        else:
            raise ValueError("Cannot infer label2 from train split.")
    if "label3" not in lab.columns:
        raise ValueError("train split missing label3")
    phq_name = "PHQ-9" if "PHQ-9" in lab.columns else ("phq" if "phq" in lab.columns else None)
    if phq_name is None:
        lab["phq"] = 0.0
        phq_name = "phq"
    return lab[["id", "label2", "label3", phq_name]].rename(columns={"label2": "y2", "label3": "y3", phq_name: "phq"})


@dataclass
class ExpertPred:
    name: str
    oof: pd.DataFrame
    test: pd.DataFrame
    oof_prob3: np.ndarray
    test_prob3: np.ndarray
    oof_prob2: np.ndarray
    test_prob2: np.ndarray
    oof_phq: Optional[np.ndarray]
    test_phq: Optional[np.ndarray]


def find_expert_dir(expert_root: Path, expert: str, explicit: Dict[str, Path]) -> Path:
    if expert in explicit:
        p = explicit[expert]
        if not p.exists():
            raise FileNotFoundError(f"explicit expert dir for {expert} not found: {p}")
        return p

    patterns = [
        f"*{expert}*5x1*",
        f"*{expert}*",
    ]
    hits = []
    for pat in patterns:
        hits.extend([p for p in expert_root.glob(pat) if p.is_dir() and (p / "oof_predictions.csv").exists()])
    # prefer exact suffix and v14v12
    def score(p: Path):
        n = p.name
        return (
            0 if n == f"v14v12_{expert}_5x1" else
            1 if expert in n and "5x1" in n else
            2
        )
    hits = sorted(set(hits), key=score)
    if not hits:
        raise FileNotFoundError(f"Cannot find expert dir for {expert} under {expert_root}. Expected */oof_predictions.csv")
    return hits[0]


def find_test_predictions(run_dir: Path) -> Path:
    for rel in ["test_predictions.csv", "predictions_normal/test_predictions.csv"]:
        p = run_dir / rel
        if p.exists():
            return p
    # fallback: any test_predictions.csv below run dir
    hits = list(run_dir.glob("**/test_predictions.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Cannot find test_predictions.csv under {run_dir}")


def load_experts(expert_root: Path, experts: List[str], explicit: Dict[str, Path], train_split_csv: Optional[Path]) -> Tuple[List[ExpertPred], pd.DataFrame]:
    loaded: List[ExpertPred] = []
    labels: Optional[pd.DataFrame] = None

    for ex in experts:
        run_dir = find_expert_dir(expert_root, ex, explicit)
        oof_path = run_dir / "oof_predictions.csv"
        test_path = find_test_predictions(run_dir)
        oof = normalize_id_col(pd.read_csv(oof_path)).sort_values("id").reset_index(drop=True)
        test = normalize_id_col(pd.read_csv(test_path)).sort_values("id").reset_index(drop=True)

        if labels is None:
            labels = extract_labels(oof, train_split_csv).sort_values("id").reset_index(drop=True)
        else:
            # Align to labels
            pass

        oof = labels[["id"]].merge(oof, on="id", how="left")
        if oof.isna().any().any():
            raise ValueError(f"OOF expert {ex} missing ids after align.")

        oof_prob3 = extract_prob3(oof)
        test_prob3 = extract_prob3(test)
        oof_prob2 = extract_prob2(oof, oof_prob3)
        test_prob2 = extract_prob2(test, test_prob3)
        oof_phq = extract_phq_pred(oof)
        test_phq = extract_phq_pred(test)

        loaded.append(ExpertPred(ex, oof, test, oof_prob3, test_prob3, oof_prob2, test_prob2, oof_phq, test_phq))
        print(f"[LOAD] {ex}: dir={run_dir} oof={len(oof)} test={len(test)}")

    assert labels is not None
    return loaded, labels


def weights_grid(experts: List[str], max_candidates: int = 5000) -> List[np.ndarray]:
    # Hand-tuned candidates plus Dirichlet samples.
    n = len(experts)
    rng = np.random.default_rng(20260527)
    candidates = []

    def add(vals: Dict[str, float]):
        w = np.array([vals.get(e, 0.0) for e in experts], dtype=np.float64)
        if w.sum() <= 0:
            return
        w /= w.sum()
        candidates.append(w)

    # Single experts and useful manual fusions.
    for e in experts:
        add({e: 1.0})

    add({"p": 0.55, "audio_controlled": 0.25, "gait": 0.10, "video": 0.05, "audio_big": 0.05})
    add({"p": 0.50, "audio_controlled": 0.30, "gait": 0.10, "video": 0.10})
    add({"p": 0.45, "audio_controlled": 0.30, "gait": 0.15, "video": 0.10})
    add({"p": 0.60, "audio_controlled": 0.20, "gait": 0.10, "video": 0.10})
    add({"p": 0.50, "audio_controlled": 0.20, "gait": 0.20, "audio_big": 0.10})
    add({"p": 0.40, "audio_controlled": 0.35, "gait": 0.15, "video": 0.10})
    add({"p": 0.35, "audio_controlled": 0.40, "gait": 0.15, "video": 0.10})

    # Random convex combinations biased toward p/audio_controlled/gait.
    alpha = np.ones(n) * 0.6
    for i, e in enumerate(experts):
        if e == "p":
            alpha[i] = 4.0
        elif e == "audio_controlled":
            alpha[i] = 2.2
        elif e == "gait":
            alpha[i] = 1.2
        elif e == "video":
            alpha[i] = 1.0
        elif e == "audio_big":
            alpha[i] = 0.8
    for _ in range(max(0, max_candidates - len(candidates))):
        candidates.append(rng.dirichlet(alpha))

    # Deduplicate rounded.
    seen = set()
    out = []
    for w in candidates:
        key = tuple(np.round(w, 3))
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


def fuse_probs(experts: List[ExpertPred], weights: np.ndarray, split: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    p3 = None
    p2 = None
    phq = None
    phq_weight = 0.0
    for ex, w in zip(experts, weights):
        if split == "oof":
            a3, a2, aphq = ex.oof_prob3, ex.oof_prob2, ex.oof_phq
        else:
            a3, a2, aphq = ex.test_prob3, ex.test_prob2, ex.test_phq
        p3 = w * a3 if p3 is None else p3 + w * a3
        p2 = w * a2 if p2 is None else p2 + w * a2
        if aphq is not None:
            phq = w * aphq if phq is None else phq + w * aphq
            phq_weight += w
    if phq is not None and phq_weight > 1e-8:
        phq = phq / phq_weight
    return p3, p2, phq


def pred_from_argmax(p3: np.ndarray) -> np.ndarray:
    return np.argmax(p3, axis=1).astype(int)


def pred_from_thresholds(p3: np.ndarray, t0: float, t1: float) -> np.ndarray:
    sev = p3[:, 1] + 2.0 * p3[:, 2]
    pred = np.zeros(len(sev), dtype=int)
    pred[sev > t0] = 1
    pred[sev > t1] = 2
    return pred


def phq_from_class_probs(p3: np.ndarray, class_means: np.ndarray, expert_phq: Optional[np.ndarray], alpha_expert: float) -> np.ndarray:
    base = p3 @ class_means
    if expert_phq is None or alpha_expert <= 0:
        return base
    return (1.0 - alpha_expert) * base + alpha_expert * expert_phq


def save_candidate(out_dir: Path, ids: np.ndarray, pred3: np.ndarray, prob3: np.ndarray, phq_pred: np.ndarray, meta: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    pred3 = np.asarray(pred3, dtype=int)
    pred2 = (pred3 > 0).astype(int)

    binary = pd.DataFrame({"id": ids.astype(int), "binary_pred": pred2})
    ternary = pd.DataFrame({"id": ids.astype(int), "ternary_pred": pred3})
    detail = pd.DataFrame({
        "id": ids.astype(int),
        "prob3_0": prob3[:, 0],
        "prob3_1": prob3[:, 1],
        "prob3_2": prob3[:, 2],
        "pred3": pred3,
        "pred2": pred2,
        "phq_pred": phq_pred,
    })

    binary.to_csv(out_dir / "binary.csv", index=False)
    ternary.to_csv(out_dir / "ternary.csv", index=False)
    detail.to_csv(out_dir / "test_predictions_fusion.csv", index=False)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "binary.csv", "binary.csv")
        z.write(out_dir / "ternary.csv", "ternary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert_root", default="outputs/elder_v14_v12loader")
    ap.add_argument("--output_dir", default="outputs/elder_v14_fusion/v14_fusion_v1")
    ap.add_argument("--experts", default=",".join(DEFAULT_EXPERTS))
    ap.add_argument("--expert_dir", action="append", default=[], help="Optional mapping expert=/path/to/run_dir")
    ap.add_argument("--train_split_csv", default="")
    ap.add_argument("--max_weight_candidates", type=int, default=3500)
    ap.add_argument("--save_top_k", type=int, default=20)
    ap.add_argument("--score_mode", default="balanced", choices=["balanced", "cls"])

    # Safety distribution constraints.
    ap.add_argument("--min_positive", type=int, default=10)
    ap.add_argument("--max_positive", type=int, default=13)
    ap.add_argument("--min_severe", type=int, default=2)
    ap.add_argument("--max_severe", type=int, default=4)
    ap.add_argument("--min_normal", type=int, default=10)

    # Optional threshold search.
    ap.add_argument("--use_threshold_search", action="store_true", default=True)
    ap.add_argument("--no_threshold_search", action="store_false", dest="use_threshold_search")
    args = ap.parse_args()

    expert_root = Path(args.expert_root)
    output_dir = ensure_dir(Path(args.output_dir))
    experts_names = [e.strip() for e in args.experts.split(",") if e.strip()]
    explicit = {}
    for item in args.expert_dir:
        if "=" not in item:
            raise ValueError(f"--expert_dir must be expert=path, got {item}")
        k, v = item.split("=", 1)
        explicit[k.strip()] = Path(v.strip())

    train_split_csv = Path(args.train_split_csv) if args.train_split_csv else None
    experts, labels = load_experts(expert_root, experts_names, explicit, train_split_csv)

    ids_oof = labels["id"].to_numpy(dtype=int)
    y2 = labels["y2"].to_numpy(dtype=int)
    y3 = labels["y3"].to_numpy(dtype=int)
    phq = labels["phq"].to_numpy(dtype=float)
    class_means = np.array([float(phq[y3 == k].mean()) if np.any(y3 == k) else float(phq.mean()) for k in range(3)], dtype=np.float64)
    # Enforce monotonicity.
    class_means = np.maximum.accumulate(class_means)
    print("[INFO] class_means", class_means.tolist())

    test_ids = experts[0].test["id"].to_numpy(dtype=int)
    # Align test ids across experts.
    for ex in experts[1:]:
        if not np.array_equal(test_ids, ex.test["id"].to_numpy(dtype=int)):
            raise ValueError(f"Test IDs mismatch for expert {ex.name}")

    weight_list = weights_grid(experts_names, args.max_weight_candidates)
    threshold_pairs = [("argmax", None, None)]
    if args.use_threshold_search:
        for t0 in np.arange(0.45, 1.11, 0.05):
            for t1 in np.arange(max(t0 + 0.20, 1.00), 1.91, 0.05):
                threshold_pairs.append(("thr", float(round(t0, 3)), float(round(t1, 3))))

    rows = []
    candidates = []
    for wi, w in enumerate(weight_list):
        oof_p3, oof_p2, oof_expert_phq = fuse_probs(experts, w, "oof")
        test_p3, test_p2, test_expert_phq = fuse_probs(experts, w, "test")

        for phq_alpha in [0.0, 0.25, 0.5]:
            oof_phq_pred = phq_from_class_probs(oof_p3, class_means, oof_expert_phq, phq_alpha)
            test_phq_pred = phq_from_class_probs(test_p3, class_means, test_expert_phq, phq_alpha)

            for pred_mode, t0, t1 in threshold_pairs:
                if pred_mode == "argmax":
                    oof_pred3 = pred_from_argmax(oof_p3)
                    test_pred3 = pred_from_argmax(test_p3)
                else:
                    oof_pred3 = pred_from_thresholds(oof_p3, t0, t1)
                    test_pred3 = pred_from_thresholds(test_p3, t0, t1)

                oof_pred2 = (oof_pred3 > 0).astype(int)
                metrics = metric_dict(y2, oof_pred2, y3, oof_pred3, phq, oof_phq_pred)
                score = overall_score(metrics, args.score_mode)
                dist = distribution_report(test_ids, test_pred3)
                safe = is_safe_distribution(
                    dist,
                    args.min_positive,
                    args.max_positive,
                    args.min_severe,
                    args.max_severe,
                    args.min_normal,
                )

                row = {
                    "score": score,
                    "safe": bool(safe),
                    "pred_mode": pred_mode,
                    "t0": t0 if t0 is not None else "",
                    "t1": t1 if t1 is not None else "",
                    "phq_alpha": phq_alpha,
                    "test_binary": json.dumps(dist["binary"], ensure_ascii=False),
                    "test_ternary": json.dumps(dist["ternary"], ensure_ascii=False),
                    "test_severe": json.dumps(dist["severe"], ensure_ascii=False),
                    **metrics,
                }
                for name, val in zip(experts_names, w):
                    row[f"w_{name}"] = float(val)
                rows.append(row)

                if safe:
                    candidates.append((score, row, w.copy(), oof_p3, test_p3, test_pred3.copy(), test_phq_pred.copy(), dist))

    summary = pd.DataFrame(rows).sort_values(["safe", "score"], ascending=[False, False]).reset_index(drop=True)
    summary.to_csv(output_dir / "fusion_search_summary.csv", index=False)
    print("[OK] saved", output_dir / "fusion_search_summary.csv")

    if not candidates:
        print("[WARN] no safe candidates found under current constraints. Saving top unsafe candidates separately.")
        # Save top 10 unsafe for analysis.
        unsafe_summary = pd.DataFrame(rows).sort_values("score", ascending=False).head(args.save_top_k)
        unsafe_summary.to_csv(output_dir / "top_unsafe_candidates.csv", index=False)
        print(unsafe_summary.head(10).to_string(index=False))
        return

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    saved = []
    for rank, (score, row, w, oof_p3, test_p3, test_pred3, test_phq_pred, dist) in enumerate(candidates[: args.save_top_k], start=1):
        cname = f"candidate_{rank:02d}_score_{score:.4f}".replace(".", "p")
        cdir = output_dir / cname
        meta = {
            "rank": rank,
            "score": float(score),
            "row": row,
            "weights": {name: float(val) for name, val in zip(experts_names, w)},
            "distribution": dist,
            "class_means": class_means.tolist(),
        }
        save_candidate(cdir, test_ids, test_pred3, test_p3, test_phq_pred, meta)
        saved.append({
            "candidate": cname,
            "path": str(cdir / "submission.zip"),
            "score": float(score),
            "binary": json.dumps(dist["binary"], ensure_ascii=False),
            "ternary": json.dumps(dist["ternary"], ensure_ascii=False),
            "severe": json.dumps(dist["severe"], ensure_ascii=False),
            **{f"w_{name}": float(val) for name, val in zip(experts_names, w)},
            "pred_mode": row["pred_mode"],
            "t0": row["t0"],
            "t1": row["t1"],
            "phq_alpha": row["phq_alpha"],
            "ternary_macro_f1": row["ternary_macro_f1"],
            "ternary_kappa": row["ternary_kappa"],
            "binary_macro_f1": row["binary_macro_f1"],
            "binary_kappa": row["binary_kappa"],
            "phq_ccc": row["phq_ccc"],
        })

    saved_df = pd.DataFrame(saved)
    saved_df.to_csv(output_dir / "saved_candidates.csv", index=False)
    print("[OK] saved candidates:")
    print(saved_df.to_string(index=False))
    print("[DONE]", output_dir)


if __name__ == "__main__":
    main()
