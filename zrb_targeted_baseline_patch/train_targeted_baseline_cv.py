#!/usr/bin/env python3
"""Targeted ternary CV training for MPDD-AVG-2026 Track2.

This script deliberately reuses the official baseline model architecture
(`models.torchcat_baseline.TorchcatBaseline`) and official feature dataset
(`dataset.MPDDElderDataset`), while replacing the training split/loss/metrics
pipeline with a strict, CV-oriented implementation for small imbalanced ternary
classification.

Key safety fix compared with earlier custom scripts:
- Do NOT call dataset.load_split_rows(), because the official helper may append
  the test counterpart CSV when it sees trainval paths. For CV we must use only
  the rows physically present in split_labels_train.csv and split == train.
- Every train/val batch is identity-checked against the original CSV by ID.
  A label/PHQ mismatch raises immediately.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import MPDDElderDataset, collate_batch, infer_input_dims, normalize_phq_target
from models.torchcat_baseline import TorchcatBaseline


EPS = 1e-7


@dataclass
class RowItem:
    pid: int
    label3: int
    phq: float
    split: str


# -----------------------------
# Utilities
# -----------------------------

def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else Path.cwd() / path


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("targeted_baseline_cv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_dir / "train_targeted_baseline_cv.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def read_train_rows_strict(split_csv: str | Path, require_train_split: bool = True) -> list[RowItem]:
    """Read ONLY rows physically present in split_csv.

    This intentionally does not reuse official dataset.load_split_rows(), because
    that helper may append split_labels_test.csv as a counterpart. That behavior
    is useful for official train/test loading, but wrong for train-only CV.
    """
    path = resolve_path(split_csv)
    if not path.exists():
        raise FileNotFoundError(f"split_csv not found: {path}")
    rows: list[RowItem] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"ID", "split", "label3", "PHQ-9"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"split_csv missing columns: {sorted(missing)}")
        for r in reader:
            split_name = str(r.get("split", "")).strip().lower()
            if require_train_split and split_name != "train":
                continue
            pid = int(float(r["ID"]))
            label3 = int(float(r["label3"]))
            phq = float(r["PHQ-9"])
            rows.append(RowItem(pid=pid, label3=label3, phq=phq, split=split_name))
    if not rows:
        raise RuntimeError(f"No train rows loaded from {path}")
    # Duplicate IDs are fatal because all maps are ID keyed.
    ids = [r.pid for r in rows]
    dup = [pid for pid, cnt in Counter(ids).items() if cnt > 1]
    if dup:
        raise ValueError(f"Duplicate IDs in train rows: {dup[:20]}")
    return rows


def validate_label_phq_consistency(rows: list[RowItem]) -> list[dict[str, Any]]:
    bad = []
    for r in rows:
        expected = 0 if r.phq < 5 else (1 if r.phq < 10 else 2)
        if r.label3 != expected:
            bad.append({"ID": r.pid, "label3": r.label3, "PHQ-9": r.phq, "expected_label3": expected})
    return bad


def build_maps(rows: Iterable[RowItem]) -> tuple[dict[int, int], dict[int, float], dict[int, str]]:
    label_map = {r.pid: int(r.label3) for r in rows}
    phq_map = {r.pid: float(r.phq) for r in rows}
    # Force official dataset to read features from train root only.
    source_split_map = {r.pid: "train" for r in rows}
    return label_map, phq_map, source_split_map


def class_counts(labels: Iterable[int]) -> dict[int, int]:
    c = Counter(int(x) for x in labels)
    return {i: int(c.get(i, 0)) for i in range(3)}


def make_cv_splits(rows: list[RowItem], n_splits: int, n_repeats: int, seed: int) -> list[dict[str, Any]]:
    y = np.array([r.label3 for r in rows], dtype=np.int64)
    idx = np.arange(len(rows))
    min_count = min(Counter(y).values())
    if n_splits > min_count:
        raise ValueError(f"cv_folds={n_splits} > minority class count={min_count}; reduce folds.")
    splits: list[dict[str, Any]] = []
    for rep in range(n_repeats):
        rs = int(seed + 1009 * rep)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
        for fold, (tr, va) in enumerate(skf.split(idx, y)):
            train_rows = [rows[i] for i in tr]
            val_rows = [rows[i] for i in va]
            splits.append({
                "repeat": rep,
                "fold": fold,
                "random_state": rs,
                "train_rows": train_rows,
                "val_rows": val_rows,
                "split_summary": {
                    "train_count_requested": len(train_rows),
                    "val_count_requested": len(val_rows),
                    "train_label_counts_requested": class_counts(r.label3 for r in train_rows),
                    "val_label_counts_requested": class_counts(r.label3 for r in val_rows),
                    "strategy": "strict_repeated_stratified_kfold_train_csv_only",
                    "n_splits": n_splits,
                    "n_repeats": n_repeats,
                    "repeat": rep,
                    "fold": fold,
                    "random_state": rs,
                    "all_label_counts": class_counts(r.label3 for r in rows),
                },
            })
    return splits


def dataset_ids(ds: MPDDElderDataset) -> list[int]:
    return [int(s["pid"]) for s in ds.samples]


def strict_check_dataset(
    ds: MPDDElderDataset,
    label_map: dict[int, int],
    phq_map: dict[int, float],
    name: str,
    strict_require_all: bool = False,
) -> dict[str, Any]:
    ids = dataset_ids(ds)
    requested = set(label_map.keys())
    present = set(ids)
    extra = sorted(present - requested)
    dropped = sorted(requested - present)
    if extra:
        raise AssertionError(f"{name}: dataset contains IDs not requested by fold: {extra[:20]}")
    if strict_require_all and dropped:
        raise AssertionError(f"{name}: dataset dropped requested IDs: {dropped[:50]}")
    # Direct sample metadata check before DataLoader.
    mismatches = []
    for s in ds.samples:
        pid = int(s["pid"])
        lab = int(s["label"])
        phq_log = float(s.get("phq9", normalize_phq_target(phq_map[pid])))
        phq_raw = float(np.expm1(phq_log))
        if lab != int(label_map[pid]) or abs(phq_raw - float(phq_map[pid])) > 2e-4:
            mismatches.append({
                "ID": pid,
                "sample_label": lab,
                "csv_label": int(label_map[pid]),
                "sample_phq_raw": phq_raw,
                "csv_phq": float(phq_map[pid]),
            })
    if mismatches:
        raise AssertionError(f"{name}: sample ID/label/PHQ mismatch: {mismatches[:10]}")
    return {
        "requested_count": len(requested),
        "loaded_count": len(ids),
        "dropped_count": len(dropped),
        "dropped_ids": dropped,
        "loaded_label_counts": class_counts(label_map[pid] for pid in ids),
    }


def strict_check_batch(batch: dict[str, torch.Tensor], label_map: dict[int, int], phq_map: dict[int, float], where: str) -> None:
    pids = batch["pid"].detach().cpu().numpy().astype(int).tolist()
    labels = batch["label"].detach().cpu().numpy().astype(int).tolist()
    phq_logs = batch.get("phq9")
    phqs = np.expm1(phq_logs.detach().cpu().numpy()).tolist() if phq_logs is not None else [None] * len(pids)
    bad = []
    for pid, lab, phq in zip(pids, labels, phqs):
        if pid not in label_map:
            bad.append({"ID": pid, "error": "ID not in label_map"})
            continue
        if lab != int(label_map[pid]) or (phq is not None and abs(float(phq) - float(phq_map[pid])) > 2e-3):
            bad.append({
                "ID": pid,
                "batch_label": int(lab),
                "csv_label": int(label_map[pid]),
                "batch_phq_raw": None if phq is None else float(phq),
                "csv_phq": float(phq_map[pid]),
            })
    if bad:
        raise AssertionError(f"{where}: batch identity mismatch: {bad[:10]}")


# -----------------------------
# Losses and metrics
# -----------------------------

def compute_class_weights(labels: list[int], mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.array([max(labels.count(i), 1) for i in range(3)], dtype=np.float32)
    if mode == "inverse":
        w = 1.0 / counts
    elif mode == "sqrt":
        w = 1.0 / np.sqrt(counts)
    elif mode == "effective":
        beta = 0.999
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(f"Unknown class_weight_mode={mode}")
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def compute_boundary_pos_weights(labels: list[int], mode: str, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if mode == "none":
        return None, None
    y = np.array(labels, dtype=np.int64)
    pos_ge5 = int((y >= 1).sum())
    neg_ge5 = int((y < 1).sum())
    pos_ge10 = int((y >= 2).sum())
    neg_ge10 = int((y < 2).sum())

    def make(pos: int, neg: int) -> float:
        pos = max(pos, 1)
        neg = max(neg, 1)
        ratio = neg / pos
        if mode == "inverse":
            return float(ratio)
        if mode == "sqrt":
            return float(math.sqrt(ratio))
        if mode == "effective":
            # Approximate effective positive weight via sqrt ratio; stable for tiny data.
            return float(math.sqrt(ratio))
        raise ValueError(f"Unknown boundary_pos_weight_mode={mode}")

    return (
        torch.tensor(make(pos_ge5, neg_ge5), dtype=torch.float32, device=device),
        torch.tensor(make(pos_ge10, neg_ge10), dtype=torch.float32, device=device),
    )


def make_sampler(labels: list[int], mode: str) -> WeightedRandomSampler | None:
    if mode == "none":
        return None
    counts = Counter(labels)
    weights = []
    for y in labels:
        c = max(counts[int(y)], 1)
        if mode == "inverse":
            weights.append(1.0 / c)
        elif mode == "sqrt":
            weights.append(1.0 / math.sqrt(c))
        else:
            raise ValueError(f"Unknown sampler_mode={mode}")
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def focal_cross_entropy(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None, gamma: float, label_smoothing: float) -> torch.Tensor:
    if gamma <= 0:
        return F.cross_entropy(logits, target, weight=weight, label_smoothing=label_smoothing)
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none", label_smoothing=label_smoothing)
    pt = torch.softmax(logits, dim=-1).gather(1, target.view(-1, 1)).squeeze(1).clamp_min(EPS)
    return ((1.0 - pt) ** gamma * ce).mean()


def boundary_logits_from_3way(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # logit(P(class>=1)) = log(P1+P2) - log(P0)
    log_ge5 = torch.logsumexp(logits[:, 1:3], dim=1)
    log_lt5 = logits[:, 0]
    ge5_logit = log_ge5 - log_lt5
    # logit(P(class>=2)) = log(P2) - log(P0+P1)
    log_ge10 = logits[:, 2]
    log_lt10 = torch.logsumexp(logits[:, 0:2], dim=1)
    ge10_logit = log_ge10 - log_lt10
    return ge5_logit, ge10_logit


def compute_losses(
    logits: torch.Tensor,
    phq_pred_log: torch.Tensor,
    labels: torch.Tensor,
    phq_log: torch.Tensor,
    class_weight: torch.Tensor | None,
    ge5_pos_weight: torch.Tensor | None,
    ge10_pos_weight: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    ce = focal_cross_entropy(logits, labels, class_weight, args.focal_gamma, args.label_smoothing)
    ge5_logit, ge10_logit = boundary_logits_from_3way(logits)
    target_ge5 = (labels >= 1).float()
    target_ge10 = (labels >= 2).float()
    loss_ge5 = F.binary_cross_entropy_with_logits(ge5_logit, target_ge5, pos_weight=ge5_pos_weight)
    loss_ge10 = F.binary_cross_entropy_with_logits(ge10_logit, target_ge10, pos_weight=ge10_pos_weight)
    boundary = args.ge5_loss_weight * loss_ge5 + args.ge10_loss_weight * loss_ge10
    phq_loss = F.smooth_l1_loss(phq_pred_log, phq_log)
    total = args.ce_loss_weight * ce + args.ord_loss_weight * boundary + args.phq_loss_weight * phq_loss
    return {
        "total": total,
        "ce": ce.detach(),
        "boundary": boundary.detach(),
        "ge5": loss_ge5.detach(),
        "ge10": loss_ge10.detach(),
        "phq": phq_loss.detach(),
    }


def concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mt, mp = float(y_true.mean()), float(y_pred.mean())
    vt, vp = float(y_true.var()), float(y_pred.var())
    cov = float(((y_true - mt) * (y_pred - mp)).mean())
    denom = vt + vp + (mt - mp) ** 2
    return float((2.0 * cov) / denom) if denom > 0 else 0.0


def metrics_for_predictions(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, Any]:
    labels = [0, 1, 2]
    precision, recall, f1s, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out: dict[str, Any] = {
        f"f1_{prefix}": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        f"acc_{prefix}": float(accuracy_score(y_true, y_pred)),
        f"kappa_{prefix}": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        f"cm_{prefix}": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }
    for i, cls in enumerate(labels):
        out[f"precision_{prefix}_{cls}"] = float(precision[i])
        out[f"recall_{prefix}_{cls}"] = float(recall[i])
        out[f"f1_{prefix}_{cls}"] = float(f1s[i])
        out[f"support_{prefix}_{cls}"] = int(support[i])
    return out


def compute_all_metrics(df: pd.DataFrame, main_col: str = "pred_main") -> dict[str, Any]:
    y = df["label3"].to_numpy(dtype=int)
    out: dict[str, Any] = {}
    for col, prefix in [
        (main_col, "main"),
        ("pred_argmax", "argmax"),
        ("pred_threshold", "threshold"),
        ("pred_phq_fixed", "phq_fixed"),
    ]:
        if col in df.columns:
            out.update(metrics_for_predictions(y, df[col].to_numpy(dtype=int), prefix))
    if "phq_pred" in df.columns:
        yt = df["PHQ-9"].to_numpy(dtype=float)
        yp = df["phq_pred"].to_numpy(dtype=float)
        out["ccc"] = concordance_ccc(yt, yp)
        out["rmse"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
        out["mae"] = float(np.mean(np.abs(yt - yp)))
    return out


def selection_score(metrics: dict[str, Any], epoch: int, args: argparse.Namespace) -> tuple[float, bool]:
    eligible = epoch >= args.min_select_epoch
    f1 = float(metrics.get("f1_main", 0.0))
    kappa = float(metrics.get("kappa_main", 0.0))
    rec1 = float(metrics.get("recall_main_1", 0.0))
    rec2 = float(metrics.get("recall_main_2", 0.0))
    ccc = float(metrics.get("ccc", 0.0))
    score = f1 + args.kappa_weight * kappa + args.recall1_weight * rec1 + args.recall2_weight * rec2 + args.ccc_weight * max(ccc, -1.0)
    if rec1 <= 0 and float(metrics.get("support_main_1", 0)) > 0:
        score -= args.zero_recall_penalty
    if rec2 <= 0 and float(metrics.get("support_main_2", 0)) > 0:
        score -= args.zero_recall_penalty
    if not eligible:
        score -= 1e6
    return float(score), bool(eligible)


def model_inputs_from_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ["audio", "video", "gait", "personality", "pair_mask"]
    return {k: batch[k].to(device) for k in keys if k in batch}


def predict_from_logits(logits: torch.Tensor, phq_pred_log: torch.Tensor, args: argparse.Namespace) -> dict[str, np.ndarray]:
    probs = torch.softmax(logits, dim=-1)
    p_ge5 = probs[:, 1] + probs[:, 2]
    p_ge10 = probs[:, 2]
    pred_argmax = probs.argmax(dim=-1)
    pred_threshold = torch.zeros_like(pred_argmax)
    pred_threshold = torch.where(p_ge5 >= args.pred_ge5_threshold, torch.ones_like(pred_threshold), pred_threshold)
    pred_threshold = torch.where(p_ge10 >= args.pred_ge10_threshold, torch.full_like(pred_threshold, 2), pred_threshold)
    pred_main = pred_argmax if args.pred_mode == "argmax" else pred_threshold
    phq_raw = torch.expm1(phq_pred_log).clamp_min(0.0)
    pred_phq_fixed = torch.zeros_like(pred_argmax)
    pred_phq_fixed = torch.where(phq_raw >= 5.0, torch.ones_like(pred_phq_fixed), pred_phq_fixed)
    pred_phq_fixed = torch.where(phq_raw >= 10.0, torch.full_like(pred_phq_fixed, 2), pred_phq_fixed)
    return {
        "prob0": probs[:, 0].detach().cpu().numpy(),
        "prob1": probs[:, 1].detach().cpu().numpy(),
        "prob2": probs[:, 2].detach().cpu().numpy(),
        "p_ge5": p_ge5.detach().cpu().numpy(),
        "p_ge10": p_ge10.detach().cpu().numpy(),
        "pred_argmax": pred_argmax.detach().cpu().numpy().astype(int),
        "pred_threshold": pred_threshold.detach().cpu().numpy().astype(int),
        "pred_main": pred_main.detach().cpu().numpy().astype(int),
        "phq_pred": phq_raw.detach().cpu().numpy(),
        "pred_phq_fixed": pred_phq_fixed.detach().cpu().numpy().astype(int),
    }


# -----------------------------
# Train / eval
# -----------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_map: dict[int, int],
    phq_map: dict[int, float],
    class_weight: torch.Tensor | None,
    ge5_pos_weight: torch.Tensor | None,
    ge10_pos_weight: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals = defaultdict(float)
    n = 0
    for batch in loader:
        strict_check_batch(batch, label_map, phq_map, "train")
        labels = batch["label"].to(device)
        phq_log = batch["phq9"].to(device)
        inputs = model_inputs_from_batch(batch, device)
        logits, phq_pred_log = model(**inputs)
        losses = compute_losses(logits, phq_pred_log, labels, phq_log, class_weight, ge5_pos_weight, ge10_pos_weight, args)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        bs = int(labels.numel())
        n += bs
        for k, v in losses.items():
            totals[k] += float(v.item()) * bs
    return {f"train_{k}_loss": (v / max(n, 1)) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_map: dict[int, int],
    phq_map: dict[int, float],
    args: argparse.Namespace,
    repeat: int,
    fold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        strict_check_batch(batch, label_map, phq_map, "eval")
        inputs = model_inputs_from_batch(batch, device)
        logits, phq_pred_log = model(**inputs)
        pred = predict_from_logits(logits, phq_pred_log, args)
        pids = batch["pid"].detach().cpu().numpy().astype(int)
        batch_labels = batch["label"].detach().cpu().numpy().astype(int)
        batch_phq_raw = np.expm1(batch["phq9"].detach().cpu().numpy())
        for i, pid in enumerate(pids):
            pid = int(pid)
            true_label = int(label_map[pid])
            true_phq = float(phq_map[pid])
            # Use CSV-backed truth in records. Batch truth is only used as assertion.
            if int(batch_labels[i]) != true_label or abs(float(batch_phq_raw[i]) - true_phq) > 2e-3:
                raise AssertionError(f"prediction identity mismatch for ID={pid}")
            records.append({
                "ID": pid,
                "repeat": int(repeat),
                "fold": int(fold),
                "label3": true_label,
                "PHQ-9": true_phq,
                "prob0": float(pred["prob0"][i]),
                "prob1": float(pred["prob1"][i]),
                "prob2": float(pred["prob2"][i]),
                "p_ge5": float(pred["p_ge5"][i]),
                "p_ge10": float(pred["p_ge10"][i]),
                "pred_argmax": int(pred["pred_argmax"][i]),
                "pred_threshold": int(pred["pred_threshold"][i]),
                "pred_main": int(pred["pred_main"][i]),
                "phq_pred": float(pred["phq_pred"][i]),
                "pred_phq_fixed": int(pred["pred_phq_fixed"][i]),
            })
    df = pd.DataFrame.from_records(records)
    metrics = compute_all_metrics(df, main_col="pred_main") if len(df) else {}
    return df, metrics


def build_model(train_dataset: MPDDElderDataset, args: argparse.Namespace, device: torch.device) -> TorchcatBaseline:
    dims = infer_input_dims(train_dataset)
    model = TorchcatBaseline(
        subtrack=args.subtrack,
        num_classes=3,
        is_regression=False,
        use_regression_head=True,
        audio_dim=dims.get("audio_dim", 0) or 1,
        video_dim=dims.get("video_dim", 0) or 1,
        gait_dim=dims.get("gait_dim", 0) or 1,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
    )
    return model.to(device)


def train_one_split(split: dict[str, Any], rows_all: list[RowItem], args: argparse.Namespace, base_log_dir: Path, ckpt_dir: Path, device: torch.device, logger: logging.Logger) -> dict[str, Any]:
    rep = int(split["repeat"])
    fold = int(split["fold"])
    tag = f"rep{rep:02d}_fold{fold:02d}"
    setup_seed(args.seed + 17 * rep + 131 * fold)

    train_rows: list[RowItem] = split["train_rows"]
    val_rows: list[RowItem] = split["val_rows"]
    train_label_map, train_phq_map, train_source = build_maps(train_rows)
    val_label_map, val_phq_map, val_source = build_maps(val_rows)

    train_ds = MPDDElderDataset(
        data_root=args.data_root,
        label_map=train_label_map,
        source_split_map=train_source,
        subtrack=args.subtrack,
        task="ternary",
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        personality_npy=args.personality_npy,
        phq_map=train_phq_map,
        target_t=args.target_t,
    )
    val_ds = MPDDElderDataset(
        data_root=args.data_root,
        label_map=val_label_map,
        source_split_map=val_source,
        subtrack=args.subtrack,
        task="ternary",
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        personality_npy=args.personality_npy,
        phq_map=val_phq_map,
        target_t=args.target_t,
    )
    train_check = strict_check_dataset(train_ds, train_label_map, train_phq_map, f"{tag}/train", args.strict_require_all_samples)
    val_check = strict_check_dataset(val_ds, val_label_map, val_phq_map, f"{tag}/val", args.strict_require_all_samples)
    logger.info("%s loaded train=%s val=%s train_counts=%s val_counts=%s dropped_train=%s dropped_val=%s",
                tag, train_check["loaded_count"], val_check["loaded_count"],
                train_check["loaded_label_counts"], val_check["loaded_label_counts"],
                train_check["dropped_ids"], val_check["dropped_ids"])

    labels_loaded_train = [train_label_map[pid] for pid in dataset_ids(train_ds)]
    class_weight = compute_class_weights(labels_loaded_train, args.class_weight_mode, device)
    ge5_pw, ge10_pw = compute_boundary_pos_weights(labels_loaded_train, args.boundary_pos_weight_mode, device)
    sampler = make_sampler(labels_loaded_train, args.sampler_mode)
    shuffle = sampler is None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_batch, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch, drop_last=False)

    model = build_model(train_ds, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_score = -1e18
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    best_pred_df: pd.DataFrame | None = None
    history: list[dict[str, Any]] = []
    patience_left = args.patience
    ckpt_path = ckpt_dir / f"best_{tag}.pth"

    for ep in range(1, args.epochs + 1):
        train_losses = run_epoch(model, train_loader, optimizer, device, train_label_map, train_phq_map, class_weight, ge5_pw, ge10_pw, args)
        if scheduler is not None:
            scheduler.step()
        pred_df, metrics = evaluate(model, val_loader, device, val_label_map, val_phq_map, args, rep, fold)
        score, eligible = selection_score(metrics, ep, args)
        row = {"epoch": ep, "selection_score": score, "eligible_for_best": int(eligible), "lr": optimizer.param_groups[0]["lr"]}
        row.update(train_losses)
        row.update({k: v for k, v in metrics.items() if not k.startswith("cm_")})
        history.append(row)
        logger.info(
            "%s ep %03d/%03d loss=%.4f main:f1=%.4f acc=%.4f kappa=%.4f r1=%.3f r2=%.3f argmax:f1=%.4f threshold:f1=%.4f ccc=%.4f score=%.4f%s",
            tag, ep, args.epochs, float(train_losses.get("train_total_loss", 0.0)),
            float(metrics.get("f1_main", 0.0)), float(metrics.get("acc_main", 0.0)), float(metrics.get("kappa_main", 0.0)),
            float(metrics.get("recall_main_1", 0.0)), float(metrics.get("recall_main_2", 0.0)),
            float(metrics.get("f1_argmax", 0.0)), float(metrics.get("f1_threshold", 0.0)), float(metrics.get("ccc", 0.0)),
            score, "" if eligible else " (not eligible)"
        )
        if eligible and score > best_score + args.min_delta:
            best_score = score
            best_epoch = ep
            best_metrics = metrics
            best_pred_df = pred_df.copy()
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "repeat": rep,
                "fold": fold,
                "epoch": ep,
                "metrics": metrics,
                "train_check": train_check,
                "val_check": val_check,
            }, ckpt_path)
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            logger.info("%s early stop at epoch %d", tag, ep)
            break

    if best_pred_df is None:
        # This only happens if min_select_epoch > epochs; save final as fallback but flag it.
        logger.warning("%s no eligible best checkpoint; using last epoch predictions", tag)
        best_pred_df = pred_df.copy()
        best_metrics = metrics
        best_epoch = ep
        ckpt_path = ckpt_dir / f"last_{tag}.pth"
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "repeat": rep, "fold": fold, "epoch": ep, "metrics": metrics}, ckpt_path)

    hist_path = base_log_dir / f"history_{tag}.csv"
    pred_path = base_log_dir / f"predictions_{tag}.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    best_pred_df.to_csv(pred_path, index=False)

    split_summary = dict(split["split_summary"])
    split_summary.update({
        "train_loaded": train_check,
        "val_loaded": val_check,
    })
    return {
        "repeat": rep,
        "fold": fold,
        "best_epoch": int(best_epoch),
        "checkpoint_path": str(ckpt_path),
        "history_path": str(hist_path),
        "predictions_path": str(pred_path),
        "train_count_loaded": int(train_check["loaded_count"]),
        "val_count_loaded": int(val_check["loaded_count"]),
        "split_summary": split_summary,
        "metrics": best_metrics,
    }


def search_probability_thresholds(oof: pd.DataFrame) -> dict[str, Any]:
    if oof.empty:
        return {}
    y = oof["label3"].to_numpy(dtype=int)
    p5 = oof["p_ge5"].to_numpy(dtype=float)
    p10 = oof["p_ge10"].to_numpy(dtype=float)
    best: dict[str, Any] = {"score": -1e18}
    for t5 in np.round(np.arange(0.35, 0.701, 0.025), 3):
        for t10 in np.round(np.arange(0.20, 0.701, 0.025), 3):
            pred = np.zeros_like(y)
            pred[p5 >= t5] = 1
            pred[p10 >= t10] = 2
            m = metrics_for_predictions(y, pred, "tmp")
            score = m["f1_tmp"] + 0.1 * m["kappa_tmp"] + 0.05 * m["recall_tmp_1"] + 0.05 * m["recall_tmp_2"]
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "t_ge5": float(t5),
                    "t_ge10": float(t10),
                    "f1": float(m["f1_tmp"]),
                    "acc": float(m["acc_tmp"]),
                    "kappa": float(m["kappa_tmp"]),
                    "recall1": float(m["recall_tmp_1"]),
                    "recall2": float(m["recall_tmp_2"]),
                    "confusion_matrix": m["cm_tmp"],
                }
    return best


def summarize_oof(pred_files: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = [pd.read_csv(p) for p in pred_files if p.exists()]
    if not frames:
        return pd.DataFrame(), {}
    oof = pd.concat(frames, ignore_index=True)
    summary = compute_all_metrics(oof, main_col="pred_main")
    summary["n_oof"] = int(len(oof))
    summary["label_counts"] = {str(k): int(v) for k, v in Counter(oof["label3"].astype(int)).items()}
    summary["best_probability_thresholds_on_oof"] = search_probability_thresholds(oof)
    return oof, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict targeted baseline CV for Track2 ternary")
    p.add_argument("--config", default="config.json")
    p.add_argument("--track", default="Track2")
    p.add_argument("--task", default="ternary")
    p.add_argument("--subtrack", default="A-V-G+P", choices=["A-V+P", "A-V-G+P", "G+P"])
    p.add_argument("--encoder_type", default="bilstm_mean", choices=["bilstm_mean", "hybrid_attn"])
    p.add_argument("--audio_feature", default="wav2vec")
    p.add_argument("--video_feature", default="openface")
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_csv", required=True)
    p.add_argument("--personality_npy", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--cv_repeats", type=int, default=3)
    p.add_argument("--fold_idx", type=int, default=-1, help="Run only one global split index; -1 runs all")
    p.add_argument("--max_folds", type=int, default=0, help="Debug cap after filtering; 0 means all")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--target_t", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--min_select_epoch", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--logs_dir", default="logs")
    p.add_argument("--experiment_name", default="track2_avgp_targeted_baseline_cv5x3")
    p.add_argument("--class_weight_mode", default="sqrt", choices=["none", "inverse", "sqrt", "effective"])
    p.add_argument("--sampler_mode", default="sqrt", choices=["none", "inverse", "sqrt"])
    p.add_argument("--boundary_pos_weight_mode", default="sqrt", choices=["none", "inverse", "sqrt", "effective"])
    p.add_argument("--ce_loss_weight", type=float, default=1.0)
    p.add_argument("--ord_loss_weight", type=float, default=0.5)
    p.add_argument("--phq_loss_weight", type=float, default=0.0)
    p.add_argument("--ge5_loss_weight", type=float, default=1.0)
    p.add_argument("--ge10_loss_weight", type=float, default=1.2)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--focal_gamma", type=float, default=0.0)
    p.add_argument("--pred_mode", default="argmax", choices=["argmax", "threshold"])
    p.add_argument("--pred_ge5_threshold", type=float, default=0.50)
    p.add_argument("--pred_ge10_threshold", type=float, default=0.45)
    p.add_argument("--kappa_weight", type=float, default=0.1)
    p.add_argument("--recall1_weight", type=float, default=0.05)
    p.add_argument("--recall2_weight", type=float, default=0.05)
    p.add_argument("--zero_recall_penalty", type=float, default=0.1)
    p.add_argument("--ccc_weight", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--cosine_scheduler", action="store_true")
    p.add_argument("--strict_require_all_samples", action="store_true")
    args = p.parse_args()
    if args.task != "ternary":
        raise ValueError("This targeted script is currently for ternary Track2 only.")
    return args


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
    log_dir = Path(args.logs_dir) / args.track / args.subtrack / args.task / args.experiment_name / timestamp
    ckpt_dir = Path(args.checkpoints_dir) / args.track / args.subtrack / args.task / args.experiment_name / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir)
    logger.info("Arguments: %s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    logger.info("Using device: %s", device)

    rows = read_train_rows_strict(args.split_csv, require_train_split=True)
    bad = validate_label_phq_consistency(rows)
    if bad:
        raise AssertionError(f"split_csv label3/PHQ inconsistency: {bad[:20]}")
    logger.info("Loaded strict train rows=%d label_counts=%s", len(rows), class_counts(r.label3 for r in rows))

    splits = make_cv_splits(rows, args.cv_folds, args.cv_repeats, args.seed)
    if args.fold_idx >= 0:
        splits = [splits[args.fold_idx]]
    if args.max_folds > 0:
        splits = splits[:args.max_folds]
    logger.info("Running %d split(s)", len(splits))

    fold_results = []
    for i, split in enumerate(splits):
        logger.info("=== Split %d/%d: rep=%d fold=%d ===", i + 1, len(splits), split["repeat"], split["fold"])
        result = train_one_split(split, rows, args, log_dir, ckpt_dir, device, logger)
        result["split_index"] = int(i if args.fold_idx < 0 else args.fold_idx)
        fold_results.append(result)

    pred_files = [Path(r["predictions_path"]) for r in fold_results]
    oof, oof_summary = summarize_oof(pred_files)
    if not oof.empty:
        oof_path = log_dir / "oof_predictions.csv"
        oof.to_csv(oof_path, index=False)
    else:
        oof_path = None

    fold_summary_rows = []
    for r in fold_results:
        row = {
            "split_index": r.get("split_index"),
            "repeat": r["repeat"],
            "fold": r["fold"],
            "best_epoch": r["best_epoch"],
            "train_count_loaded": r["train_count_loaded"],
            "val_count_loaded": r["val_count_loaded"],
        }
        for k, v in r["metrics"].items():
            if not isinstance(v, (list, dict)):
                row[k] = v
        fold_summary_rows.append(row)
    pd.DataFrame(fold_summary_rows).to_csv(log_dir / "fold_summary.csv", index=False)

    payload = {
        "experiment_name": args.experiment_name,
        "timestamp": timestamp,
        "task": args.task,
        "track": args.track,
        "subtrack": args.subtrack,
        "encoder_type": args.encoder_type,
        "audio_feature": args.audio_feature,
        "video_feature": args.video_feature,
        "config": vars(args),
        "strict_train_rows": len(rows),
        "strict_train_label_counts": class_counts(r.label3 for r in rows),
        "fold_results": fold_results,
        "oof_summary": oof_summary,
        "oof_predictions_path": str(oof_path) if oof_path else None,
        "log_dir": str(log_dir),
        "checkpoints_dir": str(ckpt_dir),
    }
    with open(log_dir / "train_targeted_baseline_cv_result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Done. log_dir=%s", log_dir)
    logger.info("OOF summary: %s", json.dumps(oof_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
