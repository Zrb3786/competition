#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V15 trained fusion for MPDD-AVG Elder.

This script trains a small neural gated fusion model on already validated expert
OOF/test predictions. It does NOT read raw official feature directories.
It uses expert probabilities as evidence and learns sample-wise expert weights.

Expected expert files per directory:
  oof_predictions.csv
  test_predictions.csv

Submission format:
  predictions_model_argmax/submission.zip
  predictions_model_safe_threshold/submission.zip
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold


# ----------------------------- utils -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def to_jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise KeyError(f"missing any of columns {candidates}; available={list(df.columns)}")
    return None


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_logits(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    prob = np.clip(prob, eps, 1.0)
    prob = prob / prob.sum(axis=-1, keepdims=True)
    return np.log(prob)


def entropy_np(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(prob, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1) / math.log(p.shape[-1])


def margin_np(prob: np.ndarray) -> np.ndarray:
    s = np.sort(prob, axis=-1)
    return s[:, -1] - s[:, -2]


# ----------------------------- metrics -----------------------------

def ccc_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(), y_pred.var()
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    den = vt + vp + (mt - mp) ** 2 + eps
    return float(2.0 * cov / den)


def compute_metrics(y2, pred2, y3, pred3, phq=None, phq_pred=None) -> Dict[str, float]:
    y2 = np.asarray(y2, dtype=int)
    pred2 = np.asarray(pred2, dtype=int)
    y3 = np.asarray(y3, dtype=int)
    pred3 = np.asarray(pred3, dtype=int)
    d = {
        "binary_acc": float(accuracy_score(y2, pred2)),
        "binary_macro_f1": float(f1_score(y2, pred2, average="macro", zero_division=0)),
        "binary_kappa": float(cohen_kappa_score(y2, pred2)),
        "ternary_acc": float(accuracy_score(y3, pred3)),
        "ternary_macro_f1": float(f1_score(y3, pred3, average="macro", zero_division=0)),
        "ternary_kappa": float(cohen_kappa_score(y3, pred3)),
        "inconsistent": float(((pred2 == 0) != (pred3 == 0)).sum()),
    }
    if phq is not None and phq_pred is not None:
        phq = np.asarray(phq, dtype=float)
        phq_pred = np.asarray(phq_pred, dtype=float)
        d.update({
            "phq_ccc": ccc_np(phq, phq_pred),
            "phq_mae": float(np.mean(np.abs(phq - phq_pred))),
            "phq_rmse": float(np.sqrt(np.mean((phq - phq_pred) ** 2))),
        })
    return d


def select_score(m: Dict[str, float], mode: str = "cls") -> float:
    if mode == "cls":
        return (
            0.28 * m.get("binary_macro_f1", 0.0)
            + 0.22 * m.get("binary_kappa", 0.0)
            + 0.30 * m.get("ternary_macro_f1", 0.0)
            + 0.20 * m.get("ternary_kappa", 0.0)
        )
    if mode == "balanced":
        return (
            0.22 * m.get("binary_macro_f1", 0.0)
            + 0.18 * m.get("binary_kappa", 0.0)
            + 0.26 * m.get("ternary_macro_f1", 0.0)
            + 0.22 * m.get("ternary_kappa", 0.0)
            + 0.12 * max(0.0, m.get("phq_ccc", 0.0))
        )
    if mode == "ternary":
        return 0.55 * m.get("ternary_macro_f1", 0.0) + 0.45 * m.get("ternary_kappa", 0.0)
    return (
        m.get("binary_macro_f1", 0.0) + m.get("binary_kappa", 0.0)
        + m.get("ternary_macro_f1", 0.0) + m.get("ternary_kappa", 0.0)
        + max(0.0, m.get("phq_ccc", 0.0))
    )


# ----------------------------- expert IO -----------------------------

@dataclass
class ExpertData:
    name: str
    oof: pd.DataFrame
    test: pd.DataFrame
    train_dir: Path


def parse_expert_dirs(spec: str, expert_root: Path) -> Dict[str, Path]:
    """Parse name:dir,name:dir or name:name-substring."""
    out: Dict[str, Path] = {}
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        if ":" in item:
            name, d = item.split(":", 1)
        else:
            name, d = item, item
        name = name.strip()
        d = d.strip()
        p = Path(d)
        if not p.is_absolute():
            p0 = expert_root / d
            if p0.exists():
                p = p0
            else:
                # fuzzy search by expert token
                hits = sorted([x for x in expert_root.glob("*") if x.is_dir() and name in x.name])
                if len(hits) == 1:
                    p = hits[0]
                elif len(hits) > 1:
                    # prefer exact suffix/prefix-ish names
                    exact = [x for x in hits if x.name.endswith(f"_{name}_5x1") or x.name.endswith(f"{name}_5x1")]
                    p = exact[0] if exact else hits[0]
                else:
                    p = p0
        out[name] = p
    return out


def load_prob_columns(df: pd.DataFrame, kind: str, n: int) -> np.ndarray:
    if kind == "prob3":
        names = [["prob3_0", "p3_0", "class0_prob", "prob_0"], ["prob3_1", "p3_1", "class1_prob", "prob_1"], ["prob3_2", "p3_2", "class2_prob", "prob_2"]]
    else:
        names = [["prob2_0", "p2_0", "binary_prob_0"], ["prob2_1", "p2_1", "binary_prob_1", "prob_positive"]]
    cols = [find_col(df, cand, required=False) for cand in names]
    if all(c is not None for c in cols):
        arr = df[cols].to_numpy(dtype=np.float32)
        arr = np.clip(arr, 1e-6, 1.0)
        arr = arr / arr.sum(axis=1, keepdims=True)
        return arr
    # Fallback from predictions.
    pred_col = find_col(df, ["pred3", "ternary_pred"], required=False) if kind == "prob3" else find_col(df, ["pred2", "binary_pred"], required=False)
    if pred_col is None:
        raise KeyError(f"missing {kind} columns and prediction fallback; cols={list(df.columns)}")
    pred = df[pred_col].astype(int).to_numpy()
    arr = np.full((len(df), n), 0.02, dtype=np.float32)
    arr[np.arange(len(df)), np.clip(pred, 0, n - 1)] = 0.96
    arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


def normalize_expert_df(df: pd.DataFrame, labels: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    id_col = find_col(df, ["id", "ID"])
    out = pd.DataFrame({"id": df[id_col].astype(int)})
    p3 = load_prob_columns(df, "prob3", 3)
    p2 = load_prob_columns(df, "prob2", 2)
    out[["prob3_0", "prob3_1", "prob3_2"]] = p3
    out[["prob2_0", "prob2_1"]] = p2
    phq_col = find_col(df, ["phq_pred", "pred_phq", "PHQ_pred", "phq9_pred"], required=False)
    if phq_col is not None:
        out["phq_pred"] = df[phq_col].astype(float).to_numpy()
    else:
        # class expectation fallback later; put nan now
        out["phq_pred"] = np.nan
    pred3_col = find_col(df, ["pred3", "ternary_pred"], required=False)
    pred2_col = find_col(df, ["pred2", "binary_pred"], required=False)
    out["pred3"] = df[pred3_col].astype(int).to_numpy() if pred3_col else p3.argmax(axis=1)
    out["pred2"] = df[pred2_col].astype(int).to_numpy() if pred2_col else (out["pred3"].to_numpy() > 0).astype(int)
    if labels is not None:
        out = out.merge(labels, on="id", how="left")
    return out.sort_values("id").reset_index(drop=True)


def load_labels(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    id_col = find_col(df, ["id", "ID"])
    y3_col = find_col(df, ["label3", "ternary", "ternary_label"])
    y2_col = find_col(df, ["label2", "binary", "binary_label"], required=False)
    phq_col = find_col(df, ["PHQ-9", "phq", "phq9", "PHQ9"], required=False)
    out = pd.DataFrame({"id": df[id_col].astype(int), "y3": df[y3_col].astype(int)})
    out["y2"] = df[y2_col].astype(int) if y2_col else (out["y3"] > 0).astype(int)
    out["phq"] = df[phq_col].astype(float) if phq_col else np.nan
    return out.sort_values("id").reset_index(drop=True)


def load_experts(expert_root: Path, experts_spec: str, label_df: pd.DataFrame) -> List[ExpertData]:
    dirs = parse_expert_dirs(experts_spec, expert_root)
    experts: List[ExpertData] = []
    for name, d in dirs.items():
        oof_path = d / "oof_predictions.csv"
        test_path = d / "test_predictions.csv"
        if not oof_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"expert={name} missing oof/test: {oof_path} {test_path}")
        oof = normalize_expert_df(read_csv_auto(oof_path), labels=label_df)
        test = normalize_expert_df(read_csv_auto(test_path), labels=None)
        experts.append(ExpertData(name=name, oof=oof, test=test, train_dir=d))
        print(f"[LOAD] {name:18s} dir={d} oof={oof.shape} test={test.shape}")
    # Validate common IDs/order.
    base_ids = experts[0].oof["id"].tolist()
    base_test_ids = experts[0].test["id"].tolist()
    for e in experts[1:]:
        if e.oof["id"].tolist() != base_ids:
            raise ValueError(f"OOF id mismatch for expert {e.name}")
        if e.test["id"].tolist() != base_test_ids:
            raise ValueError(f"TEST id mismatch for expert {e.name}")
    return experts


def build_meta_arrays(experts: List[ExpertData], class_means: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    ids_oof = experts[0].oof[["id", "y2", "y3", "phq"]].copy()
    ids_test = experts[0].test[["id"]].copy()
    E = len(experts)
    N = len(ids_oof)
    M = len(ids_test)
    oof_p3 = np.zeros((N, E, 3), dtype=np.float32)
    oof_p2 = np.zeros((N, E, 2), dtype=np.float32)
    oof_phq = np.zeros((N, E), dtype=np.float32)
    test_p3 = np.zeros((M, E, 3), dtype=np.float32)
    test_p2 = np.zeros((M, E, 2), dtype=np.float32)
    test_phq = np.zeros((M, E), dtype=np.float32)
    for j, e in enumerate(experts):
        oof_p3[:, j, :] = e.oof[["prob3_0", "prob3_1", "prob3_2"]].to_numpy(np.float32)
        oof_p2[:, j, :] = e.oof[["prob2_0", "prob2_1"]].to_numpy(np.float32)
        phq = e.oof["phq_pred"].to_numpy(np.float32)
        bad = ~np.isfinite(phq)
        if bad.any():
            phq[bad] = (oof_p3[bad, j, :] * class_means[None, :]).sum(axis=1)
        oof_phq[:, j] = phq
        test_p3[:, j, :] = e.test[["prob3_0", "prob3_1", "prob3_2"]].to_numpy(np.float32)
        test_p2[:, j, :] = e.test[["prob2_0", "prob2_1"]].to_numpy(np.float32)
        phq_t = e.test["phq_pred"].to_numpy(np.float32)
        bad_t = ~np.isfinite(phq_t)
        if bad_t.any():
            phq_t[bad_t] = (test_p3[bad_t, j, :] * class_means[None, :]).sum(axis=1)
        test_phq[:, j] = phq_t
    oof = {"p3": oof_p3, "p2": oof_p2, "phq": oof_phq}
    test = {"p3": test_p3, "p2": test_p2, "phq": test_phq}
    return oof, test, ids_oof, ids_test


# ----------------------------- model -----------------------------

class GatedEvidenceFusion(nn.Module):
    def __init__(self, n_experts: int, hidden: int = 48, dropout: float = 0.25, residual_scale: float = 0.15):
        super().__init__()
        self.n_experts = n_experts
        self.residual_scale = residual_scale
        # per expert features: prob3(3), logit3(3), entropy(1), margin(1), phq(1), prob2_pos(1) = 10
        in_dim = n_experts * 10
        self.gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_experts),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        self.phq_residual = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.expert_bias = nn.Parameter(torch.zeros(n_experts, 3))
        self.temp_log = nn.Parameter(torch.zeros(n_experts))

    def make_features(self, p3: torch.Tensor, p2: torch.Tensor, phq: torch.Tensor) -> torch.Tensor:
        # p3: [B,E,3], p2: [B,E,2], phq: [B,E]
        logits = torch.log(torch.clamp(p3, 1e-6, 1.0))
        ent = -(p3 * logits).sum(-1, keepdim=True) / math.log(3.0)
        top2 = torch.topk(p3, k=2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).unsqueeze(-1)
        phq_feat = phq.unsqueeze(-1) / 15.0
        p2pos = p2[..., 1:2]
        feat = torch.cat([p3, logits, ent, margin, phq_feat, p2pos], dim=-1)
        return feat.reshape(feat.shape[0], -1)

    def forward(self, p3: torch.Tensor, p2: torch.Tensor, phq: torch.Tensor, class_means: torch.Tensor):
        x = self.make_features(p3, p2, phq)
        gate_logits = self.gate(x)
        gate = torch.softmax(gate_logits, dim=-1)  # [B,E]
        temp = torch.exp(self.temp_log).view(1, self.n_experts, 1).clamp(0.5, 2.0)
        expert_logits = torch.log(torch.clamp(p3, 1e-6, 1.0)) / temp + self.expert_bias.view(1, self.n_experts, 3)
        mixed_logits = (gate.unsqueeze(-1) * expert_logits).sum(dim=1)
        mixed_logits = mixed_logits + self.residual_scale * torch.tanh(self.residual(x))
        prob3 = torch.softmax(mixed_logits, dim=-1)
        prob2_pos = prob3[:, 1] + prob3[:, 2]
        logits2 = torch.stack([torch.log1p(-prob2_pos.clamp(1e-6, 1 - 1e-6)), torch.log(prob2_pos.clamp(1e-6, 1 - 1e-6))], dim=1)
        exp_class_phq = (prob3 * class_means.view(1, 3)).sum(dim=1)
        expert_phq = (gate * phq).sum(dim=1)
        residual_phq = 2.5 * torch.tanh(self.phq_residual(x).squeeze(-1))
        phq_pred = 0.65 * exp_class_phq + 0.25 * expert_phq + 0.10 * (exp_class_phq + residual_phq)
        return {
            "logits3": mixed_logits,
            "prob3": prob3,
            "logits2": logits2,
            "prob2_pos": prob2_pos,
            "phq_pred": phq_pred,
            "gate": gate,
        }


# ----------------------------- losses -----------------------------

def soft_macro_f1_loss(prob: torch.Tensor, y: torch.Tensor, n_classes: int, eps: float = 1e-7) -> torch.Tensor:
    y_one = F.one_hot(y, num_classes=n_classes).float()
    tp = (prob * y_one).sum(0)
    fp = (prob * (1 - y_one)).sum(0)
    fn = ((1 - prob) * y_one).sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return 1.0 - f1.mean()


def soft_kappa_loss(prob: torch.Tensor, y: torch.Tensor, n_classes: int, eps: float = 1e-7) -> torch.Tensor:
    y_one = F.one_hot(y, num_classes=n_classes).float()
    conf = y_one.t() @ prob
    # quadratic weights for ordinal-ish kappa; acceptable for binary too
    idx = torch.arange(n_classes, device=prob.device).float()
    W = (idx[:, None] - idx[None, :]).pow(2) / max(1.0, float((n_classes - 1) ** 2))
    hist_true = y_one.sum(0)
    hist_pred = prob.sum(0)
    expected = hist_true[:, None] @ hist_pred[None, :] / y.shape[0]
    obs = (W * conf).sum()
    exp = (W * expected).sum().clamp_min(eps)
    kappa = 1.0 - obs / exp
    return 1.0 - kappa


def ccc_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(unbiased=False), y_pred.var(unbiased=False)
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    ccc = 2 * cov / (vt + vp + (mt - mp).pow(2) + eps)
    return 1.0 - ccc


def train_loss(out, y2, y3, phq, class_weights3=None, args=None):
    loss = F.cross_entropy(out["logits3"], y3, weight=class_weights3, label_smoothing=0.03)
    loss = loss + args.binary_weight * F.cross_entropy(out["logits2"], y2, label_smoothing=0.02)
    loss = loss + args.soft_f1_weight * soft_macro_f1_loss(out["prob3"], y3, 3)
    loss = loss + args.kappa_weight * soft_kappa_loss(out["prob3"], y3, 3)
    # PHQ loss, guarded because not every label CSV may contain phq.
    if torch.isfinite(phq).all():
        loss = loss + args.phq_weight * F.smooth_l1_loss(out["phq_pred"], phq)
        loss = loss + args.ccc_weight * ccc_loss_torch(phq, out["phq_pred"])
    # Encourage non-degenerate gates, but weakly.
    gate = out["gate"]
    ent = -(gate * torch.log(gate.clamp_min(1e-7))).sum(1).mean() / math.log(gate.shape[1])
    max_gate = gate.max(1).values.mean()
    loss = loss + args.gate_peak_weight * F.relu(max_gate - args.max_gate).pow(2)
    loss = loss - args.gate_entropy_weight * ent
    return loss


# ----------------------------- training -----------------------------

class MetaDataset(torch.utils.data.Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], ids: pd.DataFrame, indices: np.ndarray, class_means: np.ndarray):
        self.p3 = arrays["p3"][indices]
        self.p2 = arrays["p2"][indices]
        self.phq_exp = arrays["phq"][indices].copy()
        # Fill expert phq nan if any.
        bad = ~np.isfinite(self.phq_exp)
        if bad.any():
            cls_exp = (self.p3 * class_means[None, None, :]).sum(-1)
            self.phq_exp[bad] = cls_exp[bad]
        sub = ids.iloc[indices].reset_index(drop=True)
        self.y2 = sub["y2"].astype(int).to_numpy()
        self.y3 = sub["y3"].astype(int).to_numpy()
        self.phq = sub["phq"].astype(float).to_numpy()
        self.id = sub["id"].astype(int).to_numpy()

    def __len__(self): return len(self.id)

    def __getitem__(self, idx):
        return {
            "p3": torch.tensor(self.p3[idx], dtype=torch.float32),
            "p2": torch.tensor(self.p2[idx], dtype=torch.float32),
            "expert_phq": torch.tensor(self.phq_exp[idx], dtype=torch.float32),
            "y2": torch.tensor(self.y2[idx], dtype=torch.long),
            "y3": torch.tensor(self.y3[idx], dtype=torch.long),
            "phq": torch.tensor(self.phq[idx], dtype=torch.float32),
            "id": torch.tensor(self.id[idx], dtype=torch.long),
        }


def make_loader(ds, batch_size, shuffle):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


@torch.no_grad()
def predict_model(model, arrays, ids_df, indices, class_means, device, batch_size=64):
    ds = MetaDataset(arrays, ids_df, indices, class_means)
    loader = make_loader(ds, batch_size, False)
    model.eval()
    all_ids, p3s, p2s, phqs, gates = [], [], [], [], []
    cm = torch.tensor(class_means, dtype=torch.float32, device=device)
    for b in loader:
        p3 = b["p3"].to(device)
        p2 = b["p2"].to(device)
        ephq = b["expert_phq"].to(device)
        out = model(p3, p2, ephq, cm)
        all_ids.append(b["id"].cpu().numpy())
        p3s.append(out["prob3"].cpu().numpy())
        prob2_pos = out["prob2_pos"].cpu().numpy()
        p2s.append(np.stack([1 - prob2_pos, prob2_pos], axis=1))
        phqs.append(out["phq_pred"].cpu().numpy())
        gates.append(out["gate"].cpu().numpy())
    return {
        "id": np.concatenate(all_ids),
        "prob3": np.concatenate(p3s),
        "prob2": np.concatenate(p2s),
        "phq_pred": np.concatenate(phqs),
        "gate": np.concatenate(gates),
    }


def payload_to_df(payload, labels_df=None, expert_names=None):
    df = pd.DataFrame({"id": payload["id"].astype(int)})
    p3 = payload["prob3"]
    p2 = payload["prob2"]
    df[["prob3_0", "prob3_1", "prob3_2"]] = p3
    df[["prob2_0", "prob2_1"]] = p2
    df["phq_pred"] = payload["phq_pred"]
    df["pred3"] = p3.argmax(axis=1).astype(int)
    df["pred2"] = (df["pred3"].to_numpy() > 0).astype(int)
    if expert_names is not None and "gate" in payload:
        for j, name in enumerate(expert_names):
            df[f"gate_{name}"] = payload["gate"][:, j]
    if labels_df is not None:
        df = df.merge(labels_df, on="id", how="left")
    return df.sort_values("id").reset_index(drop=True)


def train_one_fold(arrays, ids_df, train_idx, val_idx, class_means, args, device, seed):
    seed_everything(seed)
    model = GatedEvidenceFusion(
        n_experts=arrays["p3"].shape[1],
        hidden=args.hidden,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    ).to(device)
    y3_train = ids_df.iloc[train_idx]["y3"].astype(int).to_numpy()
    counts = np.bincount(y3_train, minlength=3).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0)) ** args.class_weight_power
    weights = weights / weights.mean()
    class_weights3 = torch.tensor(weights, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
    train_ds = MetaDataset(arrays, ids_df, train_idx, class_means)
    train_loader = make_loader(train_ds, args.batch_size, True)
    cm = torch.tensor(class_means, dtype=torch.float32, device=device)
    best = {"score": -1e9, "state": None, "metrics": None, "payload": None, "epoch": 0}
    bad = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for b in train_loader:
            opt.zero_grad(set_to_none=True)
            p3 = b["p3"].to(device); p2 = b["p2"].to(device); ephq = b["expert_phq"].to(device)
            y2 = b["y2"].to(device); y3 = b["y3"].to(device); phq = b["phq"].to(device)
            out = model(p3, p2, ephq, cm)
            loss = train_loss(out, y2, y3, phq, class_weights3, args)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        payload = predict_model(model, arrays, ids_df, val_idx, class_means, device, args.batch_size)
        val_df = payload_to_df(payload, labels_df=ids_df[["id", "y2", "y3", "phq"]])
        m = compute_metrics(val_df["y2"], val_df["pred2"], val_df["y3"], val_df["pred3"], val_df["phq"], val_df["phq_pred"])
        sc = select_score(m, args.score_mode)
        if sc > best["score"]:
            best = {"score": sc, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "metrics": m, "payload": payload, "epoch": ep}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % args.log_every == 0:
            print(f"[ep={ep:03d}] loss={np.mean(losses):.4f} score={sc:.4f} tF1={m['ternary_macro_f1']:.4f} tK={m['ternary_kappa']:.4f} bF1={m['binary_macro_f1']:.4f} CCC={m.get('phq_ccc',0):.4f}", flush=True)
        if bad >= args.patience:
            print(f"[early stop] epoch={ep} best_epoch={best['epoch']} best_score={best['score']:.4f}", flush=True)
            break
    model.load_state_dict(best["state"])
    return model, best


def train_cv(oof_arrays, ids_df, class_means, expert_names, args, device):
    y = ids_df["y3"].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    oof_parts = []
    fold_rows = []
    models = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"\n[FOLD {fold}] train={len(tr)} val={len(va)}", flush=True)
        model, best = train_one_fold(oof_arrays, ids_df, tr, va, class_means, args, device, seed=args.seed + fold)
        df = payload_to_df(best["payload"], labels_df=ids_df[["id", "y2", "y3", "phq"]], expert_names=expert_names)
        df["fold"] = fold
        oof_parts.append(df)
        row = {"fold": fold, "best_epoch": best["epoch"], "best_score": best["score"]}
        row.update(best["metrics"])
        fold_rows.append(row)
        models.append(model)
        print(f"[FOLD {fold} BEST] {row}", flush=True)
    oof_df = pd.concat(oof_parts, ignore_index=True).sort_values("id").reset_index(drop=True)
    m = compute_metrics(oof_df["y2"], oof_df["pred2"], oof_df["y3"], oof_df["pred3"], oof_df["phq"], oof_df["phq_pred"])
    print("[OOF]", m, "score", select_score(m, args.score_mode), flush=True)
    return models, oof_df, pd.DataFrame(fold_rows), m


def train_full_models(oof_arrays, ids_df, test_arrays, test_ids, class_means, expert_names, args, device):
    idx = np.arange(len(ids_df))
    test_payloads = []
    for si, seed in enumerate([args.seed + 1000 + i for i in range(args.full_seeds)]):
        print(f"\n[FULL seed={seed}]", flush=True)
        # Small holdout for early stopping: use one split but train best based on internal val.
        y = ids_df["y3"].astype(int).to_numpy()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.split_seed + si)
        tr, va = next(iter(skf.split(np.zeros(len(y)), y)))
        model, best = train_one_fold(oof_arrays, ids_df, tr, va, class_means, args, device, seed=seed)
        # Fine-tune a little on all data? conservative: train all for best_epoch//3 from current state.
        # Keep simple: use best model as is; avoid overfit.
        payload = predict_test_arrays(model, test_arrays, test_ids, class_means, device, args.batch_size, expert_names)
        test_payloads.append(payload)
    # Average test payloads.
    p3 = np.mean([p["prob3"] for p in test_payloads], axis=0)
    p2 = np.mean([p["prob2"] for p in test_payloads], axis=0)
    phq = np.mean([p["phq_pred"] for p in test_payloads], axis=0)
    gate = np.mean([p["gate"] for p in test_payloads], axis=0)
    return {"id": test_payloads[0]["id"], "prob3": p3, "prob2": p2, "phq_pred": phq, "gate": gate}


@torch.no_grad()
def predict_test_arrays(model, arrays, test_ids, class_means, device, batch_size, expert_names):
    model.eval()
    N = arrays["p3"].shape[0]
    ids_df = pd.DataFrame({"id": test_ids["id"].astype(int), "y2": 0, "y3": 0, "phq": 0.0})
    idx = np.arange(N)
    return predict_model(model, arrays, ids_df, idx, class_means, device, batch_size)


# ----------------------------- postprocess and save -----------------------------

def distribution(pred2: np.ndarray, pred3: np.ndarray, ids: np.ndarray) -> Dict:
    return {
        "n": int(len(ids)),
        "binary": {str(k): int(v) for k, v in pd.Series(pred2).value_counts().sort_index().items()},
        "ternary": {str(k): int(v) for k, v in pd.Series(pred3).value_counts().sort_index().items()},
        "severe": [int(i) for i in ids[pred3 == 2]],
        "positive": [int(i) for i in ids[pred2 == 1]],
        "inconsistent": int(((pred2 == 0) != (pred3 == 0)).sum()),
    }


def save_submission(out_dir: Path, ids: np.ndarray, pred3: np.ndarray, tag: str, extra_df: Optional[pd.DataFrame] = None) -> Path:
    d = ensure_dir(out_dir / tag)
    pred2 = (pred3 > 0).astype(int)
    b = pd.DataFrame({"id": ids.astype(int), "binary_pred": pred2.astype(int)})
    t = pd.DataFrame({"id": ids.astype(int), "ternary_pred": pred3.astype(int)})
    b.to_csv(d / "binary.csv", index=False)
    t.to_csv(d / "ternary.csv", index=False)
    if extra_df is not None:
        extra_df.to_csv(d / "test_predictions.csv", index=False)
    zpath = d / "submission.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(d / "binary.csv", arcname="binary.csv")
        z.write(d / "ternary.csv", arcname="ternary.csv")
    write_json(distribution(pred2, pred3, ids), d / "distribution_report.json")
    return zpath


def threshold_search_from_probs(prob3: np.ndarray, y3: np.ndarray, y2: np.ndarray, phq: np.ndarray, phq_pred: np.ndarray, args) -> Dict:
    # Search severity score thresholds on OOF. Not pure fusion, just calibration of trained model.
    score = prob3[:, 1] + 2 * prob3[:, 2]
    best = None
    grid0 = np.linspace(args.t0_min, args.t0_max, args.t_steps)
    grid1 = np.linspace(args.t1_min, args.t1_max, args.t_steps)
    for t0 in grid0:
        for t1 in grid1:
            if t1 <= t0 + 0.05:
                continue
            pred3 = np.where(score <= t0, 0, np.where(score <= t1, 1, 2)).astype(int)
            pred2 = (pred3 > 0).astype(int)
            m = compute_metrics(y2, pred2, y3, pred3, phq, phq_pred)
            sc = select_score(m, args.score_mode)
            if best is None or sc > best["score"]:
                best = {"score": sc, "t0": float(t0), "t1": float(t1), "metrics": m}
    return best or {"score": -1e9, "t0": 0.5, "t1": 1.3, "metrics": {}}


def apply_threshold(prob3: np.ndarray, t0: float, t1: float) -> np.ndarray:
    score = prob3[:, 1] + 2 * prob3[:, 2]
    return np.where(score <= t0, 0, np.where(score <= t1, 1, 2)).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert_root", type=str, default="outputs/elder_v14_v12loader")
    ap.add_argument("--experts", type=str, default="p:v14v12_p_5x1,audio_controlled:v14v12_audio_controlled_5x1,gait:v14v12_gait_5x1,video:v14v12_video_5x1,audio_big:v14v12_audio_big_5x1")
    ap.add_argument("--train_split_csv", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="outputs/elder_v15_trained_fusion/v15_gated")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--score_mode", type=str, default="cls", choices=["cls", "balanced", "ternary", "all"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split_seed", type=int, default=2026)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full_seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=2e-3)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--residual_scale", type=float, default=0.12)
    ap.add_argument("--binary_weight", type=float, default=0.60)
    ap.add_argument("--soft_f1_weight", type=float, default=0.20)
    ap.add_argument("--kappa_weight", type=float, default=0.15)
    ap.add_argument("--phq_weight", type=float, default=0.15)
    ap.add_argument("--ccc_weight", type=float, default=0.08)
    ap.add_argument("--class_weight_power", type=float, default=0.5)
    ap.add_argument("--gate_entropy_weight", type=float, default=0.005)
    ap.add_argument("--gate_peak_weight", type=float, default=0.03)
    ap.add_argument("--max_gate", type=float, default=0.78)
    ap.add_argument("--log_every", type=int, default=10)
    # threshold calibration parameters
    ap.add_argument("--t_steps", type=int, default=81)
    ap.add_argument("--t0_min", type=float, default=0.35)
    ap.add_argument("--t0_max", type=float, default=0.95)
    ap.add_argument("--t1_min", type=float, default=0.95)
    ap.add_argument("--t1_max", type=float, default=1.75)
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.output_dir))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    label_df = load_labels(Path(args.train_split_csv))
    class_means = np.array([label_df.loc[label_df.y3 == c, "phq"].mean() for c in range(3)], dtype=np.float32)
    # fallback if no phq.
    if not np.isfinite(class_means).all():
        class_means = np.array([1.2, 4.2, 9.9], dtype=np.float32)
    print("[INFO] class_means", class_means.tolist(), flush=True)
    experts = load_experts(Path(args.expert_root), args.experts, label_df)
    expert_names = [e.name for e in experts]
    oof_arrays, test_arrays, ids_oof, ids_test = build_meta_arrays(experts, class_means)
    print("[INFO] meta shapes", {k: v.shape for k, v in oof_arrays.items()}, "test", {k: v.shape for k, v in test_arrays.items()}, flush=True)

    # Baseline expert metrics.
    baseline_rows = []
    for j, e in enumerate(experts):
        pred3 = oof_arrays["p3"][:, j, :].argmax(1)
        pred2 = (pred3 > 0).astype(int)
        m = compute_metrics(ids_oof.y2, pred2, ids_oof.y3, pred3, ids_oof.phq, oof_arrays["phq"][:, j])
        row = {"expert": e.name, "score": select_score(m, args.score_mode)}; row.update(m)
        baseline_rows.append(row)
    pd.DataFrame(baseline_rows).to_csv(out_dir / "baseline_expert_oof_metrics.csv", index=False)

    models, oof_df, fold_df, oof_metrics = train_cv(oof_arrays, ids_oof, class_means, expert_names, args, device)
    oof_df.to_csv(out_dir / "oof_predictions_v15.csv", index=False)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    write_json(oof_metrics, out_dir / "oof_metrics_argmax.json")

    # Threshold calibration on OOF.
    best_thr = threshold_search_from_probs(
        oof_df[["prob3_0", "prob3_1", "prob3_2"]].to_numpy(np.float32),
        oof_df["y3"].to_numpy(int),
        oof_df["y2"].to_numpy(int),
        oof_df["phq"].to_numpy(float),
        oof_df["phq_pred"].to_numpy(float),
        args,
    )
    write_json(best_thr, out_dir / "threshold_search_best.json")
    print("[THRESHOLD]", best_thr, flush=True)

    # Train full seed models and predict test.
    test_payload = train_full_models(oof_arrays, ids_oof, test_arrays, ids_test, class_means, expert_names, args, device)
    test_df = payload_to_df(test_payload, labels_df=None, expert_names=expert_names)
    test_df.to_csv(out_dir / "test_predictions_v15_argmax.csv", index=False)

    ids = test_df["id"].to_numpy(int)
    prob3_test = test_df[["prob3_0", "prob3_1", "prob3_2"]].to_numpy(np.float32)
    pred3_argmax = prob3_test.argmax(1).astype(int)
    save_submission(out_dir, ids, pred3_argmax, "predictions_model_argmax", test_df)

    pred3_thr = apply_threshold(prob3_test, best_thr["t0"], best_thr["t1"])
    test_df_thr = test_df.copy()
    test_df_thr["pred3"] = pred3_thr
    test_df_thr["pred2"] = (pred3_thr > 0).astype(int)
    save_submission(out_dir, ids, pred3_thr, "predictions_model_threshold", test_df_thr)

    summary = {
        "experts": expert_names,
        "class_means": class_means.tolist(),
        "oof_argmax": oof_metrics,
        "oof_threshold": best_thr,
        "test_argmax_distribution": distribution((pred3_argmax > 0).astype(int), pred3_argmax, ids),
        "test_threshold_distribution": distribution((pred3_thr > 0).astype(int), pred3_thr, ids),
        "output_dir": str(out_dir),
    }
    write_json(summary, out_dir / "run_summary.json")
    print("\n[SUMMARY]")
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), flush=True)
    print("[DONE]", out_dir, flush=True)


if __name__ == "__main__":
    main()
