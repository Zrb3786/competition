#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V14 expert probe using the validated v12/v3/v11 loader.

Key principle:
  - Do NOT implement a new official feature directory loader here.
  - Import and reuse v12_loader.ElderFeatureStore / make_store_from_args.
  - V14 only defines expert feature grouping and expert models.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold

import v12_loader as v12

EPS = 1e-8
PAIR_COUNT = 4

# ----------------------------- utils -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def np_clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pad_last(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if dim <= 0:
        return np.zeros((*x.shape[:-1], 0), dtype=np.float32) if x.ndim else np.zeros((0,), dtype=np.float32)
    cur = x.shape[-1] if x.ndim else 0
    if cur == dim:
        return x.astype(np.float32)
    if cur > dim:
        return x[..., :dim].astype(np.float32)
    pad_shape = list(x.shape)
    pad_shape[-1] = dim - cur
    return np.concatenate([x, np.zeros(pad_shape, dtype=np.float32)], axis=-1).astype(np.float32)


def safe_mean_pair_seq(x: np.ndarray) -> np.ndarray:
    """[P,T,D] -> [P,D], using mean over T. Empty last dim preserved."""
    x = np_clean(x)
    if x.ndim == 2:
        # [P,D]
        return x.astype(np.float32)
    if x.ndim == 3:
        return x.mean(axis=1).astype(np.float32)
    if x.ndim == 1:
        return x.reshape(1, -1).astype(np.float32)
    return x.reshape(x.shape[0], -1).astype(np.float32)


def pair_summary(pair: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pair = np_clean(pair)
    mask = np_clean(mask).reshape(-1)
    if pair.size == 0 or pair.shape[-1] == 0:
        return np.zeros((0,), dtype=np.float32)
    if pair.ndim != 2:
        pair = pair.reshape(pair.shape[0], -1)
    valid = mask[: pair.shape[0]] > 0
    if valid.any():
        x = pair[valid]
    else:
        x = pair
    if x.size == 0:
        return np.zeros((pair.shape[-1] * 2,), dtype=np.float32)
    return np.concatenate([x.mean(axis=0), x.std(axis=0)], axis=0).astype(np.float32)


def class_weights(y: np.ndarray, n: int, power: float = 0.5) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=n).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    w = (counts.sum() / counts) ** power
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def ccc_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(pred) == 0:
        return 0.0
    mp, mt = pred.mean(), target.mean()
    vp, vt = pred.var(), target.var()
    cov = ((pred - mp) * (target - mt)).mean()
    den = vp + vt + (mp - mt) ** 2
    return float((2 * cov) / den) if den > 1e-12 else 0.0


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    mp, mt = pred.mean(), target.mean()
    vp, vt = pred.var(unbiased=False), target.var(unbiased=False)
    cov = ((pred - mp) * (target - mt)).mean()
    ccc = (2 * cov) / (vp + vt + (mp - mt).pow(2) + 1e-8)
    return 1.0 - ccc


def soft_macro_f1_loss(logits: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    y1 = F.one_hot(y.long(), n_classes).float()
    tp = (p * y1).sum(0)
    fp = (p * (1 - y1)).sum(0)
    fn = ((1 - p) * y1).sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return 1.0 - f1.mean()


def expected_phq_from_prob(prob3: torch.Tensor, class_phq_means: torch.Tensor) -> torch.Tensor:
    return (prob3 * class_phq_means.to(prob3.device).view(1, 3)).sum(dim=-1)

# ----------------------- expert sample extraction -----------------------

EXPERTS = ["audio_big", "audio_official", "audio", "audio_controlled", "video", "gait", "p", "av"]


def extract_expert_arrays(s: Dict[str, Any], expert: str, use_p_embed: bool = False, use_official_gait: bool = False) -> Dict[str, Any]:
    """Convert a validated v12-loader sample into one expert-specific sample."""
    expert = expert.lower()
    # Base labels
    out: Dict[str, Any] = {
        "id": int(s["id"]),
        "label2": int(s.get("label2", 0)),
        "label3": int(s.get("label3", 0)),
        "phq": float(s.get("phq", 0.0)),
    }

    zero_pair = np.zeros((PAIR_COUNT, 0), dtype=np.float32)
    zero_mask = np.zeros(PAIR_COUNT, dtype=np.float32)
    zero_static = np.zeros((0,), dtype=np.float32)

    official_audio_pair = safe_mean_pair_seq(s.get("audio", np.zeros((PAIR_COUNT, 1, 0), dtype=np.float32)))
    official_audio_mask = np_clean(s.get("audio_pair_mask", np.zeros(PAIR_COUNT, dtype=np.float32))).reshape(-1)
    audio_big_pair = np_clean(s.get("audio_big", zero_pair)).reshape(PAIR_COUNT, -1) if np.asarray(s.get("audio_big", zero_pair)).size else zero_pair
    audio_big_mask = np_clean(s.get("audio_big_pair_mask", zero_mask)).reshape(-1)

    video_pair = safe_mean_pair_seq(s.get("video", np.zeros((PAIR_COUNT, 1, 0), dtype=np.float32)))
    video_mask = np_clean(s.get("video_pair_mask", zero_mask)).reshape(-1)
    vbeh_pair = np_clean(s.get("motion_extra_pair", zero_pair)).reshape(PAIR_COUNT, -1) if np.asarray(s.get("motion_extra_pair", zero_pair)).size else zero_pair
    vbeh_mask = np_clean(s.get("motion_extra_pair_mask", zero_mask)).reshape(-1)

    p_struct = np_clean(s.get("p_struct", zero_static)).reshape(-1)
    p_extra = np_clean(s.get("p_extra", zero_static)).reshape(-1)
    p_embed = np_clean(s.get("p_embed", zero_static)).reshape(-1)
    p_static = np.concatenate([p_struct, p_extra] + ([p_embed] if use_p_embed else []), axis=0).astype(np.float32)

    if expert == "audio_big":
        pair = audio_big_pair
        mask = audio_big_mask
        static = zero_static
    elif expert == "audio_official":
        pair = official_audio_pair
        mask = official_audio_mask
        static = zero_static
    elif expert == "audio":
        pair = np.concatenate([official_audio_pair, audio_big_pair], axis=-1)
        mask = np.maximum(official_audio_mask, audio_big_mask)
        static = zero_static
    elif expert == "audio_controlled":
        # A_big is main pair evidence; official audio is static reference summary.
        pair = audio_big_pair
        mask = audio_big_mask
        static = pair_summary(official_audio_pair, official_audio_mask)
    elif expert == "video":
        pair = np.concatenate([video_pair, vbeh_pair], axis=-1)
        mask = np.maximum(video_mask, vbeh_mask)
        static = np.concatenate([
            np_clean(s.get("motion_stat", zero_static)).reshape(-1),
            np_clean(s.get("motion_extra_stat", zero_static)).reshape(-1),
        ], axis=0).astype(np.float32)
    elif expert == "gait":
        static_parts = [np_clean(s.get("gait_extra", zero_static)).reshape(-1)]
        if use_official_gait:
            gait = np_clean(s.get("gait", np.zeros((1, 0), dtype=np.float32)))
            if gait.ndim == 2 and gait.shape[-1] > 0:
                static_parts.append(gait.mean(axis=0))
                static_parts.append(gait.std(axis=0))
        pair = zero_pair
        mask = zero_mask
        static = np.concatenate(static_parts, axis=0).astype(np.float32)
    elif expert == "p":
        pair = zero_pair
        mask = zero_mask
        static = p_static
    elif expert == "av":
        # Deeper AV expert candidate: pair-level interaction + P static.
        # Use common dims by pad/truncate to min for products.
        da = audio_big_pair.shape[-1]
        dv = vbeh_pair.shape[-1]
        dm = min(da, dv)
        if dm > 0:
            prod = audio_big_pair[:, :dm] * vbeh_pair[:, :dm]
            diff = np.abs(audio_big_pair[:, :dm] - vbeh_pair[:, :dm])
            pair = np.concatenate([audio_big_pair, vbeh_pair, diff, prod], axis=-1)
        else:
            pair = np.concatenate([audio_big_pair, vbeh_pair], axis=-1)
        mask = np.maximum(audio_big_mask, vbeh_mask)
        static = p_static
    else:
        raise ValueError(f"unknown expert={expert}; choices={EXPERTS}")

    out["pair"] = np_clean(pair).astype(np.float32)
    out["pair_mask"] = np_clean(mask).astype(np.float32)
    out["static"] = np_clean(static).astype(np.float32)
    return out


class ExpertScalers:
    def __init__(self) -> None:
        self.stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _flat(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0 or x.shape[-1] == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return x.reshape(-1, x.shape[-1])

    def fit(self, samples: Sequence[Dict[str, Any]]) -> "ExpertScalers":
        for k in ["pair", "static"]:
            mats = []
            for s in samples:
                x = np.asarray(s[k], dtype=np.float32)
                if x.size == 0 or x.shape[-1] == 0:
                    continue
                mats.append(self._flat(x))
            if mats:
                mat = np.concatenate(mats, axis=0)
                mu = mat.mean(axis=0).astype(np.float32)
                sd = mat.std(axis=0).astype(np.float32)
                sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
                self.stats[k] = (mu, sd)
        return self

    def transform(self, k: str, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if k not in self.stats or x.size == 0 or x.shape[-1] == 0:
            return x.astype(np.float32)
        mu, sd = self.stats[k]
        x = pad_last(x, len(mu))
        return np.clip((x - mu) / sd, -8.0, 8.0).astype(np.float32)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {k: {"mean": v[0].tolist(), "std": v[1].tolist()} for k, v in self.stats.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExpertScalers":
        obj = cls()
        for k, v in (d or {}).items():
            obj.stats[k] = (np.asarray(v["mean"], dtype=np.float32), np.asarray(v["std"], dtype=np.float32))
        return obj


class ExpertDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, Any]], scalers: Optional[ExpertScalers] = None) -> None:
        self.samples = list(samples)
        self.scalers = scalers or ExpertScalers()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "id": torch.tensor(int(s["id"]), dtype=torch.long),
            "label2": torch.tensor(int(s["label2"]), dtype=torch.long),
            "label3": torch.tensor(int(s["label3"]), dtype=torch.long),
            "phq": torch.tensor(float(s["phq"]), dtype=torch.float32),
            "pair": torch.tensor(self.scalers.transform("pair", s["pair"]), dtype=torch.float32),
            "pair_mask": torch.tensor(s["pair_mask"], dtype=torch.float32),
            "static": torch.tensor(self.scalers.transform("static", s["static"]), dtype=torch.float32),
        }


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}

# ----------------------------- model -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.enabled = self.in_dim > 0
        if self.enabled:
            self.net = nn.Sequential(
                nn.LayerNorm(self.in_dim),
                nn.Linear(self.in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(out_dim),
            )
        else:
            self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x.new_zeros((x.shape[0], self.out_dim))
        return self.net(x)


class PairEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.enabled = self.in_dim > 0
        self.frame = MLP(in_dim, hidden, hidden, dropout) if self.enabled else None
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, P = pair.shape[:2]
        if not self.enabled:
            return pair.new_zeros((B, self.hidden))
        z = self.frame(pair.reshape(B * P, pair.shape[-1])).reshape(B, P, self.hidden)
        m = mask.float()
        if m.shape[1] != P:
            m = torch.ones((B, P), device=pair.device, dtype=torch.float32)
        score = self.score(z).squeeze(-1).masked_fill(m <= 0, -1e4)
        w = torch.softmax(score, dim=1)
        bad = (m.sum(dim=1, keepdim=True) <= 0)
        w = torch.where(bad, torch.zeros_like(w), w)
        return (w.unsqueeze(-1) * z).sum(dim=1)


class ExpertModel(nn.Module):
    def __init__(self, pair_dim: int, static_dim: int, class_phq_means: Sequence[float], hidden: int = 96, dropout: float = 0.35, phq_resid_scale: float = 2.5) -> None:
        super().__init__()
        self.pair_dim = int(pair_dim)
        self.static_dim = int(static_dim)
        self.hidden = int(hidden)
        self.phq_resid_scale = float(phq_resid_scale)
        self.register_buffer("class_phq_means", torch.tensor(class_phq_means, dtype=torch.float32))
        self.pair_enc = PairEncoder(pair_dim, hidden, dropout)
        self.static_enc = MLP(static_dim, hidden, hidden, dropout)
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        self.head3 = nn.Linear(hidden, 3)
        self.resid = nn.Linear(hidden, 1)

    def forward(self, pair: torch.Tensor, static: torch.Tensor, pair_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        hp = self.pair_enc(pair, pair_mask)
        hs = self.static_enc(static)
        h = self.fuse(torch.cat([hp, hs], dim=-1))
        logits3 = self.head3(h)
        prob3 = torch.softmax(logits3, dim=-1)
        bin_logits = torch.stack([logits3[:, 0], torch.logsumexp(logits3[:, 1:3], dim=1)], dim=1)
        base_phq = expected_phq_from_prob(prob3, self.class_phq_means)
        resid = torch.tanh(self.resid(h).squeeze(-1)) * self.phq_resid_scale
        phq_pred = (base_phq + resid).clamp(0.0, 27.0)
        return {"logits3": logits3, "logits2": bin_logits, "phq_pred": phq_pred}

# ----------------------------- metrics/predict -----------------------------

def eval_pred(ids: np.ndarray, y2: np.ndarray, y3: np.ndarray, phq: np.ndarray, prob2: np.ndarray, prob3: np.ndarray, phq_pred: np.ndarray) -> Dict[str, float]:
    pred3 = prob3.argmax(axis=1).astype(int)
    pred2 = (pred3 > 0).astype(int)
    out = {
        "binary_acc": float(accuracy_score(y2, pred2)),
        "binary_macro_f1": float(f1_score(y2, pred2, average="macro", zero_division=0)),
        "binary_kappa": float(cohen_kappa_score(y2, pred2)),
        "ternary_acc": float(accuracy_score(y3, pred3)),
        "ternary_macro_f1": float(f1_score(y3, pred3, average="macro", zero_division=0)),
        "ternary_kappa": float(cohen_kappa_score(y3, pred3)),
        "phq_ccc": ccc_np(phq_pred, phq),
        "phq_mae": float(mean_absolute_error(phq, phq_pred)),
        "phq_rmse": float(math.sqrt(mean_squared_error(phq, phq_pred))),
        "inconsistent": 0.0,
    }
    return out


@torch.no_grad()
def predict_loader(model: ExpertModel, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    ids, y2s, y3s, phqs, prob2s, prob3s, phqps = [], [], [], [], [], [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out = model(b["pair"], b["static"], b["pair_mask"])
        p3 = torch.softmax(out["logits3"], dim=-1)
        p2_pos = p3[:, 1] + p3[:, 2]
        p2 = torch.stack([1.0 - p2_pos, p2_pos], dim=-1)
        ids.append(b["id"].cpu().numpy())
        y2s.append(b["label2"].cpu().numpy())
        y3s.append(b["label3"].cpu().numpy())
        phqs.append(b["phq"].cpu().numpy())
        prob2s.append(p2.cpu().numpy())
        prob3s.append(p3.cpu().numpy())
        phqps.append(out["phq_pred"].cpu().numpy())
    return {
        "ids": np.concatenate(ids),
        "y2": np.concatenate(y2s),
        "y3": np.concatenate(y3s),
        "phq": np.concatenate(phqs),
        "prob2": np.concatenate(prob2s),
        "prob3": np.concatenate(prob3s),
        "phq_pred": np.concatenate(phqps),
    }


def compute_loss(out: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], args: argparse.Namespace, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    y2, y3, phq = b["label2"], b["label3"], b["phq"]
    device = out["logits3"].device
    w2, w3 = w2.to(device), w3.to(device)
    loss = F.cross_entropy(out["logits3"], y3, weight=w3, label_smoothing=args.label_smoothing)
    loss = loss + args.binary_weight * F.cross_entropy(out["logits2"], y2, weight=w2, label_smoothing=args.label_smoothing)
    if args.soft_f1_weight > 0:
        loss = loss + args.soft_f1_weight * soft_macro_f1_loss(out["logits3"], y3, 3)
    if args.reg_weight > 0:
        loss = loss + args.reg_weight * F.smooth_l1_loss(out["phq_pred"], phq)
    if args.ccc_weight > 0:
        loss = loss + args.ccc_weight * ccc_loss(out["phq_pred"], phq)
    return loss


def make_prediction_frames(pred: Dict[str, np.ndarray], has_labels: bool = True) -> pd.DataFrame:
    ids = pred["ids"].astype(int)
    prob3 = pred["prob3"]
    pred3 = prob3.argmax(axis=1).astype(int)
    pred2 = (pred3 > 0).astype(int)
    df = pd.DataFrame({
        "id": ids,
        "pred2": pred2,
        "pred3": pred3,
        "phq_pred": pred["phq_pred"],
        "prob3_0": prob3[:, 0],
        "prob3_1": prob3[:, 1],
        "prob3_2": prob3[:, 2],
    })
    if has_labels:
        df["y2"] = pred["y2"].astype(int)
        df["y3"] = pred["y3"].astype(int)
        df["phq"] = pred["phq"].astype(float)
    return df.sort_values("id").reset_index(drop=True)


def average_preds(preds: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    ids = preds[0]["ids"]
    order = np.argsort(ids)
    ids_sorted = ids[order]
    # assume same order from deterministic test dataset; still align by id for safety
    prob3_acc, phq_acc = [], []
    for p in preds:
        idx = np.argsort(p["ids"])
        if not np.array_equal(p["ids"][idx], ids_sorted):
            raise ValueError("prediction id mismatch when averaging")
        prob3_acc.append(p["prob3"][idx])
        phq_acc.append(p["phq_pred"][idx])
    out = {k: preds[0][k][order] for k in ["ids", "y2", "y3", "phq"]}
    out["prob3"] = np.mean(prob3_acc, axis=0)
    p2pos = out["prob3"][:, 1] + out["prob3"][:, 2]
    out["prob2"] = np.stack([1 - p2pos, p2pos], axis=1)
    out["phq_pred"] = np.mean(phq_acc, axis=0)
    return out


def write_submission(test_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    b = pd.DataFrame({"id": test_df["id"].astype(int), "binary_pred": test_df["pred2"].astype(int)})
    t = pd.DataFrame({"id": test_df["id"].astype(int), "ternary_pred": test_df["pred3"].astype(int)})
    b.to_csv(out_dir / "binary.csv", index=False)
    t.to_csv(out_dir / "ternary.csv", index=False)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "binary.csv", arcname="binary.csv")
        z.write(out_dir / "ternary.csv", arcname="ternary.csv")
    dist = {
        "n": int(len(test_df)),
        "binary": {str(k): int(v) for k, v in b["binary_pred"].value_counts().sort_index().items()},
        "ternary": {str(k): int(v) for k, v in t["ternary_pred"].value_counts().sort_index().items()},
        "severe": t.loc[t["ternary_pred"].eq(2), "id"].astype(int).tolist(),
        "positive": b.loc[b["binary_pred"].eq(1), "id"].astype(int).tolist(),
        "inconsistent": int(((b["binary_pred"].to_numpy() == 0) != (t["ternary_pred"].to_numpy() == 0)).sum()),
    }
    write_json(dist, out_dir.parent / "distribution_report.json")
    return dist

# ----------------------------- data building -----------------------------

def make_store(args: argparse.Namespace, rows: pd.DataFrame, split: str, forced_dims: Optional[Dict[str, int]] = None) -> Any:
    return v12.make_store_from_args(args, rows, split, forced_dims=forced_dims)


def build_expert_samples(args: argparse.Namespace, expert: str, split: str, forced_dims: Optional[Dict[str, int]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, int]]:
    if split == "train":
        rows = v12.prepare_train_rows(args.train_split_csv)
        if args.smoke:
            rows = rows.head(min(len(rows), args.smoke_n)).copy()
        store = make_store(args, rows, "train", forced_dims=forced_dims)
    else:
        rows = v12.prepare_test_rows(args.test_split_csv)
        store = make_store(args, rows, "test", forced_dims=forced_dims)
    full_samples = [store.make_sample(r) for _, r in tqdm(rows.iterrows(), total=len(rows), desc=f"build {split} v12 samples")]
    ex_samples = [extract_expert_arrays(s, expert, use_p_embed=bool(args.use_p_embed_for_p_expert), use_official_gait=bool(args.use_official_gait)) for s in full_samples]
    report = store.report()
    report["expert"] = expert
    report["expert_pair_dim"] = int(ex_samples[0]["pair"].shape[-1]) if ex_samples else 0
    report["expert_static_dim"] = int(ex_samples[0]["static"].shape[-1]) if ex_samples else 0
    return ex_samples, report, store.dims.to_dict()

# ----------------------------- commands -----------------------------

def inspect_command(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output_dir)
    rows = v12.prepare_train_rows(args.train_split_csv)
    if args.smoke:
        rows = rows.head(min(len(rows), args.smoke_n)).copy()
    store = make_store(args, rows, "train")
    write_json(store.report(), out / "v12_loader_report_train.json")
    print("[V12 loader dims]", store.dims.to_dict())
    for expert in args.expert_list.split(","):
        expert = expert.strip()
        if not expert:
            continue
        samples, report, _ = build_expert_samples(args, expert, "train")
        print(f"[EXPERT] {expert}: pair_dim={report['expert_pair_dim']} static_dim={report['expert_static_dim']} n={len(samples)}")


def train_expert(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    out_dir = ensure_dir(args.output_dir)
    expert = args.expert
    train_samples, report_train, forced_dims = build_expert_samples(args, expert, "train")
    write_json(report_train, out_dir / "feature_report_train.json")
    test_samples, report_test, _ = build_expert_samples(args, expert, "test", forced_dims=forced_dims)
    write_json(report_test, out_dir / "feature_report_test.json")
    pair_dim = int(report_train["expert_pair_dim"])
    static_dim = int(report_train["expert_static_dim"])
    print(f"[INFO] expert={expert} pair_dim={pair_dim} static_dim={static_dim} n_train={len(train_samples)} n_test={len(test_samples)}")
    y3_all = np.asarray([s["label3"] for s in train_samples], dtype=int)
    y2_all = np.asarray([s["label2"] for s in train_samples], dtype=int)
    phq_all = np.asarray([s["phq"] for s in train_samples], dtype=np.float32)
    class_means = []
    global_mean = float(phq_all.mean()) if len(phq_all) else 3.0
    for c in range(3):
        vals = phq_all[y3_all == c]
        class_means.append(float(vals.mean()) if len(vals) else global_mean)
    # enforce weak monotonicity
    class_means = [class_means[0], max(class_means[1], class_means[0] + 0.5), max(class_means[2], class_means[1] + 1.0)]
    print("[INFO] labels3", {int(k): int(v) for k, v in zip(*np.unique(y3_all, return_counts=True))}, "class_phq_means", class_means)
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.folds = min(args.folds, 2)
        print(f"[SMOKE] folds={args.folds} epochs={args.epochs}")
    w2 = class_weights(y2_all, 2, power=args.class_weight_power)
    w3 = class_weights(y3_all, 3, power=args.class_weight_power)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    fold_rows = []
    oof_parts = []
    test_preds = []
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train_samples)), y3_all), start=1):
        seed_everything(args.seed + fold)
        tr = [train_samples[i] for i in tr_idx]
        va = [train_samples[i] for i in va_idx]
        scalers = ExpertScalers().fit(tr)
        tr_loader = DataLoader(ExpertDataset(tr, scalers), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
        va_loader = DataLoader(ExpertDataset(va, scalers), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
        model = ExpertModel(pair_dim, static_dim, class_means, hidden=args.hidden_dim, dropout=args.dropout, phq_resid_scale=args.phq_resid_scale).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
        best_score = -1e9
        best_state = None
        best_metrics = None
        bad = 0
        print(f"\n[expert={expert}][fold={fold} seed={args.seed}] train={len(tr)} val={len(va)}")
        for ep in range(1, args.epochs + 1):
            model.train()
            losses = []
            for b in tr_loader:
                b = {k: v.to(device) for k, v in b.items()}
                opt.zero_grad(set_to_none=True)
                out = model(b["pair"], b["static"], b["pair_mask"])
                loss = compute_loss(out, b, args, w2, w3)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            sched.step()
            pred = predict_loader(model, va_loader, device)
            metrics = eval_pred(pred["ids"], pred["y2"], pred["y3"], pred["phq"], pred["prob2"], pred["prob3"], pred["phq_pred"])
            score = metrics["ternary_macro_f1"] + metrics["ternary_kappa"] + 0.6 * metrics["binary_macro_f1"] + 0.3 * metrics["phq_ccc"]
            if score > best_score + 1e-8:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = metrics
                bad = 0
            else:
                bad += 1
            if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
                print(f"[fold={fold}] ep={ep:03d} loss={np.mean(losses):.4f} tF1={metrics['ternary_macro_f1']:.4f} tK={metrics['ternary_kappa']:.4f} bF1={metrics['binary_macro_f1']:.4f} CCC={metrics['phq_ccc']:.4f} score={score:.4f}")
            if args.patience > 0 and bad >= args.patience:
                print(f"[fold={fold}] early stop at ep={ep}, best_score={best_score:.4f}")
                break
        assert best_state is not None and best_metrics is not None
        model.load_state_dict(best_state)
        va_pred = predict_loader(model, va_loader, device)
        df_va = make_prediction_frames(va_pred, has_labels=True)
        df_va["fold"] = fold
        df_va["seed"] = args.seed
        oof_parts.append(df_va)
        row = {"fold": fold, "seed": args.seed, "best_score": best_score, **best_metrics}
        fold_rows.append(row)
        torch.save({
            "model_state": best_state,
            "pair_dim": pair_dim,
            "static_dim": static_dim,
            "class_phq_means": class_means,
            "scalers": scalers.to_dict(),
            "args": {k: v for k, v in vars(args).items() if k != "func" and not callable(v)},
            "metrics": row,
        }, ckpt_dir / f"fold{fold}_seed{args.seed}.pt")
        # Test prediction for this fold model
        te_loader = DataLoader(ExpertDataset(test_samples, scalers), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
        test_preds.append(predict_loader(model, te_loader, device))
        print("[fold best]", row)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    oof = pd.concat(oof_parts, ignore_index=True).sort_values("id")
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    oof_metrics = eval_pred(oof["id"].to_numpy(), oof["y2"].to_numpy(), oof["y3"].to_numpy(), oof["phq"].to_numpy(),
                            np.stack([1 - (oof["prob3_1"].to_numpy()+oof["prob3_2"].to_numpy()), oof["prob3_1"].to_numpy()+oof["prob3_2"].to_numpy()], axis=1),
                            oof[["prob3_0", "prob3_1", "prob3_2"]].to_numpy(), oof["phq_pred"].to_numpy())
    write_json(oof_metrics, out_dir / "oof_metrics.json")
    print("[OOF]", oof_metrics)
    test_avg = average_preds(test_preds)
    test_df = make_prediction_frames(test_avg, has_labels=False)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    dist = write_submission(test_df, out_dir / "predictions_normal")
    print("[TEST]", dist)
    print("[OK] saved to", out_dir)


def dummy_command(args: argparse.Namespace) -> None:
    seed_everything(0)
    samples = []
    for i in range(24):
        y3 = i % 3
        y2 = int(y3 > 0)
        full = {
            "id": i, "label2": y2, "label3": y3, "phq": float([1,4,10][y3]),
            "audio": np.random.randn(PAIR_COUNT, 8, 13).astype(np.float32), "audio_pair_mask": np.ones(PAIR_COUNT, dtype=np.float32),
            "audio_big": np.random.randn(PAIR_COUNT, 16).astype(np.float32), "audio_big_pair_mask": np.ones(PAIR_COUNT, dtype=np.float32),
            "video": np.random.randn(PAIR_COUNT, 8, 5).astype(np.float32), "video_pair_mask": np.ones(PAIR_COUNT, dtype=np.float32),
            "motion_extra_pair": np.random.randn(PAIR_COUNT, 7).astype(np.float32), "motion_extra_pair_mask": np.ones(PAIR_COUNT, dtype=np.float32),
            "motion_stat": np.random.randn(4).astype(np.float32), "motion_extra_stat": np.random.randn(6).astype(np.float32),
            "gait": np.random.randn(8, 3).astype(np.float32), "gait_extra": np.random.randn(9).astype(np.float32),
            "p_struct": np.random.randn(2).astype(np.float32), "p_extra": np.random.randn(4).astype(np.float32), "p_embed": np.random.randn(8).astype(np.float32),
        }
        samples.append(extract_expert_arrays(full, args.expert, use_p_embed=True, use_official_gait=True))
    pair_dim = samples[0]["pair"].shape[-1]
    static_dim = samples[0]["static"].shape[-1]
    scalers = ExpertScalers().fit(samples[:12])
    ds = ExpertDataset(samples, scalers)
    dl = DataLoader(ds, batch_size=4, collate_fn=collate)
    model = ExpertModel(pair_dim, static_dim, [1,4,10], hidden=16, dropout=0.1)
    b = next(iter(dl))
    out = model(b["pair"], b["static"], b["pair_mask"])
    print("[DUMMY OK]", args.expert, "pair_dim", pair_dim, "static_dim", static_dim, "logits", tuple(out["logits3"].shape))

# ----------------------------- argparse -----------------------------

def add_common(p: argparse.ArgumentParser) -> None:
    # v12 loader args
    p.add_argument("--train_data_root", default="")
    p.add_argument("--test_data_root", default="")
    p.add_argument("--train_split_csv", default="")
    p.add_argument("--test_split_csv", default="")
    p.add_argument("--p_struct_train_csv", default="")
    p.add_argument("--p_struct_test_csv", default="")
    p.add_argument("--p_embed_npy", default="")
    p.add_argument("--p_embed_test_npy", default="")
    p.add_argument("--motion_train_npz", default="")
    p.add_argument("--motion_test_npz", default="")
    p.add_argument("--audio_big_train_npz", default="")
    p.add_argument("--audio_big_test_npz", default="")
    p.add_argument("--motion_extra_train_npz", default="")
    p.add_argument("--motion_extra_test_npz", default="")
    p.add_argument("--gait_extra_train_npz", default="")
    p.add_argument("--gait_extra_test_npz", default="")
    p.add_argument("--p_extra_train_csv", default="")
    p.add_argument("--p_extra_test_csv", default="")
    p.add_argument("--audio_features", default="wav2vec,opensmile")
    p.add_argument("--official_video_features", default="")
    p.add_argument("--use_gait", type=int, default=1)
    p.add_argument("--target_t", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    # expert flags
    p.add_argument("--expert", default="audio_big", choices=EXPERTS)
    p.add_argument("--expert_list", default="audio_big,audio_official,audio,audio_controlled,video,gait,p")
    p.add_argument("--use_p_embed_for_p_expert", type=int, default=0)
    p.add_argument("--use_official_gait", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_n", type=int, default=24)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("inspect")
    add_common(p)
    p.set_defaults(func=inspect_command)
    p = sub.add_parser("train")
    add_common(p)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--binary_weight", type=float, default=0.6)
    p.add_argument("--soft_f1_weight", type=float, default=0.15)
    p.add_argument("--reg_weight", type=float, default=0.20)
    p.add_argument("--ccc_weight", type=float, default=0.10)
    p.add_argument("--class_weight_power", type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--phq_resid_scale", type=float, default=2.5)
    p.add_argument("--log_every", type=int, default=5)
    p.set_defaults(func=train_expert)
    p = sub.add_parser("dummy")
    p.add_argument("--expert", default="audio_controlled", choices=EXPERTS)
    p.set_defaults(func=dummy_command)
    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
