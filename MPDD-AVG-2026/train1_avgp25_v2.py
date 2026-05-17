from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, r2_score
from torch.utils.data import DataLoader

from dataset import REGRESSION_TASK, MPDDElderDataset, collate_batch, infer_input_dims, resolve_project_path
from models import ALL_ENCODERS, MODEL_TYPES, build_model
from train_val_split import create_train_val_split

PROJECT_ROOT = Path(__file__).resolve().parent
SUBTRACK_LOG_DIRS = {
    "A-V+P": "A-V-P",
    "A-V-G+P": "A-V-G+P",
    "G+P": "G-P",
}
METRIC_ARRAY_KEYS = {"ids", "y_true", "y_pred", "class_true", "class_pred", "phq_true", "phq_pred"}
PATH_ARG_KEYS = {"config", "data_root", "split_csv", "personality_npy", "checkpoints_dir", "logs_dir"}


class FocalLoss(nn.Module):
    """Multi-class focal loss using class-weighted CE as the base loss."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()
    var_pred = y_pred.var()
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom <= 1e-12:
        return 0.0
    return float((2.0 * cov) / denom)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(resolve_project_path(config_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def str2bool_int(value: int | str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean-like value, got {value}")


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MPDD-AVG baseline/DepFormer with AVG-P gate and CE+Focal+MSE loss.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--track", default=defaults["track"], choices=["Track1", "Track2"])
    parser.add_argument("--task", default=defaults["task"], choices=["binary", "ternary", REGRESSION_TASK])
    parser.add_argument("--regression_label", default=defaults.get("regression_label", "label2"), choices=["label2", "label3"])
    parser.add_argument("--subtrack", default=defaults["subtrack"], choices=["A-V+P", "A-V-G+P", "G+P"])

    parser.add_argument("--model_type", default=defaults.get("model_type", "baseline"), choices=MODEL_TYPES)
    parser.add_argument("--encoder_type", default=defaults["encoder_type"], choices=ALL_ENCODERS)
    parser.add_argument("--num_bct_layers", type=int, default=defaults.get("num_bct_layers", 1))
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 2))
    parser.add_argument("--ffn_mult", type=int, default=defaults.get("ffn_mult", 4))

    # New minimal-change DepFormer switches.
    parser.add_argument("--loss_type", default=defaults.get("loss_type", "ce_focal_mse"), choices=["ce_mse", "ce_focal_mse"])
    # For binary/ternary tasks, select checkpoints by the competition-aware score by default.
    # REGRESSION_TASK is unchanged: it still selects by CCC.
    parser.add_argument("--selection_mode", default=defaults.get("selection_mode", "score"),
                        choices=["score", "f1", "acc", "kappa", "ccc", "loss"])
    parser.add_argument("--label_smoothing", type=float, default=defaults.get("label_smoothing", 0.0))
    parser.add_argument("--focal_gamma", type=float, default=defaults.get("focal_gamma", 2.0))
    parser.add_argument("--focal_lambda", type=float, default=defaults.get("focal_lambda", 1.0))
    parser.add_argument("--reg_lambda", type=float, default=defaults.get("reg_lambda", 1.0))
    parser.add_argument("--force_regression_head", type=str2bool_int, default=defaults.get("force_regression_head", True))
    parser.add_argument("--use_p_gate", type=str2bool_int, default=defaults.get("use_p_gate", True))
    parser.add_argument("--av_encode_pairwise", type=str2bool_int, default=defaults.get("av_encode_pairwise", True))

    parser.add_argument("--audio_feature", default=defaults["audio_feature"])
    parser.add_argument("--video_feature", default=defaults["video_feature"])
    parser.add_argument("--data_root", default=defaults["data_root"])
    parser.add_argument("--split_csv", default=defaults["split_csv"])
    parser.add_argument("--personality_npy", default=defaults["personality_npy"])
    parser.add_argument("--val_ratio", type=float, default=defaults["val_ratio"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--target_t", type=int, default=defaults["target_t"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--hidden_dim", type=int, default=defaults["hidden_dim"])
    parser.add_argument("--dropout", type=float, default=defaults["dropout"])
    parser.add_argument("--patience", type=int, default=defaults["patience"])
    parser.add_argument("--min_delta", type=float, default=defaults["min_delta"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--checkpoints_dir", default=defaults["checkpoints_dir"])
    parser.add_argument("--logs_dir", default=defaults["logs_dir"])
    parser.add_argument("--experiment_name", default="")
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default="config.json")
    known_args, _ = base_parser.parse_known_args()
    defaults = load_config(known_args.config)
    parser = build_parser(defaults)
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"mpdd_train_{log_file.stem}_{time.time_ns()}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def resolve_track_task_dir(root: Path, track: str, subtrack: str, task: str, experiment_name: str) -> Path:
    subtrack_dir = SUBTRACK_LOG_DIRS.get(subtrack, subtrack.replace("+", "-"))
    return root / track / subtrack_dir / task / experiment_name


def to_project_relative_path(path_like: str | Path) -> str:
    path = resolve_project_path(path_like)
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def normalize_path_args(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if key in PATH_ARG_KEYS and value not in (None, ""):
            normalized[key] = to_project_relative_path(value)
        else:
            normalized[key] = value
    return normalized


def build_experiment_name(args: argparse.Namespace) -> str:
    feature_tag = "gait_only" if args.subtrack == "G+P" else f"{args.audio_feature}__{args.video_feature}"
    model_tag = "" if args.model_type == "baseline" else f"{args.model_type}_"
    loss_tag = args.loss_type
    p_gate_tag = "pgate" if args.use_p_gate else "nopgate"
    pair_tag = "pairav" if args.av_encode_pairwise else "avgav"
    h_tag = f"h{args.hidden_dim}"
    if args.task == REGRESSION_TASK:
        return args.experiment_name or (
            f"{model_tag}{args.track.lower()}_{args.task}_{args.regression_label}_"
            f"{args.subtrack}_{args.encoder_type}_{feature_tag}_{loss_tag}_{p_gate_tag}_{pair_tag}_{h_tag}"
        )
    return args.experiment_name or (
        f"{model_tag}{args.track.lower()}_{args.task}_{args.subtrack}_{args.encoder_type}_"
        f"{feature_tag}_{loss_tag}_{p_gate_tag}_{pair_tag}_{h_tag}"
    )


def get_num_classes(task: str, regression_label: str) -> int:
    if task == "binary":
        return 2
    if task == "ternary":
        return 3
    if task == REGRESSION_TASK:
        return 2 if regression_label == "label2" else 3
    raise ValueError(f"Unsupported task: {task}")


def build_class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def append_summary_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in METRIC_ARRAY_KEYS}


def get_selection_metric_name(task: str, selection_mode: str = "score") -> str:
    if task == REGRESSION_TASK:
        return "ccc"
    return selection_mode


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "audio": batch["audio"].to(device) if "audio" in batch else None,
        "video": batch["video"].to(device) if "video" in batch else None,
        "gait": batch["gait"].to(device) if "gait" in batch else None,
        "personality": batch["personality"].to(device),
        "pair_mask": batch["pair_mask"].to(device) if "pair_mask" in batch else None,
        "label": batch["label"].to(device),
        "phq9": batch["phq9"].to(device),
    }


def forward_model(model: nn.Module, batch_dev: dict[str, Any], return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, ...]:
    kwargs = {
        "audio": batch_dev["audio"],
        "video": batch_dev["video"],
        "gait": batch_dev["gait"],
        "personality": batch_dev["personality"],
        "pair_mask": batch_dev["pair_mask"],
    }
    if return_aux:
        signature = inspect.signature(model.forward)
        if "return_aux" in signature.parameters:
            kwargs["return_aux"] = True
    return model(**kwargs)


def unpack_outputs(outputs: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if not isinstance(outputs, tuple):
        return outputs, None, None
    logits = outputs[0]
    reg_out: torch.Tensor | None = None
    focal_logits: torch.Tensor | None = None
    if len(outputs) >= 2:
        # By convention in depformer_avp.py, the second output is regression when enabled.
        if outputs[1].ndim == 1 or (outputs[1].ndim == 2 and outputs[1].shape[-1] == 1):
            reg_out = outputs[1].reshape(-1)
        else:
            focal_logits = outputs[1]
    if len(outputs) >= 3:
        focal_logits = outputs[2]
    return logits, reg_out, focal_logits


def compute_total_loss(
    outputs: torch.Tensor | tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    phq9: torch.Tensor,
    criterion_cls: nn.Module,
    criterion_reg: nn.Module,
    criterion_focal: nn.Module | None,
    focal_lambda: float,
    reg_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, reg_out, focal_logits = unpack_outputs(outputs)
    cls_loss = criterion_cls(logits, labels)
    total_loss = cls_loss
    reg_loss = torch.zeros((), dtype=cls_loss.dtype, device=cls_loss.device)
    focal_loss = torch.zeros((), dtype=cls_loss.dtype, device=cls_loss.device)

    if reg_out is not None:
        reg_loss = criterion_reg(reg_out, phq9.float())
        total_loss = total_loss + reg_lambda * reg_loss
    if criterion_focal is not None and focal_logits is not None:
        focal_loss = criterion_focal(focal_logits, labels)
        total_loss = total_loss + focal_lambda * focal_loss

    return total_loss, {
        "cls_loss": float(cls_loss.detach().cpu().item()),
        "reg_loss": float(reg_loss.detach().cpu().item()),
        "focal_loss": float(focal_loss.detach().cpu().item()),
    }


@torch.no_grad()
def evaluate_model_avgp25(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    criterion_cls: nn.Module,
    criterion_reg: nn.Module,
    criterion_focal: nn.Module | None,
    focal_lambda: float,
    reg_lambda: float,
    selection_mode: str = "score",
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_focal_loss = 0.0
    total_count = 0

    y_true: list[int] = []
    y_pred: list[int] = []
    phq_true: list[float] = []
    phq_pred: list[float] = []
    ids: list[Any] = []

    for batch in loader:
        batch_dev = move_batch_to_device(batch, device)
        labels = batch_dev["label"]
        phq9 = batch_dev["phq9"]
        outputs = forward_model(model, batch_dev, return_aux=True)
        logits, reg_out, _ = unpack_outputs(outputs)
        loss, loss_parts = compute_total_loss(
            outputs,
            labels,
            phq9,
            criterion_cls,
            criterion_reg,
            criterion_focal,
            focal_lambda=focal_lambda,
            reg_lambda=reg_lambda,
        )

        batch_size = int(labels.numel())
        total_count += batch_size
        total_loss += float(loss.item()) * batch_size
        total_cls_loss += loss_parts["cls_loss"] * batch_size
        total_reg_loss += loss_parts["reg_loss"] * batch_size
        total_focal_loss += loss_parts["focal_loss"] * batch_size

        preds = torch.argmax(logits, dim=-1)
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())
        phq_true.extend(phq9.detach().cpu().numpy().astype(float).tolist())
        if reg_out is not None:
            phq_pred.extend(reg_out.detach().cpu().numpy().astype(float).tolist())
        else:
            phq_pred.extend([0.0] * batch_size)
        if "id" in batch:
            ids.extend(batch["id"])
        elif "ids" in batch:
            ids.extend(batch["ids"])

    denom = max(1, total_count)
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    phq_true_np = np.asarray(phq_true, dtype=np.float64)
    phq_pred_np = np.asarray(phq_pred, dtype=np.float64)

    acc = float(accuracy_score(y_true_np, y_pred_np)) if y_true_np.size else 0.0
    f1 = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)) if y_true_np.size else 0.0
    try:
        kappa = float(cohen_kappa_score(y_true_np, y_pred_np)) if y_true_np.size else 0.0
    except Exception:
        kappa = 0.0
    rmse = float(np.sqrt(np.mean((phq_true_np - phq_pred_np) ** 2))) if phq_true_np.size else 0.0
    mae = float(np.mean(np.abs(phq_true_np - phq_pred_np))) if phq_true_np.size else 0.0
    ccc = concordance_ccc(phq_true_np, phq_pred_np)
    try:
        r2 = float(r2_score(phq_true_np, phq_pred_np)) if phq_true_np.size >= 2 else 0.0
    except Exception:
        r2 = 0.0

    loss_avg = total_loss / denom
    score = float((f1 + kappa + ccc) / 3.0)
    if task == REGRESSION_TASK:
        selection_score = ccc
    elif selection_mode == "score":
        selection_score = score
    elif selection_mode == "f1":
        selection_score = f1
    elif selection_mode == "acc":
        selection_score = acc
    elif selection_mode == "kappa":
        selection_score = kappa
    elif selection_mode == "ccc":
        selection_score = ccc
    elif selection_mode == "loss":
        selection_score = -float(loss_avg)
    else:
        raise ValueError(f"Unsupported selection_mode={selection_mode}")

    metrics = {
        "loss": loss_avg,
        "cls_loss": total_cls_loss / denom,
        "reg_loss": total_reg_loss / denom,
        "focal_loss": total_focal_loss / denom,
        "f1": f1,
        "acc": acc,
        "kappa": kappa,
        "ccc": ccc,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "score": score,
        "selection_score": selection_score,
        "ids": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "class_true": y_true,
        "class_pred": y_pred,
        "phq_true": phq_true,
        "phq_pred": phq_pred,
    }
    return metrics


def main() -> None:
    args = parse_args()
    experiment_name = build_experiment_name(args)
    timestamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())

    checkpoints_root = resolve_project_path(args.checkpoints_dir)
    logs_root = resolve_project_path(args.logs_dir)
    checkpoints_dir = resolve_track_task_dir(checkpoints_root, args.track, args.subtrack, args.task, experiment_name)
    log_dir = resolve_track_task_dir(logs_root, args.track, args.subtrack, args.task, experiment_name)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir / f"result_{timestamp}.log")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    split_payload = create_train_val_split(
        split_csv=args.split_csv,
        task=args.task,
        val_ratio=args.val_ratio,
        regression_label=args.regression_label,
    )
    setup_seed(args.seed)

    use_regression_head = bool(args.force_regression_head)
    use_focal_head = args.loss_type == "ce_focal_mse"
    is_regression_task = args.task == REGRESSION_TASK

    train_dataset = MPDDElderDataset(
        data_root=args.data_root,
        label_map=split_payload["train_map"],
        source_split_map=split_payload["source_split_map"],
        subtrack=args.subtrack,
        task=args.task,
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        personality_npy=args.personality_npy,
        phq_map=split_payload.get("train_phq_map"),
        target_t=args.target_t,
    )
    val_dataset = MPDDElderDataset(
        data_root=args.data_root,
        label_map=split_payload["val_map"],
        source_split_map=split_payload["source_split_map"],
        subtrack=args.subtrack,
        task=args.task,
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        personality_npy=args.personality_npy,
        phq_map=split_payload.get("val_phq_map"),
        target_t=args.target_t,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    input_dims = infer_input_dims(train_dataset)
    num_classes = get_num_classes(args.task, args.regression_label)
    model_kwargs = {
        "subtrack": args.subtrack,
        "num_classes": num_classes,
        "is_regression": False,
        "use_regression_head": use_regression_head,
        "audio_dim": input_dims["audio_dim"],
        "video_dim": input_dims["video_dim"],
        "gait_dim": input_dims["gait_dim"],
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "encoder_type": args.encoder_type,
    }
    if args.model_type == "depformer":
        model_kwargs.update(
            {
                "num_bct_layers": args.num_bct_layers,
                "num_heads": args.num_heads,
                "ffn_mult": args.ffn_mult,
                "use_p_gate": bool(args.use_p_gate),
                "use_focal_head": use_focal_head,
                "av_encode_pairwise": bool(args.av_encode_pairwise),
            }
        )
    model = build_model(model_type=args.model_type, **model_kwargs).to(device)

    class_weights = build_class_weights(
        [int(sample["label"]) for sample in train_dataset.samples],
        num_classes=num_classes,
        device=device,
    )
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    criterion_reg = nn.MSELoss()
    criterion_focal = FocalLoss(weight=class_weights, gamma=args.focal_gamma) if use_focal_head else None
    selection_metric_name = get_selection_metric_name(args.task, args.selection_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    logger.info("Experiment: %s", experiment_name)
    logger.info("Model type: %s | encoder: %s", args.model_type, args.encoder_type)
    logger.info(
        "Loss: %s | focal_lambda=%.4f | reg_lambda=%.4f | force_regression_head=%s",
        args.loss_type,
        args.focal_lambda,
        args.reg_lambda,
        use_regression_head,
    )
    logger.info("AVG-P gate: %s | A/V pairwise encoding: %s", bool(args.use_p_gate), bool(args.av_encode_pairwise))
    logger.info("Selection mode: %s | label_smoothing=%.4f", args.selection_mode, args.label_smoothing)
    logger.info("Device: %s", device)
    logger.info("Input dims: %s", input_dims)
    logger.info("Train/Val: %d / %d", len(train_dataset), len(val_dataset))

    history_rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    best_val_metrics: dict[str, Any] | None = None
    best_checkpoint_path = checkpoints_dir / f"best_model_{timestamp}.pth"
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_reg_loss = 0.0
        running_focal_loss = 0.0
        running_count = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_dev = move_batch_to_device(batch, device)
            labels = batch_dev["label"]
            phq9 = batch_dev["phq9"]
            outputs = forward_model(model, batch_dev, return_aux=True)
            loss, loss_parts = compute_total_loss(
                outputs,
                labels,
                phq9,
                criterion_cls,
                criterion_reg,
                criterion_focal,
                focal_lambda=args.focal_lambda,
                reg_lambda=args.reg_lambda,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = int(labels.numel())
            running_count += batch_size
            running_loss += float(loss.item()) * batch_size
            running_cls_loss += loss_parts["cls_loss"] * batch_size
            running_reg_loss += loss_parts["reg_loss"] * batch_size
            running_focal_loss += loss_parts["focal_loss"] * batch_size

        scheduler.step()
        denom = max(1, running_count)
        train_loss = running_loss / denom
        train_cls_loss = running_cls_loss / denom
        train_reg_loss = running_reg_loss / denom
        train_focal_loss = running_focal_loss / denom

        val_metrics = evaluate_model_avgp25(
            model,
            val_loader,
            device,
            args.task,
            criterion_cls,
            criterion_reg,
            criterion_focal,
            focal_lambda=args.focal_lambda,
            reg_lambda=args.reg_lambda,
            selection_mode=args.selection_mode,
        )
        history_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_cls_loss": round(train_cls_loss, 6),
            "train_reg_loss": round(train_reg_loss, 6),
            "train_focal_loss": round(train_focal_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_cls_loss": round(val_metrics["cls_loss"], 6),
            "val_reg_loss": round(val_metrics["reg_loss"], 6),
            "val_focal_loss": round(val_metrics["focal_loss"], 6),
            "val_ccc": round(val_metrics["ccc"], 6),
            "val_rmse": round(val_metrics["rmse"], 6),
            "val_mae": round(val_metrics["mae"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "val_kappa": round(val_metrics["kappa"], 6),
            "val_score": round(val_metrics.get("score", 0.0), 6),
            "selection_score": round(val_metrics.get("selection_score", 0.0), 6),
        }
        if is_regression_task:
            history_row["val_r2"] = round(val_metrics["r2"], 6)

        logger.info(
            "Epoch %d/%d | train_loss=%.6f cls=%.6f reg=%.6f focal=%.6f | "
            "val_f1=%.6f val_acc=%.6f val_kappa=%.6f val_ccc=%.6f val_score=%.6f val_rmse=%.6f val_mae=%.6f",
            epoch,
            args.epochs,
            train_loss,
            train_cls_loss,
            train_reg_loss,
            train_focal_loss,
            val_metrics["f1"],
            val_metrics["acc"],
            val_metrics["kappa"],
            val_metrics["ccc"],
            val_metrics.get("score", 0.0),
            val_metrics["rmse"],
            val_metrics["mae"],
        )
        history_rows.append(history_row)

        current_score = float(val_metrics["selection_score"])
        if current_score > best_score + args.min_delta:
            best_score = current_score
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_val_summary = summarize_metrics(val_metrics)
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": args.model_type,
                    "model_kwargs": model_kwargs,
                    "loss_type": args.loss_type,
                    "focal_gamma": args.focal_gamma,
                    "focal_lambda": args.focal_lambda,
                    "reg_lambda": args.reg_lambda,
                    "track": args.track,
                    "task": args.task,
                    "subtrack": args.subtrack,
                    "encoder_type": args.encoder_type,
                    "audio_feature": args.audio_feature,
                    "video_feature": args.video_feature,
                    "regression_label": args.regression_label if is_regression_task else "",
                    "data_root": to_project_relative_path(args.data_root),
                    "split_csv": to_project_relative_path(args.split_csv),
                    "personality_npy": to_project_relative_path(args.personality_npy),
                    "target_t": args.target_t,
                    "seed": args.seed,
                    "experiment_name": experiment_name,
                    "best_epoch": epoch,
                    "best_val_metrics": best_val_summary,
                    "metric_split": "val",
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_val_metrics is None:
        raise RuntimeError("Training finished without a valid validation checkpoint.")

    best_val_summary = summarize_metrics(best_val_metrics)
    history_path = log_dir / f"history_{timestamp}.csv"
    with open(history_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    best_checkpoint_rel = to_project_relative_path(best_checkpoint_path)
    history_rel = to_project_relative_path(history_path)
    result_payload = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "task": args.task,
        "track": args.track,
        "subtrack": args.subtrack,
        "model_type": args.model_type,
        "encoder_type": args.encoder_type,
        "audio_feature": args.audio_feature,
        "video_feature": args.video_feature,
        "regression_label": args.regression_label if is_regression_task else "",
        "best_epoch": best_epoch,
        "selection_metric": selection_metric_name,
        "best_val_metrics": best_val_summary,
        "checkpoint_path": best_checkpoint_rel,
        "history_path": history_rel,
        "predictions_path": "",
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "config": normalize_path_args(vars(args)),
    }
    result_path = log_dir / f"train_result_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    summary_row = {
        "timestamp": timestamp,
        "task": args.task,
        "track": args.track,
        "subtrack": args.subtrack,
        "model_type": args.model_type,
        "encoder_type": args.encoder_type,
        "audio_feature": args.audio_feature,
        "video_feature": args.video_feature,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "loss_type": args.loss_type,
        "selection_mode": args.selection_mode,
        "label_smoothing": f"{args.label_smoothing:.4f}",
        "focal_lambda": f"{args.focal_lambda:.4f}",
        "reg_lambda": f"{args.reg_lambda:.4f}",
        "use_p_gate": str(bool(args.use_p_gate)),
        "av_encode_pairwise": str(bool(args.av_encode_pairwise)),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "checkpoint_path": best_checkpoint_rel,
        "predictions_path": "",
        "metric_split": "val",
        "selection_metric": selection_metric_name,
        "selection_score": f"{best_val_summary.get('selection_score', 0.0):.6f}",
        "Score": f"{best_val_summary.get('score', 0.0):.6f}",
        "Macro-F1": f"{best_val_summary.get('f1', 0.0):.6f}",
        "ACC": f"{best_val_summary.get('acc', 0.0):.6f}",
        "Kappa": f"{best_val_summary.get('kappa', 0.0):.6f}",
        "CCC": f"{best_val_summary['ccc']:.6f}",
        "RMSE": f"{best_val_summary['rmse']:.6f}",
        "MAE": f"{best_val_summary['mae']:.6f}",
        "R2": f"{best_val_summary.get('r2', 0.0):.6f}" if is_regression_task else "",
    }
    if is_regression_task:
        summary_row["regression_label"] = args.regression_label
    append_summary_row(log_dir / f"{experiment_name}.csv", summary_row)

    logger.info("Best checkpoint: %s", best_checkpoint_rel)
    logger.info("Validation metrics saved to: %s", to_project_relative_path(result_path))


if __name__ == "__main__":
    main()
