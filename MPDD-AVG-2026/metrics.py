from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    denom = y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2
    return safe_float(2 * cov / denom) if denom > 1e-10 else 0.0


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = {
        "acc": safe_float(accuracy_score(y_true, y_pred)),
        "f1": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": safe_float(cohen_kappa_score(y_true, y_pred)),
        "ccc": ccc(y_true, y_pred),
        "rmse": safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": safe_float(mean_absolute_error(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    metrics["selection_score"] = metrics["f1"]
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = {
        "ccc": ccc(y_true, y_pred),
        "rmse": safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": safe_float(mean_absolute_error(y_true, y_pred)),
        "r2": safe_float(r2_score(y_true, y_pred)),
    }
    metrics["selection_score"] = metrics["ccc"]
    return metrics


def joint_regression_metrics(
    class_true: np.ndarray,
    class_pred: np.ndarray,
    phq_true: np.ndarray,
    phq_pred: np.ndarray,
) -> dict[str, Any]:
    metrics = classification_metrics(class_true, class_pred)
    reg_metrics = regression_metrics(phq_true, phq_pred)
    metrics["ccc"] = reg_metrics["ccc"]
    metrics["rmse"] = reg_metrics["rmse"]
    metrics["mae"] = reg_metrics["mae"]
    metrics["r2"] = reg_metrics["r2"]
    metrics["selection_score"] = metrics["f1"]
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Any,
    device: torch.device,
    task: str,
) -> dict[str, Any]:
    is_joint_regression = isinstance(criterion, (tuple, list))
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []
    all_ids: list[int] = []
    all_phq_preds: list[float] = []
    all_phq_labels: list[float] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            outputs = model(
                audio=batch["audio"].to(device) if "audio" in batch else None,
                video=batch["video"].to(device) if "video" in batch else None,
                gait=batch["gait"].to(device) if "gait" in batch else None,
                personality=batch["personality"].to(device),
                pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
            )
            if is_joint_regression:
                criterion_cls, criterion_reg = criterion
                phq9 = batch["phq9"].to(device)
                logits, reg_out = outputs
                loss_cls = criterion_cls(logits, labels)
                loss_reg = criterion_reg(reg_out, phq9)
                loss = loss_cls + loss_reg
                batch_preds = logits.argmax(dim=-1).cpu().numpy().tolist()
                batch_labels = labels.cpu().numpy().tolist()
                batch_phq_preds = reg_out.cpu().numpy().tolist()
                batch_phq_labels = phq9.cpu().numpy().tolist()
                total_cls_loss += float(loss_cls.item()) * len(batch_labels)
                total_reg_loss += float(loss_reg.item()) * len(batch_labels)
            else:
                logits = outputs
                loss = criterion(logits, labels)
                batch_preds = logits.argmax(dim=-1).cpu().numpy().tolist()
                batch_labels = labels.cpu().numpy().tolist()

            total_loss += float(loss.item()) * len(batch_labels)
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            all_ids.extend(batch["pid"].cpu().numpy().tolist())
            if is_joint_regression:
                all_phq_preds.extend(batch_phq_preds)
                all_phq_labels.extend(batch_phq_labels)

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    if is_joint_regression:
        phq_true = np.asarray(all_phq_labels, dtype=np.float64)
        phq_pred = np.asarray(all_phq_preds, dtype=np.float64)
        metrics = joint_regression_metrics(y_true, y_pred, phq_true, phq_pred)
        metrics["cls_loss"] = safe_float(total_cls_loss / max(1, len(all_labels)))
        metrics["reg_loss"] = safe_float(total_reg_loss / max(1, len(all_labels)))
        metrics["class_true"] = y_true.tolist()
        metrics["class_pred"] = y_pred.tolist()
        metrics["phq_true"] = phq_true.tolist()
        metrics["phq_pred"] = phq_pred.tolist()
        metrics["y_true"] = phq_true.tolist()
        metrics["y_pred"] = phq_pred.tolist()
    else:
        metrics = classification_metrics(y_true, y_pred)
        metrics["y_true"] = y_true.tolist()
        metrics["y_pred"] = y_pred.tolist()

    if task == "regression":
        metrics["selection_score"] = safe_float(metrics.get("ccc"))
    else:
        metrics["selection_score"] = safe_float(metrics.get("f1", metrics.get("ccc", 0.0)))

    metrics["loss"] = safe_float(total_loss / max(1, len(all_labels)))
    metrics["ids"] = all_ids
    return metrics
# from __future__ import annotations

# from typing import Any

# import numpy as np
# import torch
# from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# def safe_float(value: Any) -> float:
#     try:
#         value = float(value)
#     except (TypeError, ValueError):
#         return 0.0
#     if np.isnan(value) or np.isinf(value):
#         return 0.0
#     return value


# def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     y_true = np.asarray(y_true, dtype=np.float64)
#     y_pred = np.asarray(y_pred, dtype=np.float64)
#     cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
#     denom = y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2
#     return safe_float(2 * cov / denom) if denom > 1e-10 else 0.0


# def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
#     metrics = {
#         "acc": safe_float(accuracy_score(y_true, y_pred)),
#         "f1": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
#         "kappa": safe_float(cohen_kappa_score(y_true, y_pred)),
#         "ccc": ccc(y_true, y_pred),
#         "rmse": safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
#         "mae": safe_float(mean_absolute_error(y_true, y_pred)),
#         "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
#     }
#     metrics["selection_score"] = metrics["f1"]
#     return metrics


# def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
#     metrics = {
#         "ccc": ccc(y_true, y_pred),
#         "rmse": safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
#         "mae": safe_float(mean_absolute_error(y_true, y_pred)),
#         "r2": safe_float(r2_score(y_true, y_pred)),
#     }
#     metrics["selection_score"] = metrics["ccc"]
#     return metrics


# def joint_regression_metrics(
#     class_true: np.ndarray,
#     class_pred: np.ndarray,
#     phq_true: np.ndarray,
#     phq_pred: np.ndarray,
# ) -> dict[str, Any]:
#     metrics = classification_metrics(class_true, class_pred)
#     reg_metrics = regression_metrics(phq_true, phq_pred)
#     metrics["ccc"] = reg_metrics["ccc"]
#     metrics["rmse"] = reg_metrics["rmse"]
#     metrics["mae"] = reg_metrics["mae"]
#     metrics["r2"] = reg_metrics["r2"]
#     metrics["selection_score"] = metrics["f1"]
#     return metrics


# def evaluate_model(
#     model: torch.nn.Module,
#     loader: torch.utils.data.DataLoader,
#     criterion: Any,
#     device: torch.device,
#     task: str,
# ) -> dict[str, Any]:
#     is_joint_regression = isinstance(criterion, (tuple, list))
#     model.eval()
#     total_loss = 0.0
#     total_cls_loss = 0.0
#     total_reg_loss = 0.0
#     all_preds: list[float] = []
#     all_labels: list[float] = []
#     all_ids: list[int] = []
#     all_phq_preds: list[float] = []
#     all_phq_labels: list[float] = []

#     with torch.no_grad():
#         for batch in loader:
#             labels = batch["label"].to(device)
#             outputs = model(
#                 audio=batch["audio"].to(device) if "audio" in batch else None,
#                 audio_aux=batch["audio_aux"].to(device) if "audio_aux" in batch else None,
#                 video=batch["video"].to(device) if "video" in batch else None,
#                 gait=batch["gait"].to(device) if "gait" in batch else None,
#                 personality=batch["personality"].to(device),
#                 pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
#             )
#             if is_joint_regression:
#                 criterion_cls, criterion_reg = criterion
#                 phq9 = batch["phq9"].to(device)
#                 logits, reg_out = outputs
#                 loss_cls = criterion_cls(logits, labels)
#                 loss_reg = criterion_reg(reg_out, phq9)
#                 loss = loss_cls + loss_reg
#                 batch_preds = logits.argmax(dim=-1).cpu().numpy().tolist()
#                 batch_labels = labels.cpu().numpy().tolist()
#                 batch_phq_preds = reg_out.cpu().numpy().tolist()
#                 batch_phq_labels = phq9.cpu().numpy().tolist()
#                 total_cls_loss += float(loss_cls.item()) * len(batch_labels)
#                 total_reg_loss += float(loss_reg.item()) * len(batch_labels)
#             else:
#                 logits = outputs
#                 loss = criterion(logits, labels)
#                 batch_preds = logits.argmax(dim=-1).cpu().numpy().tolist()
#                 batch_labels = labels.cpu().numpy().tolist()

#             total_loss += float(loss.item()) * len(batch_labels)
#             all_preds.extend(batch_preds)
#             all_labels.extend(batch_labels)
#             all_ids.extend(batch["pid"].cpu().numpy().tolist())
#             if is_joint_regression:
#                 all_phq_preds.extend(batch_phq_preds)
#                 all_phq_labels.extend(batch_phq_labels)

#     y_true = np.asarray(all_labels, dtype=np.int64)
#     y_pred = np.asarray(all_preds, dtype=np.int64)
#     if is_joint_regression:
#         phq_true = np.asarray(all_phq_labels, dtype=np.float64)
#         phq_pred = np.asarray(all_phq_preds, dtype=np.float64)
#         metrics = joint_regression_metrics(y_true, y_pred, phq_true, phq_pred)
#         metrics["cls_loss"] = safe_float(total_cls_loss / max(1, len(all_labels)))
#         metrics["reg_loss"] = safe_float(total_reg_loss / max(1, len(all_labels)))
#         metrics["class_true"] = y_true.tolist()
#         metrics["class_pred"] = y_pred.tolist()
#         metrics["phq_true"] = phq_true.tolist()
#         metrics["phq_pred"] = phq_pred.tolist()
#         metrics["y_true"] = phq_true.tolist()
#         metrics["y_pred"] = phq_pred.tolist()
#     else:
#         metrics = classification_metrics(y_true, y_pred)
#         metrics["y_true"] = y_true.tolist()
#         metrics["y_pred"] = y_pred.tolist()

#     if task == "regression":
#         metrics["selection_score"] = safe_float(metrics.get("ccc"))
#     else:
#         metrics["selection_score"] = safe_float(metrics.get("f1", metrics.get("ccc", 0.0)))

#     metrics["loss"] = safe_float(total_loss / max(1, len(all_labels)))
#     metrics["ids"] = all_ids
#     return metrics