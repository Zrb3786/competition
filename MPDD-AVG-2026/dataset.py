from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
EPS = 1e-6
TARGET_T = 128
PAIR_COUNT = 4
GAIT_KEEP_DIM = 9
CLASSIFICATION_TASK_TO_COLUMN = {
    "binary": "label2",
    "ternary": "label3",
}
REGRESSION_TASK = "regression"
AUDIO_FEATURE_ALIASES = {
    "mfcc": ("mfcc", "mfcc64"),
    "opensmile": ("opensmile",),
    "wav2vec": ("wav2vec", "wav2vec2", "wav2vec2-FRA"),
}
VIDEO_FEATURE_ALIASES = {
    "densenet": ("densenet",),
    "resnet": ("resnet",),
    "openface": ("openface",),
}
YOUNG_AUDIO_EVENT_NAMES = {
    1: "E1.npy",
    2: "E2.npy",
    3: "E3.npy",
}
YOUNG_VIDEO_EVENT_NAMES = {
    1: ("event_1.npy", "event_1/event_1_all.npy"),
    2: ("event_2.npy", "event_2/event_2_all.npy"),
    3: ("event_3.npy", "event_3/event_3_all.npy"),
}


def normalize_phq_target(value: float | int) -> float:
    return float(np.log1p(max(float(value), 0.0)))


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _infer_split_counterpart(path: Path) -> Path | None:
    path_str = str(path)
    if path.name == "split_labels_train.csv" and "MPDD-AVG2026-trainval" in path_str:
        return Path(path_str.replace("MPDD-AVG2026-trainval", "MPDD-AVG2026-test")).with_name("split_labels_test.csv")
    if path.name == "split_labels_test.csv" and "MPDD-AVG2026-test" in path_str:
        return Path(path_str.replace("MPDD-AVG2026-test", "MPDD-AVG2026-trainval")).with_name("split_labels_train.csv")
    return None


def load_split_rows(split_csv: str | Path) -> list[dict[str, str]]:
    csv_path = resolve_project_path(split_csv)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    split_values = {row.get("split", "").strip().lower() for row in rows}
    if {"train", "test"}.issubset(split_values):
        return rows

    counterpart = _infer_split_counterpart(csv_path)
    if counterpart is None or not counterpart.exists():
        return rows

    with open(counterpart, "r", encoding="utf-8-sig", newline="") as handle:
        counterpart_rows = list(csv.DictReader(handle))
    return rows + counterpart_rows


def get_label_column(task: str, regression_label: str = "label2") -> str:
    if task == REGRESSION_TASK:
        if regression_label not in {"label2", "label3"}:
            raise ValueError(f"Unsupported regression_label: {regression_label}")
        return regression_label
    if task in CLASSIFICATION_TASK_TO_COLUMN:
        return CLASSIFICATION_TASK_TO_COLUMN[task]
    raise ValueError(f"Unsupported task: {task}")


def get_task_label(row: dict[str, str], task: str, regression_label: str = "label2") -> int:
    column = get_label_column(task, regression_label)
    if column not in row or str(row[column]).strip() == "":
        raise KeyError(f"Column {column} not found for task={task}")
    return int(float(row[column]))


def get_phq9_target(row: dict[str, str]) -> float:
    if "PHQ-9" not in row or str(row["PHQ-9"]).strip() == "":
        raise KeyError("Column PHQ-9 not found in split csv")
    return float(row["PHQ-9"])


def load_task_maps(
    split_csv: str | Path,
    task: str,
    regression_label: str = "label2",
) -> dict[str, Any]:
    rows = load_split_rows(split_csv)
    train_map: dict[int, int] = {}
    test_map: dict[int, int] = {}
    source_split_map: dict[int, str] = {}
    train_phq_map: dict[int, float] = {}
    test_phq_map: dict[int, float] = {}

    for row in rows:
        person_id = int(row["ID"])
        split_name = row["split"].strip().lower()
        label = get_task_label(row, task, regression_label)
        phq9 = get_phq9_target(row)
        source_split_map[person_id] = split_name
        if split_name == "train":
            train_map[person_id] = label
            train_phq_map[person_id] = phq9
        else:
            test_map[person_id] = label
            test_phq_map[person_id] = phq9

    payload: dict[str, Any] = {
        "train_map": train_map,
        "test_map": test_map,
        "source_split_map": source_split_map,
        "rows": rows,
        "split_label": get_label_column(task, regression_label),
        "train_phq_map": train_phq_map,
        "test_phq_map": test_phq_map,
    }
    return payload


def load_label_maps(
    split_csv: str | Path,
    task: str,
    regression_label: str = "label2",
) -> tuple[dict[int, int], dict[int, int], dict[int, str], list[dict[str, str]]]:
    payload = load_task_maps(split_csv, task, regression_label)
    return payload["train_map"], payload["test_map"], payload["source_split_map"], payload["rows"]


def build_subset_label_map(label_map: dict[int, int | float], sample_ids: list[int]) -> dict[int, int | float]:
    return {sample_id: label_map[sample_id] for sample_id in sample_ids}


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


def infer_input_dims(dataset: "MPDDElderDataset") -> dict[str, int]:
    sample = dataset[0]
    return {
        "audio_dim": int(sample["audio"].shape[-1]) if "audio" in sample else 0,
        "video_dim": int(sample["video"].shape[-1]) if "video" in sample else 0,
        "gait_dim": int(sample["gait"].shape[-1]) if "gait" in sample else 0,
    }


def _normalize(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return np.clip((array - mean) / std, -5.0, 5.0).astype(np.float32)


def _resize(array: np.ndarray, target_len: int) -> torch.Tensor:
    tensor = torch.from_numpy(array.astype(np.float32)).transpose(0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
    return tensor.squeeze(0).transpose(0, 1).contiguous()


def _resolve_split_data_roots(data_root: Path) -> dict[str, Path]:
    parent_name = data_root.parent.name
    if parent_name == "MPDD-AVG2026-trainval":
        dataset_root = data_root.parent.parent
        test_root = dataset_root / "MPDD-AVG2026-test" / data_root.name
        return {
            "train": data_root,
            "test": test_root if test_root.exists() else data_root,
        }
    if parent_name == "MPDD-AVG2026-test":
        dataset_root = data_root.parent.parent
        train_root = dataset_root / "MPDD-AVG2026-trainval" / data_root.name
        return {
            "train": train_root if train_root.exists() else data_root,
            "test": data_root,
        }
    return {"train": data_root, "test": data_root}


def _resolve_modality_base(split_root: Path, modality: str, split_name: str) -> Path:
    candidates = [split_root / modality, split_root / modality.capitalize()]
    for base in candidates:
        if not base.exists():
            continue
        split_candidate = base / split_name
        return split_candidate if split_candidate.exists() else base
    return candidates[0]


def _resolve_feature_dir(split_root: Path, modality: str, split_name: str, feature_name: str) -> tuple[Path, str]:
    if modality == "audio":
        aliases = AUDIO_FEATURE_ALIASES.get(feature_name, (feature_name,))
    elif modality == "video":
        aliases = VIDEO_FEATURE_ALIASES.get(feature_name, (feature_name,))
    else:
        raise ValueError(f"Unknown modality={modality}")

    modality_root = _resolve_modality_base(split_root, modality, split_name)
    for alias in aliases:
        candidate = modality_root / alias
        if candidate.exists():
            return candidate, alias
    return modality_root / aliases[0], aliases[0]


def _resolve_gait_root(split_root: Path, split_name: str) -> Path:
    for gait_dir in ("IMU-ELDER", "IMU-Elder", "IMU-Young", "IMU"):
        candidate = split_root / gait_dir
        if not candidate.exists():
            continue
        split_candidate = candidate / split_name
        return split_candidate if split_candidate.exists() else candidate
    return split_root / "IMU" / split_name


def _resolve_gait_file(gait_root: Path, person_id: int) -> Path:
    candidates = (
        gait_root / f"{person_id}.npy",
        gait_root / str(person_id) / f"{person_id}.npy",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _discover_pair_files(root: Path, modality: str) -> dict[int, Path]:
    index_map: dict[int, Path] = {}
    if not root.exists():
        return index_map

    pattern = re.compile(r"^A_(\d+)\.npy$") if modality == "audio" else re.compile(r"^V_(\d+)\.npy$")
    for file_path in sorted(root.glob("*.npy")):
        match = pattern.match(file_path.name)
        if not match:
            continue
        pair_idx = int(match.group(1))
        if 1 <= pair_idx <= PAIR_COUNT:
            index_map[pair_idx] = file_path
    return index_map


def _discover_young_pair_files(audio_dir: Path, video_dir: Path, video_feature: str) -> tuple[dict[int, Path], dict[int, Path]]:
    audio_map: dict[int, Path] = {}
    video_map: dict[int, Path] = {}
    for pair_idx, audio_name in YOUNG_AUDIO_EVENT_NAMES.items():
        audio_file = audio_dir / audio_name
        if not audio_file.is_file():
            continue
        for video_candidate_name in YOUNG_VIDEO_EVENT_NAMES[pair_idx]:
            video_file = video_dir / video_candidate_name
            if video_file.is_file():
                audio_map[pair_idx] = audio_file
                video_map[pair_idx] = video_file
                break
    return audio_map, video_map


def _load_personality_map(personality_npy: str | Path) -> dict[int, np.ndarray]:
    path = resolve_project_path(personality_npy)
    if path.is_dir():
        path = path / "descriptions_embeddings_with_ids.npy"
    if not path.exists():
        return {}
    data = np.load(str(path), allow_pickle=True)
    return {int(item["id"]): np.asarray(item["embedding"], dtype=np.float32) for item in data}


def _infer_feature_dim(roots: list[Path], max_dim: int | None = None) -> int:
    for root in roots:
        if not root.exists():
            continue
        for file_path in root.rglob("*.npy"):
            if not file_path.is_file() or file_path.stat().st_size == 0:
                continue
            try:
                array = np.asarray(np.load(str(file_path), allow_pickle=True), dtype=np.float32)
            except Exception:
                continue
            if array.ndim >= 2 and array.shape[-1] > 0:
                dim = int(array.shape[-1])
                return min(dim, max_dim) if max_dim is not None else dim
            if array.ndim == 1 and array.shape[0] > 0:
                dim = int(array.shape[0])
                return min(dim, max_dim) if max_dim is not None else dim
    return 1


def _load_feature_array(path: Path, fallback_dim: int, max_dim: int | None = None) -> np.ndarray:
    if path.is_file() and path.stat().st_size > 0:
        try:
            array = _normalize(np.load(str(path), allow_pickle=True))
            if max_dim is not None:
                if array.ndim >= 2 and array.shape[-1] > max_dim:
                    array = array[..., :max_dim]
                elif array.ndim == 1 and array.shape[0] > max_dim:
                    array = array[:max_dim]
            return array
        except Exception:
            pass
    effective_dim = min(fallback_dim, max_dim) if max_dim is not None else fallback_dim
    return np.zeros((1, effective_dim), dtype=np.float32)


class MPDDElderDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        label_map: dict[int, int],
        source_split_map: dict[int, str],
        subtrack: str,
        task: str,
        audio_feature: str,
        video_feature: str,
        personality_npy: str | Path,
        phq_map: dict[int, float] | None = None,
        target_t: int = TARGET_T,
    ) -> None:
        self.data_root = resolve_project_path(data_root)
        self.split_data_roots = _resolve_split_data_roots(self.data_root)
        self.label_map = label_map
        self.phq_map = phq_map or {}
        self.source_split_map = source_split_map
        self.subtrack = subtrack
        self.task = task
        self.audio_feature = audio_feature
        self.video_feature = video_feature
        self.target_t = target_t
        self.has_phq_target = phq_map is not None
        self.is_young = any(root.name.lower() == "young" for root in self.split_data_roots.values())
        self.need_av = subtrack in ("A-V+P", "A-V-G+P")
        self.need_gait = subtrack in ("A-V-G+P", "G+P")
        self.audio_roots: dict[str, Path] = {}
        self.video_roots: dict[str, Path] = {}

        if self.need_av:
            resolved_audio = []
            resolved_video = []
            for split_name in ("train", "test"):
                split_root = self.split_data_roots[split_name]
                audio_root, audio_alias = _resolve_feature_dir(split_root, "audio", split_name, audio_feature)
                video_root, video_alias = _resolve_feature_dir(split_root, "video", split_name, video_feature)
                self.audio_roots[split_name] = audio_root
                self.video_roots[split_name] = video_root
                if audio_root.exists():
                    resolved_audio.append(audio_alias)
                if video_root.exists():
                    resolved_video.append(video_alias)
            self.resolved_audio_feature = ",".join(sorted(set(resolved_audio))) if resolved_audio else audio_feature
            self.resolved_video_feature = ",".join(sorted(set(resolved_video))) if resolved_video else video_feature
        else:
            self.resolved_audio_feature = ""
            self.resolved_video_feature = ""

        self.gait_roots = {
            split_name: _resolve_gait_root(self.split_data_roots[split_name], split_name)
            for split_name in ("train", "test")
        }
        self.audio_dim_hint = _infer_feature_dim(list(self.audio_roots.values())) if self.need_av else 0
        self.video_dim_hint = _infer_feature_dim(list(self.video_roots.values())) if self.need_av else 0
        self.gait_dim_hint = _infer_feature_dim(list(self.gait_roots.values()), max_dim=GAIT_KEEP_DIM) if self.need_gait else 0
        self.personality_map = _load_personality_map(personality_npy)
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(
                f"No valid samples found for task={task}, subtrack={subtrack}, "
                f"audio_feature={audio_feature}, video_feature={video_feature}"
            )

    def _collect_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for person_id, label in sorted(self.label_map.items()):
            source_split = self.source_split_map.get(person_id, "train").lower()
            sample: dict[str, Any] = {
                "pid": person_id,
                "label": int(label),
                "source_split": source_split,
            }

            if self.has_phq_target:
                if person_id not in self.phq_map:
                    continue
                sample["phq9"] = normalize_phq_target(self.phq_map[person_id])

            if self.need_gait:
                gait_file = _resolve_gait_file(self.gait_roots[source_split], person_id)
                if not gait_file.exists():
                    continue
                sample["gait_file"] = gait_file

            if self.need_av:
                audio_dir = self.audio_roots[source_split] / str(person_id)
                video_dir = self.video_roots[source_split] / str(person_id)
                if self.is_young:
                    audio_map, video_map = _discover_young_pair_files(audio_dir, video_dir, self.video_feature)
                else:
                    audio_map = _discover_pair_files(audio_dir, "audio")
                    video_map = _discover_pair_files(video_dir, "video")
                shared_indices = sorted(set(audio_map) & set(video_map))
                if not shared_indices:
                    continue
                sample["audio_map"] = audio_map
                sample["video_map"] = video_map
                sample["pair_indices"] = shared_indices[:PAIR_COUNT]

            samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        person_id = int(sample["pid"])
        label = torch.tensor(int(sample["label"]), dtype=torch.long)

        result: dict[str, torch.Tensor] = {
            "pid": torch.tensor(person_id, dtype=torch.long),
            "label": label,
        }

        if self.has_phq_target:
            result["phq9"] = torch.tensor(float(sample["phq9"]), dtype=torch.float32)

        if self.need_av:
            audio_pairs: list[torch.Tensor] = []
            video_pairs: list[torch.Tensor] = []
            pair_mask: list[float] = []
            audio_map: dict[int, Path] = sample["audio_map"]
            video_map: dict[int, Path] = sample["video_map"]
            pair_indices: list[int] = sample["pair_indices"]
            for pair_idx in pair_indices:
                audio_pairs.append(_resize(_load_feature_array(audio_map[pair_idx], self.audio_dim_hint), self.target_t))
                video_pairs.append(_resize(_load_feature_array(video_map[pair_idx], self.video_dim_hint), self.target_t))
                pair_mask.append(1.0)

            audio_dim = int(audio_pairs[0].shape[-1])
            video_dim = int(video_pairs[0].shape[-1])
            while len(audio_pairs) < PAIR_COUNT:
                audio_pairs.append(torch.zeros(self.target_t, audio_dim, dtype=torch.float32))
                video_pairs.append(torch.zeros(self.target_t, video_dim, dtype=torch.float32))
                pair_mask.append(0.0)

            result["audio"] = torch.stack(audio_pairs)
            result["video"] = torch.stack(video_pairs)
            result["pair_mask"] = torch.tensor(pair_mask, dtype=torch.float32)

        if self.need_gait:
            gait_arr = _load_feature_array(sample["gait_file"], self.gait_dim_hint, max_dim=GAIT_KEEP_DIM)
            result["gait"] = _resize(gait_arr, self.target_t)

        personality = self.personality_map.get(person_id, np.zeros(1024, dtype=np.float32))
        result["personality"] = torch.from_numpy(personality.astype(np.float32))
        return result
# from __future__ import annotations

# import csv
# import re
# from pathlib import Path
# from typing import Any

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset


# PROJECT_ROOT = Path(__file__).resolve().parent
# EPS = 1e-6
# TARGET_T = 128
# PAIR_COUNT = 4
# GAIT_KEEP_DIM = 9
# CLASSIFICATION_TASK_TO_COLUMN = {
#     "binary": "label2",
#     "ternary": "label3",
# }
# REGRESSION_TASK = "regression"
# AUDIO_FEATURE_ALIASES = {
#     "mfcc": ("mfcc", "mfcc64"),
#     "opensmile": ("opensmile",),
#     "wav2vec": ("wav2vec", "wav2vec2", "wav2vec2-FRA"),
# }
# VIDEO_FEATURE_ALIASES = {
#     "densenet": ("densenet",),
#     "resnet": ("resnet",),
#     "openface": ("openface",),
# }
# YOUNG_AUDIO_EVENT_NAMES = {
#     1: "E1.npy",
#     2: "E2.npy",
#     3: "E3.npy",
# }
# YOUNG_VIDEO_EVENT_NAMES = {
#     1: ("event_1.npy", "event_1/event_1_all.npy"),
#     2: ("event_2.npy", "event_2/event_2_all.npy"),
#     3: ("event_3.npy", "event_3/event_3_all.npy"),
# }


# def normalize_phq_target(value: float | int) -> float:
#     return float(np.log1p(max(float(value), 0.0)))


# def resolve_project_path(path_like: str | Path) -> Path:
#     path = Path(path_like)
#     return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


# def _infer_split_counterpart(path: Path) -> Path | None:
#     path_str = str(path)
#     if path.name == "split_labels_train.csv" and "MPDD-AVG2026-trainval" in path_str:
#         return Path(path_str.replace("MPDD-AVG2026-trainval", "MPDD-AVG2026-test")).with_name("split_labels_test.csv")
#     if path.name == "split_labels_test.csv" and "MPDD-AVG2026-test" in path_str:
#         return Path(path_str.replace("MPDD-AVG2026-test", "MPDD-AVG2026-trainval")).with_name("split_labels_train.csv")
#     return None


# def load_split_rows(split_csv: str | Path) -> list[dict[str, str]]:
#     csv_path = resolve_project_path(split_csv)
#     with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
#         rows = list(csv.DictReader(handle))

#     split_values = {row.get("split", "").strip().lower() for row in rows}
#     if {"train", "test"}.issubset(split_values):
#         return rows

#     counterpart = _infer_split_counterpart(csv_path)
#     if counterpart is None or not counterpart.exists():
#         return rows

#     with open(counterpart, "r", encoding="utf-8-sig", newline="") as handle:
#         counterpart_rows = list(csv.DictReader(handle))
#     return rows + counterpart_rows


# def get_label_column(task: str, regression_label: str = "label2") -> str:
#     if task == REGRESSION_TASK:
#         if regression_label not in {"label2", "label3"}:
#             raise ValueError(f"Unsupported regression_label: {regression_label}")
#         return regression_label
#     if task in CLASSIFICATION_TASK_TO_COLUMN:
#         return CLASSIFICATION_TASK_TO_COLUMN[task]
#     raise ValueError(f"Unsupported task: {task}")


# def get_task_label(row: dict[str, str], task: str, regression_label: str = "label2") -> int:
#     column = get_label_column(task, regression_label)
#     if column not in row or str(row[column]).strip() == "":
#         raise KeyError(f"Column {column} not found for task={task}")
#     return int(float(row[column]))


# def get_phq9_target(row: dict[str, str]) -> float:
#     if "PHQ-9" not in row or str(row["PHQ-9"]).strip() == "":
#         raise KeyError("Column PHQ-9 not found in split csv")
#     return float(row["PHQ-9"])


# def load_task_maps(
#     split_csv: str | Path,
#     task: str,
#     regression_label: str = "label2",
# ) -> dict[str, Any]:
#     rows = load_split_rows(split_csv)
#     train_map: dict[int, int] = {}
#     test_map: dict[int, int] = {}
#     source_split_map: dict[int, str] = {}
#     train_phq_map: dict[int, float] = {}
#     test_phq_map: dict[int, float] = {}

#     for row in rows:
#         person_id = int(row["ID"])
#         split_name = row["split"].strip().lower()
#         label = get_task_label(row, task, regression_label)
#         phq9 = get_phq9_target(row)
#         source_split_map[person_id] = split_name
#         if split_name == "train":
#             train_map[person_id] = label
#             train_phq_map[person_id] = phq9
#         else:
#             test_map[person_id] = label
#             test_phq_map[person_id] = phq9

#     payload: dict[str, Any] = {
#         "train_map": train_map,
#         "test_map": test_map,
#         "source_split_map": source_split_map,
#         "rows": rows,
#         "split_label": get_label_column(task, regression_label),
#         "train_phq_map": train_phq_map,
#         "test_phq_map": test_phq_map,
#     }
#     return payload


# def load_label_maps(
#     split_csv: str | Path,
#     task: str,
#     regression_label: str = "label2",
# ) -> tuple[dict[int, int], dict[int, int], dict[int, str], list[dict[str, str]]]:
#     payload = load_task_maps(split_csv, task, regression_label)
#     return payload["train_map"], payload["test_map"], payload["source_split_map"], payload["rows"]


# def build_subset_label_map(label_map: dict[int, int | float], sample_ids: list[int]) -> dict[int, int | float]:
#     return {sample_id: label_map[sample_id] for sample_id in sample_ids}


# def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
#     keys = batch[0].keys()
#     return {key: torch.stack([item[key] for item in batch]) for key in keys}


# def infer_input_dims(dataset: "MPDDElderDataset") -> dict[str, int]:
#     sample = dataset[0]
#     return {
#         "audio_dim": int(sample["audio"].shape[-1]) if "audio" in sample else 0,
#         "audio_aux_dim": int(sample["audio_aux"].shape[-1]) if "audio_aux" in sample else 0,
#         "video_dim": int(sample["video"].shape[-1]) if "video" in sample else 0,
#         "gait_dim": int(sample["gait"].shape[-1]) if "gait" in sample else 0,
#     }


# def _normalize(array: np.ndarray) -> np.ndarray:
#     array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
#     mean = array.mean(axis=0, keepdims=True)
#     std = array.std(axis=0, keepdims=True)
#     std = np.where(std < EPS, 1.0, std)
#     return np.clip((array - mean) / std, -5.0, 5.0).astype(np.float32)


# def _resize(array: np.ndarray, target_len: int) -> torch.Tensor:
#     tensor = torch.from_numpy(array.astype(np.float32)).transpose(0, 1).unsqueeze(0)
#     tensor = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
#     return tensor.squeeze(0).transpose(0, 1).contiguous()


# def _resolve_split_data_roots(data_root: Path) -> dict[str, Path]:
#     parent_name = data_root.parent.name
#     if parent_name == "MPDD-AVG2026-trainval":
#         dataset_root = data_root.parent.parent
#         test_root = dataset_root / "MPDD-AVG2026-test" / data_root.name
#         return {
#             "train": data_root,
#             "test": test_root if test_root.exists() else data_root,
#         }
#     if parent_name == "MPDD-AVG2026-test":
#         dataset_root = data_root.parent.parent
#         train_root = dataset_root / "MPDD-AVG2026-trainval" / data_root.name
#         return {
#             "train": train_root if train_root.exists() else data_root,
#             "test": data_root,
#         }
#     return {"train": data_root, "test": data_root}


# def _resolve_modality_base(split_root: Path, modality: str, split_name: str) -> Path:
#     candidates = [split_root / modality, split_root / modality.capitalize()]
#     for base in candidates:
#         if not base.exists():
#             continue
#         split_candidate = base / split_name
#         return split_candidate if split_candidate.exists() else base
#     return candidates[0]


# def _resolve_feature_dir(split_root: Path, modality: str, split_name: str, feature_name: str) -> tuple[Path, str]:
#     if modality == "audio":
#         aliases = AUDIO_FEATURE_ALIASES.get(feature_name, (feature_name,))
#     elif modality == "video":
#         aliases = VIDEO_FEATURE_ALIASES.get(feature_name, (feature_name,))
#     else:
#         raise ValueError(f"Unknown modality={modality}")

#     modality_root = _resolve_modality_base(split_root, modality, split_name)
#     for alias in aliases:
#         candidate = modality_root / alias
#         if candidate.exists():
#             return candidate, alias
#     return modality_root / aliases[0], aliases[0]


# def _resolve_gait_root(split_root: Path, split_name: str) -> Path:
#     for gait_dir in ("IMU-ELDER", "IMU-Elder", "IMU-Young", "IMU"):
#         candidate = split_root / gait_dir
#         if not candidate.exists():
#             continue
#         split_candidate = candidate / split_name
#         return split_candidate if split_candidate.exists() else candidate
#     return split_root / "IMU" / split_name


# def _resolve_gait_file(gait_root: Path, person_id: int) -> Path:
#     candidates = (
#         gait_root / f"{person_id}.npy",
#         gait_root / str(person_id) / f"{person_id}.npy",
#     )
#     for candidate in candidates:
#         if candidate.exists():
#             return candidate
#     return candidates[-1]


# def _discover_pair_files(root: Path, modality: str) -> dict[int, Path]:
#     index_map: dict[int, Path] = {}
#     if not root.exists():
#         return index_map

#     pattern = re.compile(r"^A_(\d+)\.npy$") if modality == "audio" else re.compile(r"^V_(\d+)\.npy$")
#     for file_path in sorted(root.glob("*.npy")):
#         match = pattern.match(file_path.name)
#         if not match:
#             continue
#         pair_idx = int(match.group(1))
#         if 1 <= pair_idx <= PAIR_COUNT:
#             index_map[pair_idx] = file_path
#     return index_map


# def _discover_young_pair_files(audio_dir: Path, video_dir: Path, video_feature: str) -> tuple[dict[int, Path], dict[int, Path]]:
#     audio_map: dict[int, Path] = {}
#     video_map: dict[int, Path] = {}
#     for pair_idx, audio_name in YOUNG_AUDIO_EVENT_NAMES.items():
#         audio_file = audio_dir / audio_name
#         if not audio_file.is_file():
#             continue
#         for video_candidate_name in YOUNG_VIDEO_EVENT_NAMES[pair_idx]:
#             video_file = video_dir / video_candidate_name
#             if video_file.is_file():
#                 audio_map[pair_idx] = audio_file
#                 video_map[pair_idx] = video_file
#                 break
#     return audio_map, video_map


# def _discover_young_audio_files(audio_dir: Path) -> dict[int, Path]:
#     audio_map: dict[int, Path] = {}
#     for pair_idx, audio_name in YOUNG_AUDIO_EVENT_NAMES.items():
#         audio_file = audio_dir / audio_name
#         if audio_file.is_file():
#             audio_map[pair_idx] = audio_file
#     return audio_map


# def _load_personality_map(personality_npy: str | Path) -> dict[int, np.ndarray]:
#     path = resolve_project_path(personality_npy)
#     if path.is_dir():
#         path = path / "descriptions_embeddings_with_ids.npy"
#     if not path.exists():
#         return {}
#     data = np.load(str(path), allow_pickle=True)
#     return {int(item["id"]): np.asarray(item["embedding"], dtype=np.float32) for item in data}


# def _infer_feature_dim(roots: list[Path], max_dim: int | None = None) -> int:
#     for root in roots:
#         if not root.exists():
#             continue
#         for file_path in root.rglob("*.npy"):
#             if not file_path.is_file() or file_path.stat().st_size == 0:
#                 continue
#             try:
#                 array = np.asarray(np.load(str(file_path), allow_pickle=True), dtype=np.float32)
#             except Exception:
#                 continue
#             if array.ndim >= 2 and array.shape[-1] > 0:
#                 dim = int(array.shape[-1])
#                 return min(dim, max_dim) if max_dim is not None else dim
#             if array.ndim == 1 and array.shape[0] > 0:
#                 dim = int(array.shape[0])
#                 return min(dim, max_dim) if max_dim is not None else dim
#     return 1


# def _load_feature_array(path: Path, fallback_dim: int, max_dim: int | None = None) -> np.ndarray:
#     if path.is_file() and path.stat().st_size > 0:
#         try:
#             array = _normalize(np.load(str(path), allow_pickle=True))
#             if max_dim is not None:
#                 if array.ndim >= 2 and array.shape[-1] > max_dim:
#                     array = array[..., :max_dim]
#                 elif array.ndim == 1 and array.shape[0] > max_dim:
#                     array = array[:max_dim]
#             return array
#         except Exception:
#             pass
#     effective_dim = min(fallback_dim, max_dim) if max_dim is not None else fallback_dim
#     return np.zeros((1, effective_dim), dtype=np.float32)


# class MPDDElderDataset(Dataset):
#     def __init__(
#         self,
#         data_root: str | Path,
#         label_map: dict[int, int],
#         source_split_map: dict[int, str],
#         subtrack: str,
#         task: str,
#         audio_feature: str,
#         video_feature: str,
#         audio_aux_feature: str = "",
#         personality_npy: str | Path = "",
#         phq_map: dict[int, float] | None = None,
#         target_t: int = TARGET_T,
#     ) -> None:
#         self.data_root = resolve_project_path(data_root)
#         self.split_data_roots = _resolve_split_data_roots(self.data_root)
#         self.label_map = label_map
#         self.phq_map = phq_map or {}
#         self.source_split_map = source_split_map
#         self.subtrack = subtrack
#         self.task = task
#         self.audio_feature = audio_feature
#         self.video_feature = video_feature
#         self.audio_aux_feature = audio_aux_feature
#         self.target_t = target_t
#         self.has_phq_target = phq_map is not None
#         self.is_young = any(root.name.lower() == "young" for root in self.split_data_roots.values())
#         self.need_av = subtrack in ("A-V+P", "A-V-G+P")
#         self.need_gait = subtrack in ("A-V-G+P", "G+P")
#         self.audio_roots: dict[str, Path] = {}
#         self.audio_aux_roots: dict[str, Path] = {}
#         self.video_roots: dict[str, Path] = {}

#         if self.need_av:
#             resolved_audio = []
#             resolved_video = []
#             for split_name in ("train", "test"):
#                 split_root = self.split_data_roots[split_name]
#                 audio_root, audio_alias = _resolve_feature_dir(split_root, "audio", split_name, audio_feature)
#                 video_root, video_alias = _resolve_feature_dir(split_root, "video", split_name, video_feature)
#                 self.audio_roots[split_name] = audio_root
#                 self.video_roots[split_name] = video_root
#                 if self.audio_aux_feature:
#                     audio_aux_root, _ = _resolve_feature_dir(split_root, "audio", split_name, self.audio_aux_feature)
#                     self.audio_aux_roots[split_name] = audio_aux_root
#                 if audio_root.exists():
#                     resolved_audio.append(audio_alias)
#                 if video_root.exists():
#                     resolved_video.append(video_alias)
#             self.resolved_audio_feature = ",".join(sorted(set(resolved_audio))) if resolved_audio else audio_feature
#             self.resolved_video_feature = ",".join(sorted(set(resolved_video))) if resolved_video else video_feature
#         else:
#             self.resolved_audio_feature = ""
#             self.resolved_video_feature = ""

#         self.gait_roots = {
#             split_name: _resolve_gait_root(self.split_data_roots[split_name], split_name)
#             for split_name in ("train", "test")
#         }
#         self.audio_dim_hint = _infer_feature_dim(list(self.audio_roots.values())) if self.need_av else 0
#         self.audio_aux_dim_hint = _infer_feature_dim(list(self.audio_aux_roots.values())) if self.audio_aux_feature else 0
#         self.video_dim_hint = _infer_feature_dim(list(self.video_roots.values())) if self.need_av else 0
#         self.gait_dim_hint = _infer_feature_dim(list(self.gait_roots.values()), max_dim=GAIT_KEEP_DIM) if self.need_gait else 0
#         self.personality_map = _load_personality_map(personality_npy)
#         self.samples = self._collect_samples()
#         if not self.samples:
#             raise RuntimeError(
#                 f"No valid samples found for task={task}, subtrack={subtrack}, "
#                 f"audio_feature={audio_feature}, video_feature={video_feature}"
#             )

#     def _collect_samples(self) -> list[dict[str, Any]]:
#         samples: list[dict[str, Any]] = []
#         for person_id, label in sorted(self.label_map.items()):
#             source_split = self.source_split_map.get(person_id, "train").lower()
#             sample: dict[str, Any] = {
#                 "pid": person_id,
#                 "label": int(label),
#                 "source_split": source_split,
#             }

#             if self.has_phq_target:
#                 if person_id not in self.phq_map:
#                     continue
#                 sample["phq9"] = normalize_phq_target(self.phq_map[person_id])

#             if self.need_gait:
#                 gait_file = _resolve_gait_file(self.gait_roots[source_split], person_id)
#                 if not gait_file.exists():
#                     continue
#                 sample["gait_file"] = gait_file

#             if self.need_av:
#                 audio_dir = self.audio_roots[source_split] / str(person_id)
#                 video_dir = self.video_roots[source_split] / str(person_id)
#                 if self.is_young:
#                     audio_map, video_map = _discover_young_pair_files(audio_dir, video_dir, self.video_feature)
#                 else:
#                     audio_map = _discover_pair_files(audio_dir, "audio")
#                     video_map = _discover_pair_files(video_dir, "video")

#                 if self.audio_aux_feature:
#                     audio_aux_dir = self.audio_aux_roots[source_split] / str(person_id)
#                     if self.is_young:
#                         audio_aux_map = _discover_young_audio_files(audio_aux_dir)
#                     else:
#                         audio_aux_map = _discover_pair_files(audio_aux_dir, "audio")
#                     shared_indices = sorted(set(audio_map) & set(video_map) & set(audio_aux_map))
#                     sample["audio_aux_map"] = audio_aux_map
#                 else:
#                     shared_indices = sorted(set(audio_map) & set(video_map))

#                 if not shared_indices:
#                     continue
#                 sample["audio_map"] = audio_map
#                 sample["video_map"] = video_map
#                 sample["pair_indices"] = shared_indices[:PAIR_COUNT]

#             samples.append(sample)
#         return samples

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
#         sample = self.samples[index]
#         person_id = int(sample["pid"])
#         label = torch.tensor(int(sample["label"]), dtype=torch.long)

#         result: dict[str, torch.Tensor] = {
#             "pid": torch.tensor(person_id, dtype=torch.long),
#             "label": label,
#         }

#         if self.has_phq_target:
#             result["phq9"] = torch.tensor(float(sample["phq9"]), dtype=torch.float32)

#         if self.need_av:
#             audio_pairs: list[torch.Tensor] = []
#             audio_aux_pairs: list[torch.Tensor] = []
#             video_pairs: list[torch.Tensor] = []
#             pair_mask: list[float] = []
#             audio_map: dict[int, Path] = sample["audio_map"]
#             video_map: dict[int, Path] = sample["video_map"]
#             audio_aux_map: dict[int, Path] = sample.get("audio_aux_map", {})
#             pair_indices: list[int] = sample["pair_indices"]
#             for pair_idx in pair_indices:
#                 audio_pairs.append(_resize(_load_feature_array(audio_map[pair_idx], self.audio_dim_hint), self.target_t))
#                 video_pairs.append(_resize(_load_feature_array(video_map[pair_idx], self.video_dim_hint), self.target_t))
#                 if self.audio_aux_feature:
#                     audio_aux_pairs.append(_resize(_load_feature_array(audio_aux_map[pair_idx], self.audio_aux_dim_hint), self.target_t))
#                 pair_mask.append(1.0)

#             audio_dim = int(audio_pairs[0].shape[-1])
#             video_dim = int(video_pairs[0].shape[-1])
#             if self.audio_aux_feature:
#                 audio_aux_dim = int(audio_aux_pairs[0].shape[-1])
#             while len(audio_pairs) < PAIR_COUNT:
#                 audio_pairs.append(torch.zeros(self.target_t, audio_dim, dtype=torch.float32))
#                 video_pairs.append(torch.zeros(self.target_t, video_dim, dtype=torch.float32))
#                 if self.audio_aux_feature:
#                     audio_aux_pairs.append(torch.zeros(self.target_t, audio_aux_dim, dtype=torch.float32))
#                 pair_mask.append(0.0)

#             result["audio"] = torch.stack(audio_pairs)
#             if self.audio_aux_feature:
#                 result["audio_aux"] = torch.stack(audio_aux_pairs)
#             result["video"] = torch.stack(video_pairs)
#             result["pair_mask"] = torch.tensor(pair_mask, dtype=torch.float32)

#         if self.need_gait:
#             gait_arr = _load_feature_array(sample["gait_file"], self.gait_dim_hint, max_dim=GAIT_KEEP_DIM)
#             result["gait"] = _resize(gait_arr, self.target_t)

#         personality = self.personality_map.get(person_id, np.zeros(1024, dtype=np.float32))
#         result["personality"] = torch.from_numpy(personality.astype(np.float32))
#         return result