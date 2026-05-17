from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_temporal_encoder import HybridTemporalEncoder


class ModalityEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.5,
        pre_dim: int | None = None,
    ) -> None:
        super().__init__()
        if pre_dim is not None and input_dim > pre_dim:
            self.pre_proj = nn.Linear(input_dim, pre_dim)
            lstm_in = pre_dim
        else:
            self.pre_proj = None
            lstm_in = input_dim
        self.proj = nn.Linear(lstm_in, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_proj is not None:
            x = F.relu(self.pre_proj(x))
        x = F.relu(self.proj(x))
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.norm(x.mean(dim=1))


class PersonalityEncoder(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchcatBaseline(nn.Module):
    SUBTRACKS = {
        "A-V+P": ["audio", "video", "personality"],
        "A-V-G+P": ["audio", "video", "gait", "personality"],
        "G+P": ["gait", "personality"],
    }
    ENCODER_TYPES = {"bilstm_mean", "hybrid_attn"}

    def __init__(
        self,
        subtrack: str = "A-V-G+P",
        num_classes: int = 3,
        is_regression: bool = False,
        use_regression_head: bool = False,
        audio_dim: int = 64,
        video_dim: int = 1000,
        gait_dim: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        encoder_type: str = "bilstm_mean",
    ) -> None:
        super().__init__()
        if subtrack not in self.SUBTRACKS:
            raise ValueError(f"Unknown subtrack: {subtrack}")
        if encoder_type not in self.ENCODER_TYPES:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.subtrack = subtrack
        self.modalities = self.SUBTRACKS[subtrack]
        self.encoder_type = encoder_type
        self.is_regression = is_regression
        self.use_regression_head = use_regression_head

        if "audio" in self.modalities:
            pre_audio = 128 if audio_dim > 128 else None
            self.audio_enc = (
                HybridTemporalEncoder(audio_dim, hidden_dim, dropout, pre_dim=pre_audio)
                if encoder_type == "hybrid_attn"
                else ModalityEncoder(audio_dim, hidden_dim, dropout, pre_dim=pre_audio)
            )
        if "video" in self.modalities:
            pre_video = 128 if video_dim > 128 else None
            self.video_enc = (
                HybridTemporalEncoder(video_dim, hidden_dim, dropout, pre_dim=pre_video)
                if encoder_type == "hybrid_attn"
                else ModalityEncoder(video_dim, hidden_dim, dropout, pre_dim=pre_video)
            )
        if "gait" in self.modalities:
            self.gait_enc = ModalityEncoder(gait_dim, hidden_dim, dropout)
        if "personality" in self.modalities:
            self.pers_enc = PersonalityEncoder(1024, hidden_dim, dropout)

        fused_dim = hidden_dim * len(self.modalities)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1 if is_regression else num_classes),
        )
        if use_regression_head:
            self.regressor = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    @staticmethod
    def _masked_average_sequences(x: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pair_mask is None:
            return x.mean(dim=1)
        weights = pair_mask.unsqueeze(-1).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    @staticmethod
    def _masked_average_features(x: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pair_mask is None:
            return x.mean(dim=1)
        weights = pair_mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def _encode_pairwise_sequences(
        self,
        x: torch.Tensor,
        encoder: nn.Module,
        pair_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, pair_count, seq_len, feat_dim = x.shape
        encoded = encoder(x.reshape(batch_size * pair_count, seq_len, feat_dim))
        encoded = encoded.reshape(batch_size, pair_count, -1)
        return self._masked_average_features(encoded, pair_mask)

    def forward(
        self,
        audio: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        gait: torch.Tensor | None = None,
        personality: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = []

        if "audio" in self.modalities:
            if self.encoder_type == "hybrid_attn":
                features.append(self._encode_pairwise_sequences(audio, self.audio_enc, pair_mask))
            else:
                features.append(self.audio_enc(self._masked_average_sequences(audio, pair_mask)))

        if "video" in self.modalities:
            if self.encoder_type == "hybrid_attn":
                features.append(self._encode_pairwise_sequences(video, self.video_enc, pair_mask))
            else:
                features.append(self.video_enc(self._masked_average_sequences(video, pair_mask)))

        if "gait" in self.modalities:
            features.append(self.gait_enc(gait))

        if "personality" in self.modalities:
            features.append(self.pers_enc(personality))

        fused = torch.cat(features, dim=-1)
        logits = self.classifier(fused)
        if self.use_regression_head:
            return logits, self.regressor(fused).squeeze(-1)
        return logits
