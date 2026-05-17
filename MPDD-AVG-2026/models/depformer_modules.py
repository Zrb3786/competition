from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_DEPFORMER_ENCODERS = {
    "bilstm_mean",
    "hybrid_attn",
    "report_lstm",
    "report_attn",
    "report_transformer",
    # aliases, useful when older scripts use DepFormer wording
    "dep_lstm",
    "depformer_lstm",
    "depformer_attn",
    "depformer_transformer",
}


def _canonical_encoder_name(encoder_type: str) -> str:
    alias_map = {
        "dep_lstm": "report_lstm",
        "depformer_lstm": "report_lstm",
        "depformer_attn": "report_attn",
        "depformer_transformer": "report_transformer",
    }
    return alias_map.get(encoder_type, encoder_type)


def make_seq_mask(x: torch.Tensor) -> torch.Tensor:
    """Build a [B, T] valid-step mask from a [B, T, C] sequence."""
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.any(torch.abs(x) > 0, dim=-1)
    all_masked = ~mask.any(dim=1)
    if all_masked.any():
        mask = mask.clone()
        mask[all_masked] = True
    return mask


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None, dim: int = 1) -> torch.Tensor:
    """Masked average for [B, T, C] or [B, P, T, C]-like tensors."""
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.to(dtype=x.dtype, device=x.device)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return (x * mask).sum(dim=dim) / denom


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            mask = mask.bool()
            all_masked = ~mask.any(dim=1)
            if all_masked.any():
                mask = mask.clone()
                mask[all_masked] = True
            scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (attn * x).sum(dim=1)


class BiLSTMSequenceEncoder(nn.Module):
    """Sequence version of the official bilstm_mean encoder.

    It keeps the official idea: optional pre-projection for high-dimensional
    features, linear projection to hidden_dim, then BiLSTM. Unlike the baseline
    encoder, this module returns all time steps [B, T, H], because DepFormer needs
    sequence features for cross-modal attention.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.5, pre_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even for BiLSTM, got {hidden_dim}")
        if pre_dim is not None and input_dim > pre_dim:
            self.pre_proj = nn.Linear(input_dim, pre_dim)
            lstm_in = pre_dim
        else:
            self.pre_proj = None
            lstm_in = input_dim
        self.proj = nn.Linear(lstm_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.pre_proj is not None:
            x = F.relu(self.pre_proj(x))
        x = F.relu(self.proj(x))
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.norm(x)


class HybridSequenceEncoder(nn.Module):
    """Sequence version of the official hybrid_attn encoder.

    The official HybridTemporalEncoder uses Conv1d -> BiLSTM -> attention pool.
    Here we keep Conv1d -> BiLSTM, but delay pooling until after BCT fusion.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3, pre_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even for BiLSTM, got {hidden_dim}")
        if pre_dim is not None and input_dim > pre_dim:
            self.pre_proj = nn.Linear(input_dim, pre_dim)
            conv_in = pre_dim
        else:
            self.pre_proj = None
            conv_in = input_dim
        self.conv1 = nn.Conv1d(conv_in, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.pre_proj is not None:
            x = self.pre_proj(x)
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = self.dropout(x)
        x = F.gelu(self.conv2(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.norm(x)


class ReportLSTMSequenceEncoder(nn.Module):
    """DepFormer-paper style encoder: LSTM followed by a dense tanh transform."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3, pre_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even for BiLSTM, got {hidden_dim}")
        if pre_dim is not None and input_dim > pre_dim:
            self.pre_proj = nn.Linear(input_dim, pre_dim)
            lstm_in = pre_dim
        else:
            self.pre_proj = None
            lstm_in = input_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.pre_proj is not None:
            x = F.relu(self.pre_proj(x))
        x, _ = self.lstm(x)
        x = torch.tanh(self.dense(self.dropout(x)))
        return self.norm(x)


class ReportAttentionSequenceEncoder(nn.Module):
    """Report-compatible LSTM encoder with a light temporal self-attention block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.3,
        pre_dim: Optional[int] = None,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.lstm_encoder = ReportLSTMSequenceEncoder(input_dim, hidden_dim, dropout, pre_dim)
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = make_seq_mask(x)
        h = self.lstm_encoder(x)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = self.norm1(h + self.dropout(attn_out))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class ReportTransformerSequenceEncoder(nn.Module):
    """Linear projection + TransformerEncoder for report-style temporal encoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.3,
        pre_dim: Optional[int] = None,
        num_heads: int = 2,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if pre_dim is not None and input_dim > pre_dim:
            self.pre_proj = nn.Sequential(nn.Linear(input_dim, pre_dim), nn.ReLU())
            proj_in = pre_dim
        else:
            self.pre_proj = None
            proj_in = input_dim
        self.proj = nn.Linear(proj_in, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mask = make_seq_mask(x)
        if self.pre_proj is not None:
            x = self.pre_proj(x)
        x = self.proj(x)
        x = self.encoder(x, src_key_padding_mask=~mask)
        return self.norm(x)


def build_sequence_encoder(
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    encoder_type: str,
    pre_dim: Optional[int] = None,
    num_heads: int = 2,
) -> nn.Module:
    encoder_type = _canonical_encoder_name(encoder_type)
    if encoder_type == "bilstm_mean":
        return BiLSTMSequenceEncoder(input_dim, hidden_dim, dropout, pre_dim=pre_dim)
    if encoder_type == "hybrid_attn":
        return HybridSequenceEncoder(input_dim, hidden_dim, dropout, pre_dim=pre_dim)
    if encoder_type == "report_lstm":
        return ReportLSTMSequenceEncoder(input_dim, hidden_dim, dropout, pre_dim=pre_dim)
    if encoder_type == "report_attn":
        return ReportAttentionSequenceEncoder(input_dim, hidden_dim, dropout, pre_dim=pre_dim, num_heads=num_heads)
    if encoder_type == "report_transformer":
        return ReportTransformerSequenceEncoder(input_dim, hidden_dim, dropout, pre_dim=pre_dim, num_heads=num_heads)
    raise ValueError(f"Unsupported DepFormer encoder_type={encoder_type}")


class CrossModalTransformerLayer(nn.Module):
    """One branch of the Bimodal Collaborative Transformer.

    target sequence provides Q; context sequence provides K/V.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 2, dropout: float = 0.3, ffn_mult: int = 4) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding_mask = None if context_mask is None else ~context_mask.bool()
        attn_out, _ = self.cross_attn(
            query=target,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(target + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class BimodalCollaborativeTransformer(nn.Module):
    """DepFormer BCT: two symmetric cross-modal branches."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.3,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.audio_layers = nn.ModuleList(
            [CrossModalTransformerLayer(hidden_dim, num_heads, dropout, ffn_mult) for _ in range(num_layers)]
        )
        self.video_layers = nn.ModuleList(
            [CrossModalTransformerLayer(hidden_dim, num_heads, dropout, ffn_mult) for _ in range(num_layers)]
        )
        self.out_norm_audio = nn.LayerNorm(hidden_dim)
        self.out_norm_video = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        audio_seq: torch.Tensor,
        video_seq: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_ctx = audio_seq
        video_ctx = video_seq
        for audio_layer, video_layer in zip(self.audio_layers, self.video_layers):
            next_audio = audio_layer(audio_ctx, video_ctx, context_mask=video_mask)
            next_video = video_layer(video_ctx, audio_ctx, context_mask=audio_mask)
            audio_ctx, video_ctx = next_audio, next_video
        return self.out_norm_audio(audio_ctx), self.out_norm_video(video_ctx)
