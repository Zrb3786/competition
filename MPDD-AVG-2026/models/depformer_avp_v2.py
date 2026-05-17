from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .depformer_modules import (
    BimodalCollaborativeTransformer,
    SUPPORTED_DEPFORMER_ENCODERS,
    TemporalAttentionPool,
    build_sequence_encoder,
    make_seq_mask,
)
from .torchcat_baseline import PersonalityEncoder


class PairAttentionPool(nn.Module):
    """Attention pooling over interview pairs.

    Input shape:
      x    : [B, P, H]
      mask : [B, P], True for valid pair
    Output:
      pooled: [B, H]
    """

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
            mask = mask.bool().to(device=x.device)
            all_masked = ~mask.any(dim=1)
            if all_masked.any():
                mask = mask.clone()
                mask[all_masked] = True
            scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (attn * x).sum(dim=1)


class ModalityAttentionPool(nn.Module):
    """Attention pooling over modality tokens such as A/V/G/P."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        score = self.score(tokens).squeeze(-1)
        attn = torch.softmax(score, dim=1).unsqueeze(-1)
        return (tokens * attn).sum(dim=1)


class DepFormerAVP(nn.Module):
    """DepFormerAVP-v2 for MPDD-AVG 2026.

    Main changes compared with the previous AVG-P implementation:
      1. Keep interview pairs until after A/V BCT, instead of averaging pairs first.
      2. Use temporal attention pooling for each pair, not simple masked mean.
      3. Use pair attention pooling to learn which interview pair is important.
      4. Convert A/V/G/P into modality tokens and fuse them with a light Transformer.
      5. Keep the same external forward interface and return convention:
         - use_regression_head=True  -> (logits, reg_out)
         - return_aux=True and use_focal_head=True -> (logits, reg_out, focal_logits)

    This file is intended to replace models/depformer_avp.py directly.
    """

    SUBTRACKS = {
        "A-V+P": ["audio", "video", "personality"],
        "A-V-G+P": ["audio", "video", "gait", "personality"],
        "G+P": ["gait", "personality"],
    }
    ENCODER_TYPES = SUPPORTED_DEPFORMER_ENCODERS

    def __init__(
        self,
        subtrack: str = "A-V-G+P",
        num_classes: int = 3,
        is_regression: bool = False,
        use_regression_head: bool = True,
        audio_dim: int = 64,
        video_dim: int = 1000,
        gait_dim: int = 9,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        encoder_type: str = "report_lstm",
        num_bct_layers: int = 1,
        num_heads: int = 2,
        ffn_mult: int = 4,
        use_p_gate: bool = True,
        use_focal_head: bool = False,
        av_encode_pairwise: bool = True,
    ) -> None:
        super().__init__()
        if subtrack not in self.SUBTRACKS:
            raise ValueError(f"Unknown subtrack: {subtrack}")
        if encoder_type not in self.ENCODER_TYPES:
            raise ValueError(f"Unknown encoder_type for DepFormerAVP: {encoder_type}")
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even, got {hidden_dim}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.subtrack = subtrack
        self.modalities = self.SUBTRACKS[subtrack]
        self.encoder_type = encoder_type
        self.is_regression = is_regression
        self.use_regression_head = use_regression_head
        self.use_p_gate = use_p_gate
        self.use_focal_head = use_focal_head
        self.av_encode_pairwise = av_encode_pairwise
        self.hidden_dim = hidden_dim

        if "audio" in self.modalities:
            pre_audio = 128 if audio_dim > 128 else None
            self.audio_enc = build_sequence_encoder(
                input_dim=audio_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                encoder_type=encoder_type,
                pre_dim=pre_audio,
                num_heads=num_heads,
            )
            self.audio_time_pool = TemporalAttentionPool(hidden_dim)
            self.audio_pair_pool = PairAttentionPool(hidden_dim)

        if "video" in self.modalities:
            pre_video = 128 if video_dim > 128 else None
            self.video_enc = build_sequence_encoder(
                input_dim=video_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                encoder_type=encoder_type,
                pre_dim=pre_video,
                num_heads=num_heads,
            )
            self.video_time_pool = TemporalAttentionPool(hidden_dim)
            self.video_pair_pool = PairAttentionPool(hidden_dim)

        if "audio" in self.modalities and "video" in self.modalities:
            self.bct = BimodalCollaborativeTransformer(
                hidden_dim=hidden_dim,
                num_layers=num_bct_layers,
                num_heads=num_heads,
                dropout=dropout,
                ffn_mult=ffn_mult,
            )

        if "gait" in self.modalities:
            self.gait_enc = build_sequence_encoder(
                input_dim=gait_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                encoder_type="bilstm_mean",
                pre_dim=None,
                num_heads=num_heads,
            )
            self.gait_time_pool = TemporalAttentionPool(hidden_dim)

        if "personality" in self.modalities:
            self.pers_enc = PersonalityEncoder(1024, hidden_dim, dropout)

        max_modalities = len(self.modalities)
        self.modality_pos = nn.Parameter(torch.zeros(1, max_modalities, hidden_dim))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.modality_fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)
        self.modality_pool = ModalityAttentionPool(hidden_dim)

        if "personality" in self.modalities and use_p_gate:
            # Residual personality gate on the final fused representation.
            # 2 * sigmoid(0) = 1, so initialization does not alter the fused feature.
            self.p_gate = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.p_gate.weight)
            nn.init.zeros_(self.p_gate.bias)
        else:
            self.p_gate = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1 if is_regression else num_classes),
        )

        if use_regression_head:
            self.regressor = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        if use_focal_head:
            self.focal_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    @staticmethod
    def _safe_pair_mask(
        pair_mask: Optional[torch.Tensor],
        batch_size: int,
        pair_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if pair_mask is None:
            return torch.ones(batch_size, pair_count, dtype=torch.bool, device=device)
        pair_mask = pair_mask.bool().to(device=device)
        all_masked = ~pair_mask.any(dim=1)
        if all_masked.any():
            pair_mask = pair_mask.clone()
            pair_mask[all_masked] = True
        return pair_mask

    def _encode_single_modality_pairs(
        self,
        x: torch.Tensor,
        encoder: nn.Module,
        time_pool: nn.Module,
        pair_pool: nn.Module,
        pair_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode [B, P, T, C] into [B, H] for single-modality A or V usage."""
        if x is None:
            raise ValueError("Expected A/V tensor but got None")
        batch_size, pair_count, seq_len, feat_dim = x.shape
        safe_mask = self._safe_pair_mask(pair_mask, batch_size, pair_count, x.device)
        flat_x = x.reshape(batch_size * pair_count, seq_len, feat_dim)
        flat_seq = encoder(flat_x)
        flat_time_mask = make_seq_mask(flat_x)
        flat_pair_feat = time_pool(flat_seq, flat_time_mask)
        pair_feat = flat_pair_feat.reshape(batch_size, pair_count, -1)
        return pair_pool(pair_feat, safe_mask)

    def _encode_av_with_pairwise_bct(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        pair_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run BCT on every interview pair before pair-level pooling.

        audio/video: [B, P, T, C]
        returns: audio_feat, video_feat, both [B, H]
        """
        if audio is None or video is None:
            raise ValueError("Expected both audio and video tensors for A/V BCT")
        batch_size, pair_count, seq_len, audio_dim = audio.shape
        _, video_pair_count, video_seq_len, video_dim = video.shape
        if video_pair_count != pair_count or video_seq_len != seq_len:
            raise ValueError(
                f"Audio/video pair or sequence mismatch: audio={tuple(audio.shape)}, video={tuple(video.shape)}"
            )

        safe_mask = self._safe_pair_mask(pair_mask, batch_size, pair_count, audio.device)
        audio_flat = audio.reshape(batch_size * pair_count, seq_len, audio_dim)
        video_flat = video.reshape(batch_size * pair_count, seq_len, video_dim)

        audio_seq = self.audio_enc(audio_flat)
        video_seq = self.video_enc(video_flat)
        audio_time_mask = make_seq_mask(audio_flat)
        video_time_mask = make_seq_mask(video_flat)

        audio_bct, video_bct = self.bct(
            audio_seq,
            video_seq,
            audio_mask=audio_time_mask,
            video_mask=video_time_mask,
        )

        audio_pair = self.audio_time_pool(audio_bct, audio_time_mask).reshape(batch_size, pair_count, -1)
        video_pair = self.video_time_pool(video_bct, video_time_mask).reshape(batch_size, pair_count, -1)
        audio_feat = self.audio_pair_pool(audio_pair, safe_mask)
        video_feat = self.video_pair_pool(video_pair, safe_mask)
        return audio_feat, video_feat

    def forward(
        self,
        audio: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        gait: torch.Tensor | None = None,
        personality: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        tokens: list[torch.Tensor] = []
        p_feat: torch.Tensor | None = None

        if "audio" in self.modalities and "video" in self.modalities:
            # av_encode_pairwise is kept for checkpoint/config compatibility. In v2, True is the recommended path.
            if self.av_encode_pairwise:
                audio_feat, video_feat = self._encode_av_with_pairwise_bct(audio, video, pair_mask)
            else:
                # Compatibility fallback: still use pair attention, but without A/V BCT.
                audio_feat = self._encode_single_modality_pairs(
                    audio, self.audio_enc, self.audio_time_pool, self.audio_pair_pool, pair_mask
                )
                video_feat = self._encode_single_modality_pairs(
                    video, self.video_enc, self.video_time_pool, self.video_pair_pool, pair_mask
                )
            tokens.append(audio_feat)
            tokens.append(video_feat)

        elif "audio" in self.modalities:
            audio_feat = self._encode_single_modality_pairs(
                audio, self.audio_enc, self.audio_time_pool, self.audio_pair_pool, pair_mask
            )
            tokens.append(audio_feat)

        elif "video" in self.modalities:
            video_feat = self._encode_single_modality_pairs(
                video, self.video_enc, self.video_time_pool, self.video_pair_pool, pair_mask
            )
            tokens.append(video_feat)

        if "gait" in self.modalities:
            if gait is None:
                raise ValueError("Expected gait tensor but got None")
            gait_mask = make_seq_mask(gait)
            gait_seq = self.gait_enc(gait)
            gait_feat = self.gait_time_pool(gait_seq, gait_mask)
            tokens.append(gait_feat)

        if "personality" in self.modalities:
            if personality is None:
                raise ValueError("Expected personality tensor but got None")
            p_feat = self.pers_enc(personality)
            tokens.append(p_feat)

        if not tokens:
            raise ValueError("No modality tokens were produced.")

        token_tensor = torch.stack(tokens, dim=1)  # [B, M, H]
        token_tensor = token_tensor + self.modality_pos[:, : token_tensor.size(1), :]
        fused_tokens = self.modality_fusion(token_tensor)
        fused = self.modality_pool(fused_tokens)

        if p_feat is not None and self.p_gate is not None:
            gate = 2.0 * torch.sigmoid(self.p_gate(p_feat))
            fused = fused * gate

        logits = self.classifier(fused)
        outputs: list[torch.Tensor] = [logits]

        if self.use_regression_head:
            outputs.append(self.regressor(fused).squeeze(-1))

        if return_aux and self.use_focal_head:
            outputs.append(self.focal_classifier(fused))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
