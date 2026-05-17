from __future__ import annotations

from typing import Any

import torch.nn as nn

from .depformer_avp_v2 import DepFormerAVP
from .depformer_modules import SUPPORTED_DEPFORMER_ENCODERS
from .torchcat_baseline import TorchcatBaseline

MODEL_TYPES = ("baseline", "depformer")
BASELINE_ENCODERS = ("bilstm_mean", "hybrid_attn")
ALL_ENCODERS = tuple(sorted(set(BASELINE_ENCODERS) | set(SUPPORTED_DEPFORMER_ENCODERS)))


def build_model(model_type: str = "baseline", **kwargs: Any) -> nn.Module:
    """Build a model while preserving old baseline checkpoints.

    Extra DepFormer-only kwargs are stripped before building TorchcatBaseline, so
    old baseline behavior remains unchanged when model_type='baseline'.
    """
    model_type = model_type or kwargs.pop("model_type", "baseline")
    if "model_type" in kwargs:
        kwargs = dict(kwargs)
        kwargs.pop("model_type", None)

    if model_type == "baseline":
        baseline_kwargs = dict(kwargs)
        for key in (
            "num_bct_layers",
            "num_heads",
            "ffn_mult",
            "use_p_gate",
            "use_focal_head",
            "av_encode_pairwise",
        ):
            baseline_kwargs.pop(key, None)
        encoder_type = baseline_kwargs.get("encoder_type", "bilstm_mean")
        if encoder_type not in BASELINE_ENCODERS:
            raise ValueError(
                f"TorchcatBaseline only supports encoder_type={BASELINE_ENCODERS}, got {encoder_type}. "
                "Use --model_type depformer for report_lstm/report_attn/report_transformer."
            )
        return TorchcatBaseline(**baseline_kwargs)

    if model_type == "depformer":
        return DepFormerAVP(**kwargs)

    raise ValueError(f"Unknown model_type={model_type}; available={MODEL_TYPES}")
