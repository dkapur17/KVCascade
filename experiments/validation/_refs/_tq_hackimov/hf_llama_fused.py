"""
Backward-compatible entry points for decoder fused attention.

Implementation lives in :mod:`turboquant.hf_fused_attention` (Llama, Mistral, Qwen2, Gemma2, Phi3, Cohere, …).
"""

from __future__ import annotations

import warnings
from typing import Any, List

import torch.nn as nn

from .core import TurboQuantProd
from .hf_fused_attention import (
    TurboQuantLlamaAttention,
    install_turboquant_fused_attention,
    triton_cuda_available,
    uninstall_turboquant_fused_attention,
)

install_decoder_fused_attention = install_turboquant_fused_attention
uninstall_decoder_fused_attention = uninstall_turboquant_fused_attention


def _inner_llama_stack(model: nn.Module) -> nn.Module:
    inner = model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        inner = model.model
    if not hasattr(inner, "layers"):
        raise TypeError("Expected a decoder-only HF model with .model.layers or .layers")
    return inner


def install_turboquant_llama_attention(
    model: nn.Module,
    quantizer: TurboQuantProd,
    *,
    allow_attention_subclass: bool = False,
) -> None:
    """Same as ``install_turboquant_fused_attention(..., architecture=\"auto\")``."""
    install_turboquant_fused_attention(
        model,
        quantizer,
        architecture="auto",
        allow_attention_subclass=allow_attention_subclass,
    )


def uninstall_turboquant_llama_attention(model: nn.Module) -> None:
    uninstall_turboquant_fused_attention(model)


def apply_llama_turboquant_fused_patch(model: nn.Module, quantizer: TurboQuantProd) -> None:
    warnings.warn(
        "apply_llama_turboquant_fused_patch is deprecated; use install_turboquant_llama_attention",
        DeprecationWarning,
        stacklevel=2,
    )
    install_turboquant_llama_attention(model, quantizer)


def remove_llama_turboquant_fused_patch(model: nn.Module) -> None:
    warnings.warn(
        "remove_llama_turboquant_fused_patch is deprecated; use uninstall_turboquant_llama_attention",
        DeprecationWarning,
        stacklevel=2,
    )
    uninstall_turboquant_llama_attention(model)


def _iter_llama_self_attn_modules(model: nn.Module) -> List[Any]:
    return [layer.self_attn for layer in _inner_llama_stack(model).layers]
