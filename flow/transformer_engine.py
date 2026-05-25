from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch
import torch.nn as nn

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
except ImportError:  # pragma: no cover - optional dependency
    te = None
    DelayedScaling = None
    Format = None


def is_available() -> bool:
    return te is not None and DelayedScaling is not None and Format is not None


def _clone_linear_to_te(module: nn.Linear) -> nn.Module:
    if te is None:
        raise RuntimeError("Transformer Engine is not available.")
    te_linear = te.Linear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        params_dtype=module.weight.dtype,
        device=module.weight.device,
    )
    with torch.no_grad():
        te_linear.weight.copy_(module.weight)
        if module.bias is not None:
            te_linear.bias.copy_(module.bias)
    return te_linear


def wrap_linears(
    module: nn.Module,
    should_replace: Callable[[str, nn.Module], bool] | None = None,
    prefix: str = "",
) -> nn.Module:
    if not is_available():
        raise RuntimeError("Transformer Engine is not installed.")
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and (should_replace is None or should_replace(child_prefix, child)):
            setattr(module, name, _clone_linear_to_te(child))
            continue
        wrap_linears(child, should_replace=should_replace, prefix=child_prefix)
    return module


def build_fp8_recipe(cfg) -> DelayedScaling:
    if not is_available():
        raise RuntimeError("Transformer Engine is not installed.")
    fp8_format_name = getattr(cfg.transformer_engine, "recipe_format", "HYBRID")
    fp8_format = getattr(Format, fp8_format_name)
    return DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=cfg.transformer_engine.amax_history_len,
        amax_compute_algo=cfg.transformer_engine.amax_compute_algo,
    )


def fp8_autocast_context(enabled: bool, fp8_recipe: DelayedScaling | None):
    if not enabled or te is None or fp8_recipe is None:
        return nullcontext()
    if hasattr(te, "autocast"):
        return te.autocast(enabled=True, recipe=fp8_recipe)
    return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
