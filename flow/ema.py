import copy
import math
import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def create_ema_model(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(unwrap_model(model)).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad = False
    return ema_model


def update_ema_warmup(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    global_step: int,
    decay: float = 0.9999,
    warmup_steps: int = 2000,
):
    ema_model = unwrap_model(ema_model)
    model = unwrap_model(model)
    effective_decay = decay * (1 - math.exp(-global_step / warmup_steps)) if warmup_steps > 0 else decay

    ema_params = []
    model_params = []
    model_sd = model.state_dict()
    for key, value in ema_model.state_dict().items():
        if value.dtype.is_floating_point:
            ema_params.append(value)
            model_params.append(model_sd[key].detach())

    one_minus_decay = 1.0 - effective_decay
    diff = torch._foreach_sub(ema_params, model_params)
    torch._foreach_sub_(ema_params, torch._foreach_mul(diff, one_minus_decay))


def update_ema(ema_model, model, decay=0.9999):
    ema_model = unwrap_model(ema_model)
    model = unwrap_model(model)
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()): # todo use c++ foreach
        if model_param.requires_grad:
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
