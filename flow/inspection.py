from collections import OrderedDict

import torch
import torch.nn as nn

from .logging_utils import basic_tensor_stats_dict


def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def default_module_names(model: nn.Module) -> list[str]:
    model = unwrap_model(model)
    names = [name for name in ("x_embedder", "t_embedder", "y_embedder", "final_layer") if hasattr(model, name)]
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        block_indices = [0, len(model.blocks) // 2, len(model.blocks) - 1]
        names.append("blocks")
        names.extend(f"blocks.{index}" for index in OrderedDict.fromkeys(block_indices))
    return names


def resolve_module_names(model: nn.Module, module_names: list[str] | None) -> list[str]:
    model = unwrap_model(model)
    available = dict(model.named_modules())
    requested = default_module_names(model) if not module_names else module_names
    return [name for name in OrderedDict.fromkeys(requested) if name in available or any(k.startswith(f"{name}.") for k in dict(model.named_parameters()))]


def _matching_parameters(model: nn.Module, module_name: str) -> list[torch.nn.Parameter]:
    prefix = f"{module_name}."
    return [param for name, param in unwrap_model(model).named_parameters() if name == module_name or name.startswith(prefix)]


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _param_stats(params: list[torch.nn.Parameter], use_grads: bool) -> dict[str, float]:
    tensors = []
    for param in params:
        tensor = param.grad if use_grads else param.data
        if tensor is not None:
            tensors.append(tensor.detach().float().reshape(-1))
    if not tensors:
        return {}
    flat = torch.cat(tensors)
    return {
        "norm": flat.norm().item(),
        "rms": flat.square().mean().sqrt().item(),
        "mean_abs": flat.abs().mean().item(),
        "max_abs": flat.abs().max().item(),
    }


class ActivationInspector:
    def __init__(self, model: nn.Module, module_names: list[str]):
        self.model = unwrap_model(model)
        self.module_names = [name for name in module_names if name in dict(self.model.named_modules())]
        self.latest: dict[str, dict[str, float]] = {}
        self.handles = [
            dict(self.model.named_modules())[name].register_forward_hook(self._make_hook(name))
            for name in self.module_names
        ]

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is not None:
                self.latest[name] = basic_tensor_stats_dict("", tensor)

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class ModelInspector:
    def __init__(
        self,
        model: nn.Module,
        module_names: list[str] | None = None,
        capture_weights: bool = True,
        capture_gradients: bool = True,
        capture_activations: bool = True,
    ):
        self.model = model
        self.module_names = resolve_module_names(model, module_names)
        self.capture_weights = capture_weights
        self.capture_gradients = capture_gradients
        self.capture_activations = capture_activations
        self.activation_inspector = ActivationInspector(model, self.module_names) if capture_activations else None

    def collect(self) -> dict[str, dict[str, dict[str, float]]]:
        stats = {"weight": {}, "grad": {}, "activation": {}}
        for name in self.module_names:
            params = _matching_parameters(self.model, name)
            if self.capture_weights:
                weight_stats = _param_stats(params, use_grads=False)
                if weight_stats:
                    stats["weight"][name] = weight_stats
            if self.capture_gradients:
                grad_stats = _param_stats(params, use_grads=True)
                if grad_stats:
                    stats["grad"][name] = grad_stats
        if self.activation_inspector is not None:
            stats["activation"] = dict(self.activation_inspector.latest)
        return stats

    def close(self):
        if self.activation_inspector is not None:
            self.activation_inspector.close()


def flatten_inspection_stats(stats: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
    flat = {}
    for category, category_stats in stats.items():
        for module_name, module_stats in category_stats.items():
            for metric_name, value in module_stats.items():
                flat[f"inspect/{category}/{module_name}/{metric_name}"] = value
    return flat


def format_inspection_summary(stats: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    lines = []
    for category, category_stats in stats.items():
        if not category_stats:
            continue
        parts = []
        for module_name, module_stats in category_stats.items():
            if category == "activation":
                summary = f"std={module_stats.get('std', 0.0):.3e}"
            else:
                summary = f"norm={module_stats.get('norm', 0.0):.3e}"
            parts.append(f"{module_name}({summary})")
        lines.append(f"{category}: " + " | ".join(parts))
    return lines
