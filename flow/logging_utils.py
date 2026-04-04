from collections.abc import Sequence

import torch


def basic_tensor_stats_dict(name: str, tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.detach().float()
    prefix = f"{name}/" if name else ""
    return {
        f"{prefix}mean": tensor.mean().item(),
        f"{prefix}std": tensor.std(unbiased=False).item(),
        f"{prefix}rms": tensor.square().mean().sqrt().item(),
        f"{prefix}mean_abs": tensor.abs().mean().item(),
        f"{prefix}max_abs": tensor.abs().max().item(),
    }


def tensor_stats_dict(name: str, tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.detach().float()
    quantiles = torch.quantile(
        tensor.flatten(),
        torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=tensor.device),
    )
    return {
        f"{name}/mean": tensor.mean().item(),
        f"{name}/std": tensor.std(unbiased=False).item(),
        f"{name}/min": tensor.min().item(),
        f"{name}/max": tensor.max().item(),
        f"{name}/q01": quantiles[0].item(),
        f"{name}/q05": quantiles[1].item(),
        f"{name}/median": quantiles[2].item(),
        f"{name}/q95": quantiles[3].item(),
        f"{name}/q99": quantiles[4].item(),
    }


def format_kv_block(title: str, rows: Sequence[tuple[str, object]]) -> str:
    width = max(len(key) for key, _ in rows)
    lines = [title]
    lines.extend(f"  {key:<{width}} : {value}" for key, value in rows)
    return "\n".join(lines)


def format_step_log(step: int, epoch: int, metrics: dict[str, float]) -> str:
    parts = [f"step={step}", f"epoch={epoch}"]
    parts.extend(f"{key}={_format_value(value)}" for key, value in metrics.items())
    return " | ".join(parts)


def _format_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if value.is_integer():
            return str(int(value))
        if abs(value) >= 1e4 or abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def append_log(log_file: Optional[str], *lines: str):
    if not log_file:
        return
    with open(log_file, "a") as f:
        for line in lines:
            f.write(line + "\n")


def log_rank0(message: str, log_file: Optional[str] = None):
    rank0_print(message)
    append_log(log_file, message)
