import math

import torch

TRAIN_TIMESTEP_MODES = {"uniform", "logit_normal", "beta"}
SAMPLE_TIME_SCHEDULES = {"linear", "quadratic", "cosine", "power"}


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    eps: float = 1e-5,
    mode: str = "uniform",
    mu: float = 0.0,
    sigma: float = 1.0,
    beta_a: float = 1.0,
    beta_b: float = 1.0,
) -> torch.Tensor:
    if mode == "uniform":
        t = torch.rand(batch_size, device=device)
    elif mode == "logit_normal":
        t = torch.sigmoid(torch.randn(batch_size, device=device) * sigma + mu)
    elif mode == "beta":
        concentration1 = torch.tensor(beta_a, device=device)
        concentration0 = torch.tensor(beta_b, device=device)
        t = torch.distributions.Beta(concentration1, concentration0).sample((batch_size,))
    else:
        raise ValueError(f"Unsupported timestep sampling mode: {mode}")
    return t.clamp(eps, 1.0 - eps)


def make_time_grid(
    sample_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    t_start: float = 0.0,
    t_end: float = 1.0,
    schedule: str = "linear",
    schedule_power: float = 2.0,
) -> torch.Tensor:
    if sample_steps <= 0:
        raise ValueError("sample_steps must be positive.")

    u = torch.linspace(0.0, 1.0, sample_steps + 1, device=device, dtype=dtype)
    if schedule == "linear":
        tau = u
    elif schedule == "quadratic":
        tau = u.square()
    elif schedule == "cosine":
        tau = 1.0 - torch.cos(0.5 * math.pi * u)
    elif schedule == "power":
        tau = u.pow(schedule_power)
    else:
        raise ValueError(f"Unsupported sample time schedule: {schedule}")
    return t_start + (t_end - t_start) * tau
