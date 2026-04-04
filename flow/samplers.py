import torch

from .timestep_schedules import make_time_grid

SOLVERS = {"euler", "heun"}


def make_unconditional_labels(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch_size, device=device, dtype=torch.long)


def predict_velocity(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    labels = make_unconditional_labels(x.shape[0], x.device)
    return model(x, t, labels)[:, : x.shape[1]]


@torch.no_grad()
def sample_velocity_model(
    model: torch.nn.Module,
    noise: torch.Tensor,
    sample_steps: int,
    solver: str = "euler",
    t_start: float = 0.0,
    t_end: float = 1.0,
    time_schedule: str = "linear",
    time_schedule_power: float = 2.0,
    progress_callback=None,
) -> torch.Tensor:
    if solver not in SOLVERS:
        raise ValueError(f"Unsupported solver: {solver}")

    x = noise
    times = make_time_grid(
        sample_steps,
        device=noise.device,
        dtype=noise.dtype,
        t_start=t_start,
        t_end=t_end,
        schedule=time_schedule,
        schedule_power=time_schedule_power,
    )
    batch_size = noise.shape[0]

    for step_idx, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
        dt = t1 - t0
        t = torch.full((batch_size,), t0.item(), device=noise.device, dtype=noise.dtype)
        v0 = predict_velocity(model, x, t)
        if solver == "euler":
            x = x + dt * v0
        else:
            x_euler = x + dt * v0
            t_next = torch.full((batch_size,), t1.item(), device=noise.device, dtype=noise.dtype)
            v1 = predict_velocity(model, x_euler, t_next)
            x = x + dt * 0.5 * (v0 + v1)
        
        if progress_callback is not None:
            progress_callback(step_idx + 1, sample_steps - 1)

    return x


@torch.no_grad()
def euler_sample(model: torch.nn.Module, noise: torch.Tensor, sample_steps: int, **kwargs) -> torch.Tensor:
    return sample_velocity_model(model, noise, sample_steps, solver="euler", **kwargs)


@torch.no_grad()
def heun_sample(model: torch.nn.Module, noise: torch.Tensor, sample_steps: int, **kwargs) -> torch.Tensor:
    return sample_velocity_model(model, noise, sample_steps, solver="heun", **kwargs)
