import torch
import torch.nn as nn
from torch.nn import functional as F

from .samplers import make_unconditional_labels
from .timestep_schedules import sample_timesteps


class RectifiedFlowObjective:
    def __init__(
        self,
        device: torch.device,
        eps: float = 1e-5,
        mode: str = "uniform",
        mu: float = 0.0,
        sigma: float = 1.0,
        beta_a: float = 1.0,
        beta_b: float = 1.0,
    ):
        self.device = device
        self.eps = eps
        self.mode = mode
        self.mu = mu
        self.sigma = sigma
        self.beta_a = beta_a
        self.beta_b = beta_b

    def sample_t(self, batch_size: int) -> torch.Tensor:
        return sample_timesteps(
            batch_size,
            device=self.device,
            eps=self.eps,
            mode=self.mode,
            mu=self.mu,
            sigma=self.sigma,
            beta_a=self.beta_a,
            beta_b=self.beta_b,
        )

    def compute_loss(self, model: nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        t = self.sample_t(batch_size)
        t_view = t.view(batch_size, 1, 1, 1, 1)

        noise = torch.randn_like(images)
        x_t = t_view * images + (1.0 - t_view) * noise
        labels = make_unconditional_labels(batch_size, images.device)

        v_pred = model(x_t, t, labels)[:, : images.shape[1]]
        target_v = images - noise
        loss = F.mse_loss(v_pred, target_v)

        return {
            "loss": loss,
            "t": t,
            "x_t": x_t,
            "noise": noise,
            "target_v": target_v,
            "v_pred": v_pred,
            "x1_pred": x_t + (1.0 - t_view) * v_pred,
        }
