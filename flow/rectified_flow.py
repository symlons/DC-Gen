import math

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
        latent_mean: float = 0.0,
        latent_std: float = 1.0,
        tweo_enabled: bool = False,
        tweo_weight: float = 0.01,
        tweo_tau: float = 3.0,
        tweo_power: float = 4.0,
        tweo_eps: float = 1e-6,
        tweo_schedule: str = "constant",
        log_block_activations: bool = True,
        max_steps: int | None = None,
    ):
        self.device = device
        self.eps = eps
        self.mode = mode
        self.mu = mu
        self.sigma = sigma
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.tweo_enabled = tweo_enabled
        self.tweo_weight = tweo_weight
        self.tweo_tau = tweo_tau
        self.tweo_power = tweo_power
        self.tweo_eps = tweo_eps
        self.tweo_schedule = tweo_schedule
        self.log_block_activations = log_block_activations
        self.max_steps = max_steps

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

    def tweo_lambda(self, global_step: int | None = None) -> float:
        if not self.tweo_enabled:
            return 0.0
        if self.tweo_schedule == "constant" or global_step is None or self.max_steps is None or self.max_steps <= 0:
            return self.tweo_weight
        if self.tweo_schedule == "cosine":
            progress = min(max(global_step / self.max_steps, 0.0), 1.0)
            return 0.5 * self.tweo_weight * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported TWEO schedule: {self.tweo_schedule!r}")

    def compute_tweo_loss(self, activations: list[torch.Tensor]) -> torch.Tensor:
        if not activations:
            raise ValueError("TWEO is enabled but no block activations were returned by the model.")
        scale = self.tweo_tau + self.tweo_eps
        terms = [(activation.abs() / scale).pow(self.tweo_power).mean() for activation in activations]
        return torch.stack(terms).mean()

    @staticmethod
    def compute_activation_stats(activations: list[torch.Tensor], like: torch.Tensor) -> dict[str, torch.Tensor]:
        if not activations:
            zero = like.new_zeros(())
            return {
                "block_activation_abs_max": zero,
                "block_activation_abs_mean": zero,
            }
        abs_activations = [activation.detach().abs() for activation in activations]
        return {
            "block_activation_abs_max": torch.stack([activation.max() for activation in abs_activations]).max(),
            "block_activation_abs_mean": torch.stack([activation.mean() for activation in abs_activations]).mean(),
        }

    def compute_loss(self, model: nn.Module, images: torch.Tensor, global_step: int | None = None) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        t = self.sample_t(batch_size)
        t_view = t.view(batch_size, 1, 1, 1, 1)

        # Normalize latents
        images_normalized = (images - self.latent_mean) / self.latent_std

        noise = torch.randn_like(images_normalized)
        x_t = t_view * images_normalized + (1.0 - t_view) * noise
        labels = make_unconditional_labels(batch_size, images.device)

        return_block_activations = self.tweo_enabled or self.log_block_activations
        model_output = model(x_t, t, labels, return_block_activations=return_block_activations)
        if return_block_activations:
            model_output, block_activations = model_output
        else:
            block_activations = []

        v_pred = model_output[:, : images_normalized.shape[1]]
        target_v = images_normalized - noise
        flow_loss = F.mse_loss(v_pred, target_v)
        tweo_loss = self.compute_tweo_loss(block_activations) if self.tweo_enabled else flow_loss.new_zeros(())
        tweo_weight = self.tweo_lambda(global_step)
        loss = flow_loss + tweo_weight * tweo_loss
        activation_stats = self.compute_activation_stats(block_activations, flow_loss)

        return {
            "loss": loss,
            "flow_loss": flow_loss,
            "tweo_loss": tweo_loss,
            "tweo_weight": flow_loss.new_tensor(tweo_weight),
            **activation_stats,
            "t": t, # timesteps sampled within batch
            "x_t": x_t, # interpolated sample up to timestep t
            "noise": noise,
            "target_v": target_v, # ground truth
            "v_pred": v_pred, # models velocity prediction
            "x1_pred": x_t + (1.0 - t_view) * v_pred, # "reconstruction"
        }
