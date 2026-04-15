# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import diffusers
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from .aecore.models.dc_ae import (
    DCAE,
    DCAEConfig,
    dc_ae_f32_1_1,
    dc_ae_f32_1_5,
    dc_ae_f32c32,
    dc_ae_f64_1_5,
    dc_ae_f64c128,
    dc_ae_f128c512,
    dc_ae_f32c128,
    dc_ae_f32c256,
    build_dcae_cfg_from_variant
)
from .aecore.models.sd_vae import sd_vae_f8, sd_vae_f16, sd_vae_f32

__all__ = ["create_dc_ae_model_cfg", "DCAE_HF", "AutoencoderKL"]

def create_dc_ae_model_cfg(name: str = None, pretrained_path: Optional[str] = None, variant: dict = None) -> DCAEConfig:
    if variant is not None:
        return build_dcae_cfg_from_variant(variant, pretrained_path)
    assert name in REGISTERED_DCAE_MODEL, f"{name} is not supported"
    dc_ae_cls, default_pt_path, _ = REGISTERED_DCAE_MODEL[name]
    return dc_ae_cls(name, pretrained_path or default_pt_path)

REGISTERED_DCAE_MODEL: dict[str, tuple[Callable, Optional[str], Optional[str]]] = {
    "dc-ae-f32c32-in-1.0": (dc_ae_f32c32, None, "mit-han-lab"),
    "dc-ae-f32c32-in-1.0_2d": (dc_ae_f32c32, None, "mit-han-lab"),
    "dc-ae-f32c32-in-1.0_3d": (dc_ae_f32c32, None, "mit-han-lab"),
    "dc-ae-f32c32-in-1.0_3d-depth-last": (dc_ae_f32c32, None, "mit-han-lab"),
    "dc-ae-f32c32-in-1.0_3d-shallow": (dc_ae_f32c32, None, "mit-han-lab"),
    "dc-ae-f64c128-in-1.0": (dc_ae_f64c128, None, "mit-han-lab"),
    "dc-ae-f128c512-in-1.0": (dc_ae_f128c512, None, "mit-han-lab"),
    ## custom f64c128
    "dc-ae-f64c128-in-1.0_shallow_3d": (dc_ae_f64c128, None, "mit-han-lab"),

    #custom f32c128
    "dc-ae-f32c128-in-1.0_shallow_3d": (dc_ae_f32c128, None, "mit-han-lab"),
    "dc-ae-f32c128-in-1.0_deep_3d": (dc_ae_f32c128, None, "mit-han-lab"),
    #custom f32c256
    "dc-ae-f32c256-in-1.0_shallow_3d": (dc_ae_f32c256, None, "mit-han-lab"),
    #################################################################################################
    "dc-ae-f32c32-mix-1.0": (dc_ae_f32c32, None, "mit-han-lab"),
    "dc-ae-f64c128-mix-1.0": (dc_ae_f64c128, None, "mit-han-lab"),
    "dc-ae-f128c512-mix-1.0": (dc_ae_f128c512, None, "mit-han-lab"),
    #################################################################################################
    "dc-ae-f32c32-sana-1.0": (dc_ae_f32c32, None, "mit-han-lab"),
    "dc-ae-f32c32-sana-1.1": (dc_ae_f32c32, None, "mit-han-lab"),
    "dc-ae-lite-f32c32-sana-1.1": (dc_ae_f32c32, None, "mit-han-lab"),
    #################################################################################################
    "dc-ae-f32c32-in-1.0-256px": (dc_ae_f32c32, None, "mit-han-lab"),
    "dc-ae-f32c32-in-1.0-256px_3d": (dc_ae_f32c32, None, "mit-han-lab"),
    #################################################################################################
    "dc-ae-f32c128-1.5": (dc_ae_f32_1_5, None, "dc-ai"),
    "dc-ae-f64c128-1.5": (dc_ae_f64_1_5, None, "dc-ai"),
    "dc-ae-f32c128-1.1-c16": (
        dc_ae_f32_1_1,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_phase_3_16_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.1-c32": (
        dc_ae_f32_1_1,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_phase_3_32_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.1-c48": (
        dc_ae_f32_1_1,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_phase_3_48_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.1-c64": (
        dc_ae_f32_1_1,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_phase_3_64_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.1-c80": (
        dc_ae_f32_1_1,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_phase_3_80_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.1-c96": (
        dc_ae_f32_1_1,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_phase_3_96_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.1-c112": (
        dc_ae_f32_1_1,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_phase_3_112_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.1": (dc_ae_f32_1_1, "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_phase_3.pt", "dc-ai"),
    "dc-ae-f32c128-1.5-c16": (
        dc_ae_f32_1_5,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_1.5_phase_3_16_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.5-c32": (
        dc_ae_f32_1_5,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_1.5_phase_3_32_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.5-c48": (
        dc_ae_f32_1_5,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_1.5_phase_3_48_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.5-c64": (
        dc_ae_f32_1_5,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_1.5_phase_3_64_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.5-c80": (
        dc_ae_f32_1_5,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_1.5_phase_3_80_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.5-c96": (
        dc_ae_f32_1_5,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_1.5_phase_3_96_channel.pt",
        "dc-ai",
    ),
    "dc-ae-f32c128-1.5-c112": (
        dc_ae_f32_1_5,
        "assets/checkpoints/dc_ae_1.5_paper/figure_3/f32c128_1.5_phase_3_112_channel.pt",
        "dc-ai",
    ),
    #################################################################################################
}


class DCAE_HF(DCAE, PyTorchModelHubMixin):
    def __init__(self, model_name: str = None, model_cfg: DCAEConfig = None):
        cfg = model_cfg if model_cfg is not None else create_dc_ae_model_cfg(model_name)
        DCAE.__init__(self, cfg)

    def sample_c_prime(self, C, device, min_channels=16, step=4):
        possible = torch.arange(min_channels, C + 1, step, device=device)
        idx = torch.randint(0, possible.shape[0], (1,), device=device)
        return possible[idx]

    def forward(self, x: torch.Tensor, use_mask: bool = False):
        z = self.encode(x)

        if use_mask:
            B, C = z.shape[:2]

            c_prime = self.sample_c_prime(C, z.device)

            channel_idx = torch.arange(C, device=z.device).view(1, C, *([1] * (z.dim() - 2)))
            mask = (channel_idx < c_prime).to(z.dtype)

            z = z * mask

            recon = self.decode(z)
            return recon, c_prime

        return self.decode(z)

class AutoencoderKL(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        if self.model_name in ["stabilityai/sd-vae-ft-ema", "stabilityai/sdxl-vae"]:
            self.model = diffusers.models.AutoencoderKL.from_pretrained(self.model_name)
            self.spatial_compression_ratio = 8
        elif self.model_name == "flux-vae":
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
            self.model = diffusers.models.AutoencoderKL.from_pretrained(pipe.vae.config._name_or_path)
            self.spatial_compression_ratio = 8
        elif self.model_name == "sd3-vae":
            from diffusers import StableDiffusion3Pipeline

            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
            )
            self.model = diffusers.models.AutoencoderKL.from_pretrained(pipe.vae.config._name_or_path)
            self.spatial_compression_ratio = 8
        elif self.model_name in ["asymmetric-autoencoder-kl-x-1-5", "asymmetric-autoencoder-kl-x-2"]:
            self.model = diffusers.models.AsymmetricAutoencoderKL.from_pretrained(f"cross-attention/{self.model_name}")
            self.spatial_compression_ratio = 8
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name in [
            "stabilityai/sd-vae-ft-ema",
            "stabilityai/sdxl-vae",
            "flux-vae",
            "sd3-vae",
            "asymmetric-autoencoder-kl-x-1-5",
            "asymmetric-autoencoder-kl-x-2",
        ]:
            return self.model.encode(x).latent_dist.sample()
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.model_name in [
            "stabilityai/sd-vae-ft-ema",
            "stabilityai/sdxl-vae",
            "flux-vae",
            "sd3-vae",
            "asymmetric-autoencoder-kl-x-1-5",
            "asymmetric-autoencoder-kl-x-2",
        ]:
            return self.model.decode(latent).sample
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")


REGISTERED_SD_VAE_MODEL: dict[str, tuple[Callable, str]] = {
    "sd-vae-f8": (sd_vae_f8, "assets/checkpoints/ae/sd-vae-kl-f8.ckpt"),
    "sd-vae-f16": (sd_vae_f16, "assets/checkpoints/ae/sd-vae-kl-f16.ckpt"),
    "sd-vae-f32": (sd_vae_f32, "assets/checkpoints/ae/sd-vae-kl-f32.ckpt"),
    "mar-vae-f16": (sd_vae_f16, "assets/checkpoints/ae/mar-vae-kl-f16.ckpt"),
    "vavae-imagenet256-f16d32-dinov2": (sd_vae_f16, "assets/checkpoints/ae/vavae-imagenet256-f16d32-dinov2.pt"),
}

REGISTERED_DCAE_DIFFUSERS_MODEL: set[str] = {
    "dc-ae-f32c32-in-1.0-diffusers",
    "dc-ae-f64c128-in-1.0-diffusers",
    "dc-ae-f128c512-in-1.0-diffusers",
    "dc-ae-f32c32-mix-1.0-diffusers",
    "dc-ae-f64c128-mix-1.0-diffusers",
    "dc-ae-f128c512-mix-1.0-diffusers",
    "dc-ae-f32c32-sana-1.0-diffusers",
    "dc-ae-f32c32-sana-1.1-diffusers",
}
