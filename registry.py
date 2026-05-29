import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CTVolumeDataset
from pytorch_msssim import MS_SSIM
from monai.losses import PerceptualLoss
from gan_loss import PatchGAN
from multigpu import rank0_print
import kornia

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        B, C, D, H, W = pred.shape

        pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        target_2d = target.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        grad_pred = kornia.filters.spatial_gradient(pred_2d)
        grad_target = kornia.filters.spatial_gradient(target_2d)

        return F.l1_loss(grad_pred, grad_target)

class MSSSIMLoss(nn.Module):
    def __init__(self, channel=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=channel)

    def forward(self, pred, target):
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        return 1 - self.ms_ssim(pred, target)


class GANLoss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        if cfg.objective.gan_loss_type != "hinge":
            raise ValueError(f"Unsupported gan_loss_type={cfg.objective.gan_loss_type!r}; only hinge is implemented")
        self.cfg = cfg
        self.patch_size = tuple(int(x) for x in cfg.objective.gan_patch_size)
        if len(self.patch_size) != 3 or any(size < 16 for size in self.patch_size):
            raise ValueError("gan_patch_size must be [D, H, W] with each dimension >= 16 for the 3D PatchGAN")
        self.gan_module = PatchGAN(
            in_channels=1,
            ndf=cfg.objective.gan_ndf,
            n_layers=3,
        ).to(device)

        self.discriminator_optimizer = torch.optim.Adam(
            self.gan_module.discriminator.parameters(),
            lr=(cfg.hparams.discriminator_learning_rate or (cfg.hparams.learning_rate * 0.1)),
            betas=(0.5, 0.999),
        )
        self.step_interval = cfg.objective.gan_discriminator_steps + 1
        rank0_print(
            f"[GAN] Initialized hinge patch discriminator "
            f"(ndf={cfg.objective.gan_ndf}, patch={self.patch_size}, d_steps={cfg.objective.gan_discriminator_steps})"
        )

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _paired_patch_crop(self, real: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if real.ndim != 5 or fake.ndim != 5:
            raise ValueError(f"PatchGAN expects [B, C, D, H, W], got real={tuple(real.shape)} fake={tuple(fake.shape)}")
        _, _, depth, height, width = real.shape
        crop_d, crop_h, crop_w = self.patch_size
        crop_d = min(crop_d, depth)
        crop_h = min(crop_h, height)
        crop_w = min(crop_w, width)

        def start(max_start: int) -> int:
            if max_start <= 0:
                return 0
            return int(torch.randint(max_start + 1, (), device=real.device).item())

        d0 = start(depth - crop_d)
        h0 = start(height - crop_h)
        w0 = start(width - crop_w)
        sl = (slice(None), slice(None), slice(d0, d0 + crop_d), slice(h0, h0 + crop_h), slice(w0, w0 + crop_w))
        return real[sl].contiguous(), fake[sl].contiguous()

    def step(self, real: torch.Tensor, fake: torch.Tensor, global_step: int):
        real_patch, fake_patch = self._paired_patch_crop(real, fake)
        is_gen_step = (global_step % self.step_interval == 0)

        if is_gen_step:
            self._set_requires_grad(self.gan_module.discriminator, False)
            g_loss = self.gan_module.g_loss(fake_patch)
            return g_loss, None

        self._set_requires_grad(self.gan_module.discriminator, True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        d_loss = self.gan_module.d_loss(real_patch, fake_patch.detach())
        d_loss.backward()
        self.discriminator_optimizer.step()
        return real.new_zeros(()), d_loss.detach()

dataset_registry = {
    "CTVolume": CTVolumeDataset,
}

def ssimWrapper(recon, batch):
    B, C, D, H, W = recon.shape
    recon_2d = recon.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    batch_2d = batch.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    loss = MSSSIMLoss(channel=1)
    return loss(recon_2d, batch_2d) 

loss_registry = {
    "l1": F.l1_loss,
    "mse": F.mse_loss,
    "ssim_loss": ssimWrapper,
    "perceptual": lambda device: PerceptualLoss(spatial_dims=3, network_type="vgg", is_fake_3d=True).to(device),
    "grad": GradientLoss
}
