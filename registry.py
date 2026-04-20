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
        self.cfg = cfg
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
        rank0_print(f"[GAN] Initialized hinge patch discriminator (ndf={cfg.objective.gan_ndf})")

    def step(self, real: torch.Tensor, fake: torch.Tensor, global_step: int):
        is_gen_step = (global_step % self.step_interval == 0)
        
        if is_gen_step:
            g_loss = self.gan_module.g_loss(fake)
            return g_loss, None
        else:
            self.discriminator_optimizer.zero_grad()
            d_loss = self.gan_module.d_loss(real, fake.detach())
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
