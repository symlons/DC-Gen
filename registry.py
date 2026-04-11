import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CTVolumeDataset
from pytorch_msssim import MS_SSIM
from monai.losses import PerceptualLoss
from gan_loss import LocalPatchGAN
from multigpu import rank0_print


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
        self.gan_module = LocalPatchGAN(
            in_channels=1,
            patch_size=tuple(cfg.objective.gan_patch_size),
            ndf=cfg.objective.gan_ndf,
            loss_type=cfg.objective.gan_loss_type,
        ).to(device)

        self.discriminator_optimizer = torch.optim.Adam(
            self.gan_module.discriminator.parameters(),
            lr=(cfg.hparams.discriminator_learning_rate or (cfg.hparams.learning_rate * 0.1)),
            betas=(0.5, 0.999),
        )

        self.step_interval = cfg.objective.gan_discriminator_steps + 1
        rank0_print(f"[GAN] Initialized patch discriminator on all ranks.")

    def step(self, real, fake, global_step):
        is_gen_step = (global_step % self.step_interval == 0)
        if is_gen_step: 
            return self.gan_module.compute_generator_loss(real, fake), None

        self.discriminator_optimizer.zero_grad()
        d_loss = self.gan_module.compute_discriminator_loss(real, fake.detach())
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
}
