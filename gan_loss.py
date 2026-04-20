import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.norm1 = nn.InstanceNorm3d(channels)
        self.conv2 = nn.utils.spectral_norm(nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.norm2 = nn.InstanceNorm3d(channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, ndf: int = 64, n_layers: int = 4):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers = []
       
        layers.append(sn(nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
       
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
           
            layers.append(sn(nn.Conv3d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=False)))
            layers.append(nn.InstanceNorm3d(nf))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
           
            layers.append(ResidualBlock(nf))
       
        layers.append(sn(nn.Conv3d(nf, 1, kernel_size=4, stride=1, padding=1, bias=False)))
        self.main = nn.Sequential(*layers)
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class PatchGAN:
    def __init__(self, in_channels: int = 1, ndf: int = 64, n_layers: int = 4):
        self.discriminator = PatchDiscriminator(in_channels=in_channels, ndf=ndf, n_layers=n_layers)
        self.gan_loss = _HingeGANLoss()
   
    def d_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return self.gan_loss.discriminator_loss(real, fake, self.discriminator)
   
    def g_loss(self, fake: torch.Tensor) -> torch.Tensor:
        return self.gan_loss.generator_loss(fake, self.discriminator)
   
    def to(self, device):
        self.discriminator = self.discriminator.to(device)
        return self


class _HingeGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
   
    def discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor, discriminator: nn.Module) -> torch.Tensor:
        real_logits = discriminator(real.detach())
        fake_logits = discriminator(fake.detach())
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return real_loss + fake_loss
   
    def generator_loss(self, fake: torch.Tensor, discriminator: nn.Module) -> torch.Tensor:
        fake_logits = discriminator(fake)
        return -fake_logits.mean()
