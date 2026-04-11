import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = []
        sn = nn.utils.spectral_norm
        
        layers.append(sn(nn.Conv3d(in_channels * 2, ndf, kernel_size=4, stride=2, padding=1, bias=True)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            
            layers.append(sn(nn.Conv3d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=False)))
            layers.append(nn.InstanceNorm3d(nf))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        layers.append(sn(nn.Conv3d(nf, 1, kernel_size=4, stride=1, padding=1, bias=True)))
        self.main = nn.Sequential(*layers)
    
    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        x = torch.cat([real, fake], dim=1)
        return self.main(x)


class GANLoss(nn.Module):
    def __init__(self, loss_type: str = "hinge"):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "hinge":
            self.loss_fn = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def discriminator_loss(self, real_patches: torch.Tensor, fake_patches: torch.Tensor, discriminator: nn.Module) -> torch.Tensor:
        real_logits = discriminator(real_patches.detach(), real_patches.detach())
        fake_logits = discriminator(real_patches.detach(), fake_patches.detach())
        
        if self.loss_type == "bce":
            real_loss = self.loss_fn(real_logits, torch.ones_like(real_logits))
            fake_loss = self.loss_fn(fake_logits, torch.zeros_like(fake_logits))
            return real_loss + fake_loss
        else:
            # Hinge loss: max(0, 1 - real) + max(0, 1 + fake)
            real_loss = F.relu(1.0 - real_logits).mean()
            fake_loss = F.relu(1.0 + fake_logits).mean()
            return real_loss + fake_loss
    
    def generator_loss(self, real_patches: torch.Tensor, fake_patches: torch.Tensor, discriminator: nn.Module) -> torch.Tensor:
        fake_logits = discriminator(real_patches, fake_patches)
        if self.loss_type == "bce": return self.loss_fn(fake_logits, torch.ones_like(fake_logits))
        else: return -fake_logits.mean()


class LocalPatchGAN:
    def __init__(self, in_channels: int = 1, patch_size: Tuple[int, int, int] = (32, 32, 16), ndf: int = 64, loss_type: str = "hinge"):
        self.patch_size = patch_size
        self.discriminator = PatchDiscriminator(in_channels=in_channels, ndf=ndf, n_layers=3)
        self.gan_loss = GANLoss(loss_type=loss_type)
    
    def extract_patch_pairs(self, real: torch.Tensor, fake: torch.Tensor, num_patches: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, D, H, W = real.shape
        pd, ph, pw = self.patch_size
        
        real_patches = []
        fake_patches = []
        for _ in range(num_patches):
            d_start = torch.randint(0, max(1, D - pd), (B,))
            h_start = torch.randint(0, max(1, H - ph), (B,))
            w_start = torch.randint(0, max(1, W - pw), (B,))
            
            real_batch = []
            fake_batch = []
            for b in range(B):
                sl = (
                    slice(b, b+1),
                    slice(None),
                    slice(d_start[b], d_start[b]+pd),
                    slice(h_start[b], h_start[b]+ph),
                    slice(w_start[b], w_start[b]+pw),
                )
                real_batch.append(real[sl])
                fake_batch.append(fake[sl])
            real_patches.append(torch.cat(real_batch, dim=0))
            fake_patches.append(torch.cat(fake_batch, dim=0))
        
        return torch.cat(real_patches, dim=0), torch.cat(fake_patches, dim=0)
    
    def compute_discriminator_loss(self, real_volume: torch.Tensor, fake_volume: torch.Tensor) -> torch.Tensor:
        real_patches, fake_patches = self.extract_patch_pairs(real_volume, fake_volume)
        
        return self.gan_loss.discriminator_loss(real_patches, fake_patches, self.discriminator)
    
    def compute_generator_loss(self, real_volume: torch.Tensor, fake_volume: torch.Tensor) -> torch.Tensor:
        real_patches, fake_patches = self.extract_patch_pairs(real_volume, fake_volume)
        return self.gan_loss.generator_loss(real_patches, fake_patches, self.discriminator)
    
    def to(self, device):
        self.discriminator = self.discriminator.to(device)
        return self
