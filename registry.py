import torch
from data import CTVolumeDataset
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM

class MSSSIMLoss(torch.nn.Module):
    def __init__(self, channel=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=channel)

    def forward(self, pred, target):
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        return 1 - self.ms_ssim(pred, target)

dataset_registry = {
    "CTVolume": CTVolumeDataset,
}

loss_registry = {
    "l1": F.l1_loss,
    "mse": F.mse_loss,
    "ms_ssim": lambda: MSSSIMLoss(channel=1)
}
