import torch
from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.utils import MetricReduction

def evaluate(recon, target, loss=None):
    spatial_dims = len(recon.shape) - 2
    max_val = float(target.max() - target.min())

    psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.NONE)
    ssim_metric = SSIMMetric(spatial_dims=spatial_dims, data_range=max_val, reduction=MetricReduction.NONE)

    psnr_value = psnr_metric(recon, target).mean().item()
    ssim_value = ssim_metric(recon, target).mean().item()
    loss_value = loss.item() if loss is not None else None

    return loss_value, psnr_value, ssim_value
