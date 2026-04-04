import torch
from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.utils import MetricReduction


def _flatten_depth_to_batch(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.movedim(2, 1).contiguous()
    batch, depth, channels, height, width = tensor.shape
    return tensor.view(batch * depth, channels, height, width)


def evaluate(recon, target, loss=None):
    spatial_dims = len(recon.shape) - 2
    max_val = max(float(target.max() - target.min()), 1e-8)

    psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.NONE)
    win_size = 3 if spatial_dims == 3 else 11
    ssim_metric = SSIMMetric(spatial_dims=spatial_dims, win_size=win_size, data_range=max_val, reduction=MetricReduction.NONE)

    psnr_value = psnr_metric(recon, target).mean().item()
    ssim_value = ssim_metric(recon, target).mean().item()
    loss_value = loss.item() if loss is not None else None
    slice_psnr_value = psnr_value
    slice_ssim_value = ssim_value

    if spatial_dims == 3:
        recon_2d = _flatten_depth_to_batch(recon)
        target_2d = _flatten_depth_to_batch(target)
        slice_psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.NONE)
        slice_ssim_metric = SSIMMetric(spatial_dims=2, win_size=11, data_range=max_val, reduction=MetricReduction.NONE)
        slice_psnr_value = slice_psnr_metric(recon_2d, target_2d).mean().item()
        slice_ssim_value = slice_ssim_metric(recon_2d, target_2d).mean().item()

    return loss_value, psnr_value, ssim_value, slice_psnr_value, slice_ssim_value