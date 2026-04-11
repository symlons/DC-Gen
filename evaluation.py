import torch
from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.utils import MetricReduction


def flatten_depth_to_batch(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.movedim(2, 1).contiguous()
    batch, depth, channels, height, width = tensor.shape
    return tensor.view(batch * depth, channels, height, width)


def evaluate(recon, target):
    spatial_dims = len(recon.shape) - 2
    max_val = max(float(target.max() - target.min()), 1e-8)
    psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.NONE)
    win_size = 3 if spatial_dims == 3 else 11
    
    min_spatial_size = min(recon.shape[-2:])
    if min_spatial_size < win_size: win_size = min(min_spatial_size, 3)
    
    ssim_metric = SSIMMetric(spatial_dims=spatial_dims, win_size=win_size, data_range=max_val, reduction=MetricReduction.NONE)
    psnr_value = psnr_metric(recon, target).mean().item()
    ssim_value = ssim_metric(recon, target).mean().item()
    
    slice_psnr_value, slice_ssim_value = psnr_value, ssim_value
    
    if spatial_dims == 3:
        B, _, D, _, _ = recon.shape
        recon_2d, target_2d = flatten_depth_to_batch(recon), flatten_depth_to_batch(target)
        slice_psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.NONE)
        
        min_2d_size = min(recon_2d.shape[-2:])
        slice_win_size = 11
        if min_2d_size < slice_win_size: slice_win_size = min(min_2d_size, 3)
        
        slice_ssim_metric = SSIMMetric(spatial_dims=2, win_size=slice_win_size, data_range=max_val, reduction=MetricReduction.NONE)
        slice_psnr_value = slice_psnr_metric(recon_2d, target_2d).view(B, D)
        slice_ssim_value = slice_ssim_metric(recon_2d, target_2d).view(B, D)
    
    return {"psnr": psnr_value, "ssim": ssim_value, "slice_psnr": slice_psnr_value, "slice_ssim": slice_ssim_value}
