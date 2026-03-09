from monai.metrics.utils import MetricReduction
from monai.metrics import PSNRMetric, SSIMMetric

def evaluate_3d(recon, target, loss=None):
    max_val = float(target.max() - target.min())
    
    psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.MEAN)
    psnr_metric(recon, target)
    psnr_value = psnr_metric.aggregate().item()
    
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=max_val, reduction=MetricReduction.MEAN)
    ssim_metric(recon, target)
    ssim_value = ssim_metric.aggregate().item()
    
    loss_value = loss.item() if loss is not None else None
    
    return loss_value, psnr_value, ssim_value

# def evaluate_slices(recon, target, loss_fn=None):
#     max_val = float(target.max() - target.min())
#     psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.MEAN)
#     ssim_metric = SSIMMetric(data_range=max_val, reduction=MetricReduction.MEAN)

#     psnr_vals, ssim_vals, loss_vals = [], [], []

#     for i in range(recon.shape[1]):
#         slice_recon, slice_target = recon[:, i], target[:, i]
#         psnr_metric(slice_recon, slice_target)
#         ssim_metric(slice_recon, slice_target)
#         psnr_vals.append(psnr_metric.aggregate().item())
#         ssim_vals.append(ssim_metric.aggregate().item())
#         if loss_fn is not None:
#             loss_vals.append(loss_fn(slice_recon, slice_target).item())

#     loss_val = sum(loss_vals)/len(loss_vals) if loss_vals else None
#     psnr_val = sum(psnr_vals)/len(psnr_vals)
#     ssim_val = sum(ssim_vals)/len(ssim_vals)
    
#     return loss_val, psnr_val, ssim_val