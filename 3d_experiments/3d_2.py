import os
from collections import deque
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from monai.transforms import Compose, ScaleIntensity, Resize
from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.utils import MetricReduction

from omegaconf import OmegaConf
import wandb

from dc_gen.ae_model_zoo import DCAE_HF
from data import CTVolumeDataset
from viz import plot_training_curves, save_volumes, volumes_wandb
from config import get_default_config
from basics import save_checkpoint

def evaluate_3d(recon, target, loss=None):
    max_val = float(target.max() - target.min())
    
    psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.MEAN)
    psnr_metric(recon, target)
    psnr_value = psnr_metric.aggregate().item()
    
    ssim_metric = SSIMMetric(data_range=max_val, reduction=MetricReduction.MEAN)
    ssim_metric(recon, target)
    ssim_value = ssim_metric.aggregate().item()
    
    loss_value = loss.item() if loss is not None else None
    
    return loss_value, psnr_value, ssim_value

def evaluate_slices(recon, target, loss_fn=None):
    max_val = float(target.max() - target.min())
    psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.MEAN)
    ssim_metric = SSIMMetric(data_range=max_val, reduction=MetricReduction.MEAN)

    psnr_vals, ssim_vals, loss_vals = [], [], []

    for i in range(recon.shape[1]):
        slice_recon, slice_target = recon[:, i], target[:, i]
        psnr_metric(slice_recon, slice_target)
        ssim_metric(slice_recon, slice_target)
        psnr_vals.append(psnr_metric.aggregate().item())
        ssim_vals.append(ssim_metric.aggregate().item())
        if loss_fn is not None:
            loss_vals.append(loss_fn(slice_recon, slice_target).item())

    loss_val = sum(loss_vals)/len(loss_vals) if loss_vals else None
    psnr_val = sum(psnr_vals)/len(psnr_vals)
    ssim_val = sum(ssim_vals)/len(ssim_vals)
    
    return loss_val, psnr_val, ssim_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_override.yaml")
    args = parser.parse_args()

    global cfg
    cfg = get_default_config()
    if os.path.exists(args.config):
        yaml_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    cfg.training.device = torch.device(cfg.training.device)
    cfg.training.dtype = getattr(torch, cfg.training.dtype)

    os.makedirs(cfg.paths.artifact_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True))

    pipeline_3d = Compose([ScaleIntensity(minv=-1.0, maxv=1.0), Resize(spatial_size=[-1] + cfg.pipeline.resize_hw)])

    dataset_3d = CTVolumeDataset(cfg.paths.hdf_path, group_names=["Vol_full"], volume=True, n_slices=cfg.pipeline.n_slices, transform=pipeline_3d)
    loader_3d = DataLoader(dataset_3d, batch_size=cfg.training.batch_size, shuffle=cfg.training.shuffle_data)

    model = DCAE_HF(model_name=cfg.model.name).to(dtype=cfg.training.dtype, device=cfg.training.device)
    model.train()
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    loss_history, psnr_history, ssim_history = [], [], []
    checkpoint_queue = deque()
    log_file = os.path.join(cfg.paths.artifact_dir, "training_log.txt")
    global_step = 0

    log_metrics = cfg.wandb.enabled
    save_diff = cfg.logging.save_volumes and global_step % cfg.training.diff_save_every == 0
    save_ckpt = global_step % cfg.training.checkpoint_every == 0

    for epoch in range(cfg.training.num_iters):
        for batch_3d in loader_3d:
            if batch_3d.ndim == 4: batch_3d = batch_3d.unsqueeze(1)
            batch_3d = batch_3d.to(dtype=cfg.training.dtype, device=cfg.training.device)

            latent = model.encoder(batch_3d)
            recon = model.decoder(latent)

            loss = F.l1_loss(recon, batch_3d)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            loss_value, psnr_value, ssim_value = evaluate_3d(recon, batch_3d, loss)
            for h, v in zip([loss_history, psnr_history, ssim_history], [loss_value, psnr_value, ssim_value]): h.append(v)

            with open(log_file, "a") as f:
                f.write(f"iter {global_step}: loss={loss.item():.6f}, PSNR={psnr_value:.6f}, SSIM={ssim_value:.6f}\n")
                f.flush()
            print(f"Iter {global_step}: loss={loss.item():.6f}, PSNR={psnr_value:.6f}, SSIM={ssim_value:.6f}")

            if log_metrics: 
                wandb.log({"loss": loss.item(), "PSNR": psnr_value, "SSIM": ssim_value}, step=global_step)

            if save_diff:
                save_volumes(recon, batch_3d, cfg.paths.artifact_dir, global_step)
                volumes_wandb(cfg, recon, batch_3d, cfg.paths.artifact_dir, global_step)

            if save_ckpt: 
                save_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, global_step, checkpoint_queue, cfg.training.max_checkpoints)

            global_step += 1

    curve_path = os.path.join(cfg.paths.artifact_dir, f'training_curves_lr{cfg.training.lr}_bs{cfg.training.batch_size}.png')
    if cfg.logging.save_training_curves:
        plot_training_curves(loss_history, psnr_history, curve_path, cfg.training.lr, cfg.training.batch_size)
        if cfg.wandb.enabled: wandb.log({"training_curves": wandb.Image(curve_path)})

if __name__ == "__main__":
    main()