import os
from collections import deque
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import wandb

from monai.transforms import Compose, ScaleIntensity, Resize
from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.utils import MetricReduction
from omegaconf import OmegaConf

from dc_gen.ae_model_zoo import DCAE_HF
from data import CTVolumeDataset
from config import get_default_config
from basics import save_checkpoint, load_checkpoint, to_numpy

dataset_registry = {"CTVolume": CTVolumeDataset}
loss_registry = {"l1": F.l1_loss, "mse": F.mse_loss}

def evaluate_2d(recon, target, loss=None):
    max_val = float(target.max() - target.min())
    psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.NONE)
    ssim_metric = SSIMMetric(spatial_dims=2, data_range=max_val, reduction=MetricReduction.NONE)

    psnr_values = psnr_metric(recon, target)
    ssim_values = ssim_metric(recon, target)

    psnr_value = psnr_values.mean().item()
    ssim_value = ssim_values.mean().item()
    loss_value = loss.item() if loss is not None else None

    return loss_value, psnr_value, ssim_value

def plot_training_curves(loss_history, psnr_history, save_path, lr, batch_size):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(loss_history, 'r-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.plot(psnr_history, 'b-')
    ax2.set_ylabel('PSNR', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    fig.tight_layout()
    plt.title(f'Training Curves (lr={lr}, batch_size={batch_size})')
    plt.savefig(save_path, dpi=300)
    plt.close()

def diff_visualization(gt, recon, save_path, title_suffix=""):
    gt = gt.squeeze().cpu().numpy() if torch.is_tensor(gt) else gt.squeeze()
    recon = recon.squeeze().cpu().numpy() if torch.is_tensor(recon) else recon.squeeze()
    B = 1 if gt.ndim == 2 else gt.shape[0]
    cols = 3
    rows = B

    diff = gt - recon
    diff_abs = np.max(np.abs(diff))
    vmin = min(gt.min(), recon.min())
    vmax = max(gt.max(), recon.max())

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(B):
        axes[i, 0].imshow(gt if B == 1 else gt[i], cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Ground Truth {title_suffix} (Sample {i})")

        axes[i, 1].imshow(recon if B == 1 else recon[i], cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Reconstruction {title_suffix} (Sample {i})")

        im2 = axes[i, 2].imshow(diff if B == 1 else diff[i], cmap='bwr', vmin=-diff_abs, vmax=diff_abs)
        axes[i, 2].axis('off')
        axes[i, 2].set_title(f"Difference Map {title_suffix} (Sample {i})")

    fig.colorbar(im2, ax=axes.ravel().tolist(), location='right', shrink=0.85, pad=0.02, label="Difference Intensity")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_volumes(recon, batch_2d, artifact_dir, it):
    batch_size = batch_2d.shape[0]
    for b in range(batch_size):
        recon_slice = to_numpy(recon[b, 0])
        gt_slice = to_numpy(batch_2d[b, 0])
        save_path = os.path.join(artifact_dir, f"slice_diff_iter{it}_b{b}.png")
        diff_visualization(gt_slice, recon_slice, save_path, title_suffix=f"Iter {it} B{b}")

def volumes_wandb(cfg, recon, gt, artifact_dir, global_step, log_images=True, batch_indices=None):
    if batch_indices is None:
        batch_indices = [0]
    for idx in batch_indices:
        save_volumes(recon[idx:idx+1], gt[idx:idx+1], artifact_dir, f"{global_step}_b{idx}")
        if log_images and cfg.wandb.enabled:
            save_path = os.path.join(artifact_dir, f"slice_diff_iter{global_step}_b{idx}.png")
            wandb.log({f"slice_diff_iter_{global_step}_b{idx}": wandb.Image(save_path)}, step=global_step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    global cfg
    cfg = get_default_config()
    if os.path.exists(args.config):
        yaml_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    device = torch.device(cfg.training.device)
    dtype = getattr(torch, cfg.training.dtype)

    os.makedirs(cfg.paths.artifact_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True))

    pipeline_2d = Compose([ScaleIntensity(minv=-1.0, maxv=1.0), Resize(spatial_size=cfg.pipeline.resize_hw)])

    dataset_cls = dataset_registry[cfg.dataset.name]
    dataset_2d = dataset_cls(
        cfg.paths.hdf_path,
        group_names=cfg.dataset.group_names,
        volume=False,
        n_slices=cfg.pipeline.n_slices,
        transform=pipeline_2d
    )
    loader_2d = DataLoader(dataset_2d, batch_size=cfg.training.batch_size, shuffle=cfg.training.shuffle_data, pin_memory=True, num_workers=8, prefetch_factor=2)

    model = DCAE_HF(model_name=cfg.model.name).to(dtype=dtype, device=device)
    model.train()
    # model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=1e-1)
    global_step = load_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, device)

    loss_history, psnr_history, ssim_history = [], [], []
    checkpoint_queue = deque()
    log_file = os.path.join(cfg.paths.artifact_dir, "training_log.txt")
    num_epochs = cfg.training.num_epochs

    for epoch in range(num_epochs):
        print("epoch: ", epoch, "out of ", num_epochs)
        print("dataset size: ", len(dataset_2d))
        for batch_2d in loader_2d:
            log_metrics = cfg.wandb.enabled
            save_diff = cfg.logging.save_volumes and global_step % cfg.training.diff_save_every == 0
            save_ckpt = global_step % cfg.training.checkpoint_every == 0

            batch_2d = batch_2d.to(dtype=dtype, device=device, non_blocking=True)
            latent = model.encoder(batch_2d)
            recon = model.decoder(latent)

            loss = loss_registry[cfg.training.loss_fn](recon, batch_2d)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            loss_val, psnr_val, ssim_val = evaluate_2d(recon, batch_2d, loss)
            for h, v in zip([loss_history, psnr_history, ssim_history], [loss_val, psnr_val, ssim_val]):
                h.append(v)

            try:
                with open(log_file, "a") as f:
                    f.write(f"iter {global_step}: loss={loss.item():.6f}, PSNR={psnr_val:.6f}, SSIM={ssim_val:.6f}\n")
            except Exception as e:
                print(f"[WARNING] Failed to write to log file {log_file}: {e}")

            print(f"Iter {global_step}: loss={loss.item():.6f}, PSNR={psnr_val:.6f}, SSIM={ssim_val:.6f}")

            if log_metrics:
                wandb.log({"loss": loss.item(), "PSNR": psnr_val, "SSIM": ssim_val}, step=global_step)
            if save_diff:
                save_volumes(recon, batch_2d, cfg.paths.artifact_dir, global_step)
                volumes_wandb(cfg, recon, batch_2d, cfg.paths.artifact_dir, global_step)
            if save_ckpt:
                save_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, global_step, checkpoint_queue, cfg.training.max_checkpoints)

            global_step += 1

    curve_path = os.path.join(cfg.paths.artifact_dir, f'training_curves_lr{cfg.training.lr}_bs{cfg.training.batch_size}.png')
    if cfg.logging.save_training_curves:
        plot_training_curves(loss_history, psnr_history, curve_path, cfg.training.lr, cfg.training.batch_size)
        if cfg.wandb.enabled:
            wandb.log({"training_curves": wandb.Image(curve_path)})

if __name__ == "__main__":
    main()