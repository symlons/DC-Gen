import matplotlib.pyplot as plt
import numpy as np
from basics import to_numpy
import nibabel as nib
import wandb
import os

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
    if gt.ndim == 2:
        gt = gt[None, ...]
        recon = recon[None, ...]
    B = gt.shape[0]
    cols = 3
    rows = B

    diff = gt - recon
    diff_abs = np.max(np.abs(diff))

    vmin = min(gt.min(), recon.min())
    vmax = max(gt.max(), recon.max())

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]  # make it 2D for uniform indexing

    for i in range(B):
        axes[i, 0].imshow(gt[i], cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Ground Truth {title_suffix} (Sample {i})")

        axes[i, 1].imshow(recon[i], cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Reconstruction {title_suffix} (Sample {i})")

        im2 = axes[i, 2].imshow(diff[i], cmap='bwr', vmin=-diff_abs, vmax=diff_abs)
        axes[i, 2].axis('off')
        axes[i, 2].set_title(f"Difference Map {title_suffix} (Sample {i})")

    fig.colorbar(im2, ax=axes.ravel().tolist(), location='right', shrink=0.85, pad=0.02, label="Difference Intensity")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_volumes(recon, batch_3d, artifact_dir, it):
    batch_size = batch_3d.shape[0]
    depth_center = batch_3d.shape[2] // 2

    for b in range(batch_size):
        recon_slice = to_numpy(recon[b, 0, depth_center])
        gt_slice = to_numpy(batch_3d[b, 0, depth_center])

        save_path = os.path.join(artifact_dir, f"central_slice_diff_iter{it}_b{b}.png")
        diff_visualization(gt_slice, recon_slice, save_path, title_suffix=f"Iter {it} B{b}")

        recon_cpu = to_numpy(recon[b, 0])
        gt_cpu = to_numpy(batch_3d[b, 0])
        diff_cpu = np.abs(gt_cpu - recon_cpu)

        nib.save(nib.Nifti1Image(recon_cpu, affine=np.eye(4)), os.path.join(artifact_dir, f"recon_iter{it}_b{b}.nii.gz"))
        nib.save(nib.Nifti1Image(gt_cpu, affine=np.eye(4)), os.path.join(artifact_dir, f"gt_iter{it}_b{b}.nii.gz"))
        nib.save(nib.Nifti1Image(diff_cpu, affine=np.eye(4)), os.path.join(artifact_dir, f"diff_iter{it}_b{b}.nii.gz"))

def volumes_wandb(cfg, recon, gt, artifact_dir, global_step, log_images=True, batch_indices=None):
    if batch_indices is None:
        batch_indices = [0]

    for idx in batch_indices:
        save_volumes(recon[idx:idx+1], gt[idx:idx+1], artifact_dir, f"{global_step}_b{idx}")

        if log_images and cfg.wandb.enabled:
            depth_center = gt.shape[2] // 2
            recon_slice = to_numpy(recon[idx, 0, depth_center])
            gt_slice = to_numpy(gt[idx, 0, depth_center])

            save_path = os.path.join(artifact_dir, f"central_slice_diff_iter{global_step}_b{idx}.png")
            diff_visualization(gt_slice, recon_slice, save_path, title_suffix=f"Iter {global_step} B{idx}")
            wandb.log({
                f"central_slice_diff_iter_{global_step}_b{idx}": wandb.Image(save_path)
            }, step=global_step)

            # Save full 3D volumes as NIfTI files
            recon_nifti_path = os.path.join(artifact_dir, f"recon_iter{global_step}_b{idx}.nii.gz")
            gt_nifti_path = os.path.join(artifact_dir, f"gt_iter{global_step}_b{idx}.nii.gz")
            nib.save(nib.Nifti1Image(to_numpy(recon[idx, 0]), np.eye(4)), recon_nifti_path)
            nib.save(nib.Nifti1Image(to_numpy(gt[idx, 0]), np.eye(4)), gt_nifti_path)

            # Log them as artifacts
            recon_artifact = wandb.Artifact(f"recon_iter{global_step}_b{idx}", type="volume")
            recon_artifact.add_file(recon_nifti_path)
            wandb.log_artifact(recon_artifact)

            gt_artifact = wandb.Artifact(f"gt_iter{global_step}_b{idx}", type="volume")
            gt_artifact.add_file(gt_nifti_path)
            wandb.log_artifact(gt_artifact)