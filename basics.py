import torch
import numpy as np
import os
import wandb
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch

def to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()

def ensure_numpy(x):
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float32).numpy()
    return np.array(x)

def ensure_batch(x):
    if x.ndim == 2:
        return x[np.newaxis, ...]
    return x

def compute_map(gt, recon, map_fn=None):
    if map_fn:
        return map_fn(gt, recon)
    return gt - recon

def get_plot_range(data, symmetric=False):
    if symmetric:
        abs_max = max(np.max(np.abs(data)), 1e-8)
        return -abs_max, abs_max
    return data.min(), data.max()

def plot_batch(batch_data, titles=None, cmaps=None, vmin_vmax=None):
    B = len(batch_data)
    cols = len(batch_data[0])
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(B, cols, figsize=(5*cols, 5*B), constrained_layout=True)
    if B == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(B):
        for j in range(cols):
            axes[i, j].imshow(batch_data[i][j], cmap=cmaps[j], **vmin_vmax[j])
            axes[i, j].axis('off')
            if titles:
                axes[i, j].set_title(titles[i][j])
    return fig, axes

def save_figure(fig, save_path, im_for_colorbar=None, label=None):
    if im_for_colorbar is not None:
        fig.colorbar(im_for_colorbar, ax=fig.axes, location='right', shrink=0.85, pad=0.02, label=label)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def batch_statistics(batch):
    batch_np = batch.detach().cpu().numpy() if isinstance(batch, torch.Tensor) else batch
    B = batch_np.shape[0]

    sample_stats = [(batch_np[i].min(), batch_np[i].max(), batch_np[i].mean(), batch_np[i].std()) for i in range(B)]
    for i, stats in enumerate(sample_stats):
        print(f"Sample {i}: min={stats[0]:.3f}, max={stats[1]:.3f}, mean={stats[2]:.3f}, std={stats[3]:.3f}")

    overall = (batch_np.min(), batch_np.max(), batch_np.mean(), batch_np.std())
    print(f"Batch overall: min={overall[0]:.3f}, max={overall[1]:.3f}, mean={overall[2]:.3f}, std={overall[3]:.3f}")
    return sample_stats, overall


def get_autocast_ctx(cfg, device):
    if cfg.training.use_autocast:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()
