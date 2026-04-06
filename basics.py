import torch
import numpy as np
import os
import wandb
import glob
from contextlib import nullcontext
from typing import Optional
import matplotlib.pyplot as plt

from dc_gen.models.utils.network import get_dtype_from_str
from flow.flow_utils import DTYPE_NAME_MAP

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


def resolve_autocast_dtype(cfg, device):
    requested_dtype = getattr(cfg.training, "autocast_dtype", "auto")

    if device.type == "cuda":
        if requested_dtype == "auto":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if requested_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            return torch.float16
    elif device.type == "mps":
        return torch.float16
    elif device.type == "cpu":
        return torch.bfloat16
    else:
        return None

    return getattr(torch, requested_dtype)


def get_autocast_ctx(cfg, device):
    if cfg.training.use_autocast:
        autocast_dtype = resolve_autocast_dtype(cfg, device)
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=True)
    return nullcontext()


def resolve_device(rank: int) -> torch.device:
    use_cuda = torch.cuda.is_available()
    return torch.device("mps" if torch.backends.mps.is_available() and not use_cuda else f"cuda:{rank}" if use_cuda else "cpu")

def move_batch(batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]: # todo: uses this for dc-ae training as well
    return {key: value.to(device=device, dtype=dtype, non_blocking=True) for key, value in batch.items()}

def should_run(step: int, every: Optional[int]) -> bool:
     return every is not None and step > 0 and step % every == 0

def torch_dtype(name: str) -> torch.dtype:
     return get_dtype_from_str(DTYPE_NAME_MAP[name])

def shutdown_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    rank0_print(f"\n[{os.getpid()}] Shutdown signal received. Finishing current batch...")



def get_git_info():
    """Get git commit hash, status, and diff."""
    git_info = {}
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_info["git_commit_hash"] = commit_hash

        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        git_info["git_status"] = status if status else "clean"

        diff = subprocess.check_output(["git", "diff", "HEAD"], text=True).strip()
        git_info["git_diff"] = diff if diff else "no changes"

        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        git_info["git_branch"] = branch
    except Exception as e:
        git_info["git_error"] = str(e)

    return git_info
