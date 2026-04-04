import argparse
import os
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional

import nibabel as nib
import numpy as np
import torch
import wandb
from dc_gen.models.utils.network import get_dtype_from_str
from omegaconf import OmegaConf
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler

from basics import get_autocast_ctx, resolve_autocast_dtype
from checkpointing import load_checkpoint, save_checkpoint
from flow.dit import DiT, DiT_models
from flow.ema import create_ema_model, update_ema_warmup
from flow.inspection import ModelInspector, flatten_inspection_stats, format_inspection_summary
from flow.latent_dataset import LatentTensorDataset, infer_latent_shape
from flow.logging_utils import format_kv_block, format_step_log, tensor_stats_dict
from flow.rectified_flow import RectifiedFlowObjective
from flow.samplers import sample_velocity_model
from flow.timestep_schedules import SAMPLE_TIME_SCHEDULES, TRAIN_TIMESTEP_MODES
from multigpu import barrier, cleanup, init_distributed, is_main_process, rank0_print, wrap_ddp

VALID_TRAIN_DTYPES = {"float32", "float16", "bfloat16"}
VALID_AUTOCAST_DTYPES = {"auto", "float16", "bfloat16"}
DTYPE_NAME_MAP = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


@dataclass
class DatasetConfig:
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    extensions: list[str] = field(default_factory=lambda: [".npy", ".pt", ".pth"])
    recursive: bool = True
    fraction: float = 1.0
    val_fraction: float = 1.0
    seed: int = 42


@dataclass
class PathsConfig:
    hdf_path: str = "/data/ct_rate_train_batch_0_v13.hdf"
    checkpoint_root_dir: str = "/cluster/home/kostfab1/DC-GEN/checkpoints/checkpoints_v03"
    checkpoint_dir: Optional[str] = None
    save_root_dir: str = "/cluster/home/kostfab1/DC-Gen/flow"
    save_dir: Optional[str] = None
    nifti_dir: str = "/cluster/projects/ac3t/data/ac3t_ct_rate/processed/train/"
    nifti_val_dir: str = "/cluster/projects/ac3t/data/ac3t_ct_rate/processed/valid/"
    latent_train_dir: str = "/cluster/projects/2025_stmd_VT_diff/latents/latents_v04_shallow_phase1_train"
    latent_val_dir: str = "/cluster/projects/2025_stmd_VT_diff/latents/latents_v04_shallow_phase1_valid"


@dataclass
class ObjectiveConfig:
    eps: float = 1e-5
    mode: str = "uniform"
    mu: float = 0.0
    sigma: float = 1.0
    beta_a: float = 1.0
    beta_b: float = 1.0


@dataclass
class TrainingConfig:
    num_epochs: int = 350
    batch_size: int = 512
    shuffle_data: bool = True
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: Optional[int] = None
    dtype: str = "float32"
    autocast_dtype: str = "auto"
    use_autocast: bool = True
    compile: bool = False
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    grad_clip_norm: Optional[float] = 1.0
    checkpoint_every: int = 1000
    max_checkpoints: int = 4
    resume_from_checkpoint: bool = False
    resume_from_checkpoint_path: Optional[str] = None
    log_every: int = 10
    ema_decay: Optional[float] = 0.9999
    ema_warmup_steps: int = 2000


@dataclass
class ModelConfig:
    variant: str = "DiT-L/1"
    input_size: Optional[list[int]] = None
    patch_size: int = 1
    in_channels: Optional[int] = None
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_classes: int = 1
    class_dropout_prob: float = 0.0
    learn_sigma: bool = False


@dataclass
class SamplingConfig:
    enabled: bool = True
    sample_steps: int = 50
    solver: str = "euler"
    time_schedule: str = "linear"
    time_schedule_power: float = 2.0
    num_samples: int = 32
    batch_size: int = 4
    sample_every: Optional[int] = 1000
    output_dir: Optional[str] = None
    save_dtype: str = "float32"
    t_start: float = 0.0
    t_end: float = 1.0
    ae_model_name: str = "dc-ae-f32c32-in-1.0_3d-shallow"
    ae_pretrained_path: Optional[str] = "/cluster/projects/2025_stmd_VT_diff/DC-GEN_backup/checkpoints_v04_shallow_phase1/checkpoint_iter141000.pt"
    decode_samples: bool = True


@dataclass
class LoggingConfig:
    wandb: bool = True
    wandb_project: str = "flow"
    validate_every: Optional[int] = 5000
    max_val_batches: Optional[int] = 10
    save_samples: bool = True


@dataclass
class InspectionConfig:
    enabled: bool = False
    inspect_every: Optional[int] = 100
    module_names: list[str] = field(default_factory=list)
    capture_weights: bool = True
    capture_gradients: bool = True
    capture_activations: bool = True
    print_summary: bool = True
    log_to_wandb: bool = True


@dataclass
class ExperimentConfig:
    name: str = "dit_3d_rectified_flow"


@dataclass
class TrainDiT3DConfig:
    dims: str = "3d"
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inspection: InspectionConfig = field(default_factory=InspectionConfig)


def resolve_device(rank: int) -> torch.device:
    use_cuda = torch.cuda.is_available()
    return torch.device("mps" if torch.backends.mps.is_available() and not use_cuda else f"cuda:{rank}" if use_cuda else "cpu")


def torch_dtype(name: str) -> torch.dtype:
    return get_dtype_from_str(DTYPE_NAME_MAP[name])


def should_run(step: int, every: Optional[int]) -> bool:
    return every is not None and step > 0 and step % every == 0


def move_batch(batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=dtype, non_blocking=True) for key, value in batch.items()}


def _load_ae_decoder(cfg, device):
    from dc_gen.ae_model_zoo import create_dc_ae_model_cfg
    from dc_gen.aecore.models.dc_ae import DCAE
    model_cfg = create_dc_ae_model_cfg(cfg.sampling.ae_model_name, cfg.sampling.ae_pretrained_path)
    return DCAE(model_cfg).eval().to(device=device, dtype=torch.float32)


def _save_decoded_samples(latents, ae, output_dir, global_step, device):
    for i in range(latents.shape[0]):
        latent = latents[i:i+1].to(device=device, dtype=torch.float32)
        vol = ae.decode(latent).squeeze(0).squeeze(0).detach().cpu().numpy()
        nib.save(nib.Nifti1Image(vol, np.eye(4)), os.path.join(output_dir, f"sample_step{global_step:07d}_{i:03d}.nii.gz"))
        c = vol.shape[0] // 2
        slc = np.clip(vol[c], -1000, 1000)
        slc = (slc + 1000) / 2000
        Image.fromarray((slc * 255).astype(np.uint8)).save(os.path.join(output_dir, f"sample_step{global_step:07d}_{i:03d}_center.png"))


def _save_reference_latents(val_loader, ae, output_dir, global_step, device, max_items=8):
    batch = next(iter(val_loader))
    latents = batch["image"][:max_items]
    for i in range(latents.shape[0]):
        latent = latents[i:i+1].to(device=device, dtype=torch.float32)
        vol = ae.decode(latent).squeeze(0).squeeze(0).detach().cpu().numpy()
        nib.save(nib.Nifti1Image(vol, np.eye(4)), os.path.join(output_dir, f"ref_step{global_step:07d}_{i:03d}.nii.gz"))
        c = vol.shape[0] // 2
        slc = np.clip(vol[c], -1000, 1000)
        slc = (slc + 1000) / 2000
        Image.fromarray((slc * 255).astype(np.uint8)).save(os.path.join(output_dir, f"ref_step{global_step:07d}_{i:03d}_center.png"))
