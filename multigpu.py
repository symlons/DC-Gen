import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import functools
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist

from dc_gen.ae_model_zoo import create_dc_ae_model_cfg
from dc_gen.aecore.models.dc_ae import DCAE
from dc_gen.apps.utils.config import get_config

from data import CTVolumeDataset


def init_distributed(rank: int, world_size: int):
    if not dist.is_available():
        return
    if not dist.is_initialized():
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        if "MASTER_PORT" not in os.environ: os.environ["MASTER_PORT"] = "29500"
        rank0_print("[DDP init] "
            f"world_size={world_size}, "
            f"master_addr={os.environ['MASTER_ADDR']}, "
            f"master_port={os.environ['MASTER_PORT']}"
        )
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{rank}"))


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return get_rank() == 0


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def wrap_ddp(model):
    if get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[get_rank()],
            output_device=get_rank(),
            find_unused_parameters=False,
        )
    return model


@contextmanager
def main_process_first():
    if get_world_size() > 1:
        barrier()
    yield
    if get_world_size() > 1:
        barrier()


def aggregate_metrics(metrics_dict):
    """Aggregate metrics across all distributed processes."""
    if get_world_size() <= 1 or not dist.is_initialized():
        return metrics_dict
    
    aggregated = {}
    for key, val in metrics_dict.items():
        if isinstance(val, (int, float)):
            tensor = torch.tensor(val, dtype=torch.float32, device=torch.cuda.current_device())
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            aggregated[key] = tensor.item()
        else:
            aggregated[key] = val
    return aggregated


@dataclass
class GenerateLatent3DConfig:
    latent_root_path: str = MISSING

    hdf_path: Optional[str] = None
    nifti_dir: Optional[str] = None
    csv_metadata: Optional[str] = None
    group_names: list[str] = field(default_factory=lambda: ["Vol_full"])
    n_slices: Optional[int] = None
    fraction: float = 1.0

    clip_input_range: list[float] = field(default_factory=lambda: [-1000.0, 1000.0])
    normalize_mode: str = "sample"
    normalize_input_range: Optional[list[float]] = None
    normalize_output_range: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    resize_hw: list[int] = field(default_factory=lambda: [256, 256])

    model_name: str = MISSING
    pretrained_path: Optional[str] = None

    batch_size: int = 4
    num_workers: int = 8
    pin_memory: bool = False

    dtype: str = "fp32"
    save_dtype: str = "fp32"
    save_ext: str = ".npy"
    resume: bool = True


class VolumeLatentDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.dataset = CTVolumeDataset(
            hdf_path=cfg.hdf_path,
            nifti_dir=cfg.nifti_dir,
            csv_metadata=cfg.csv_metadata,
            group_names=cfg.group_names,
            dims="3d",
            n_slices=cfg.n_slices,
            fraction=cfg.fraction,
            transform=None,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        vol = self.dataset[idx]
        if vol is None:
            return None

        vol = preprocess_volume(vol, self.cfg)

        entry = self.dataset.backend.index_map[idx]
        sid = entry[0].replace("/", "__")

        return {
            "image": vol,
            "sample_id": sid,
        }


def collate_fn_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    return {
        "image": torch.stack([b["image"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
    }


def preprocess_volume(volume, cfg):
    volume = volume.clamp(cfg.clip_input_range[0], cfg.clip_input_range[1])

    out_min, out_max = cfg.normalize_output_range

    if cfg.normalize_mode == "sample":
        mn = volume.amin()
        mx = volume.amax()
    elif cfg.normalize_mode == "fixed":
        if cfg.normalize_input_range is None:
            raise ValueError("normalize_input_range required for fixed mode")
        mn, mx = cfg.normalize_input_range
    else:
        raise ValueError("invalid normalize_mode")

    mn = float(mn)
    mx = float(mx)

    if mx > mn:
        volume = (volume - mn) / (mx - mn)
        volume = volume * (out_max - out_min) + out_min
    else:
        volume = torch.full_like(volume, (out_min + out_max) * 0.5)

    volume = F.interpolate(
        volume.unsqueeze(0),
        size=(cfg.n_slices, cfg.resize_hw[0], cfg.resize_hw[1]),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)

    return volume


class LatentEncoderAdapter(torch.nn.Module):
    def __init__(self, model, scaling_factor):
        super().__init__()
        self.model = model
        self.scaling_factor = 1.0 if scaling_factor is None else scaling_factor

    def forward(self, x):
        latent = self.model.encoder(x)
        return latent * self.scaling_factor


def build_model(cfg):
    model_cfg = create_dc_ae_model_cfg(cfg.model_name, cfg.pretrained_path)
    model = DCAE(model_cfg)

    scaling = model_cfg.scaling_factor
    if scaling is None:
        scaling = getattr(model, "scaling_factor", None)
    if scaling is None:
        scaling = 1.0

    return LatentEncoderAdapter(model, scaling)


def run_worker(rank, world_size, cfg, device):
    print(f"[RANK {rank}] using device {device}", flush=True)

    if device.type == "cuda":
        torch.cuda.set_device(rank)

    model = build_model(cfg).to(device)

    if device.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            print(f"[RANK {rank}] compile failed, continuing without", flush=True)

    model = model.eval()

    dataset = VolumeLatentDataset(cfg)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn_skip_none,
        persistent_workers=cfg.num_workers > 0,
    )

    if rank == 0:
        os.makedirs(cfg.latent_root_path, exist_ok=True)

    compute_dtype = torch.bfloat16 if (cfg.dtype == "bf16" and device.type == "cuda") else torch.float32

    sampler.set_epoch(0)

    for batch in tqdm(loader, disable=(rank != 0)):
        if batch is None:
            continue

        sample_ids = batch["sample_id"]

        paths = [
            Path(cfg.latent_root_path) / f"{sid}{cfg.save_ext}"
            for sid in sample_ids
        ]

        if cfg.resume:
            keep = [not p.exists() for p in paths]
            if not any(keep):
                continue
            images = batch["image"][keep]
            paths = [p for p, k in zip(paths, keep) if k]
        else:
            images = batch["image"]

        images = images.to(device, dtype=compute_dtype, non_blocking=cfg.pin_memory)

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                latents = model(images)

        for lat, path in zip(latents, paths):
            os.makedirs(path.parent, exist_ok=True)

            if cfg.save_dtype == "fp16":
                arr = lat.detach().half().cpu().numpy()
            else:
                arr = lat.detach().float().cpu().numpy()

            np.save(path, arr, allow_pickle=False)

        del images, latents
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_worker_entry(rank, world_size, cfg):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    run_worker(rank, world_size, cfg, device)
