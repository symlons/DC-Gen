import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from queue import Queue
from threading import Thread

import numpy as np
import torch
import torch.multiprocessing as mp
from monai.transforms import Resize
from omegaconf import MISSING
from torch.utils.data import Dataset
from tqdm import tqdm

from dc_gen.ae_model_zoo import create_dc_ae_model_cfg
from dc_gen.aecore.models.dc_ae import DCAE
from dc_gen.apps.data_provider.sampler import DistributedRangedSampler
from dc_gen.apps.utils.config import get_config
from dc_gen.apps.utils.dist import (
    dist_barrier,
    dist_init,
    get_dist_local_rank,
    get_dist_rank,
    get_dist_size,
    is_master,
)

from data import CTVolumeDataset


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

    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 8

    dtype: str = "bf16"
    save_dtype: str = "fp32"
    save_ext: str = ".npy"
    resume: bool = True
    save_queue_size: int = 64


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

        spatial_size = [cfg.n_slices, cfg.resize_hw[0], cfg.resize_hw[1]]
        self.resize = Resize(spatial_size=spatial_size, mode="trilinear")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        vol = self.dataset[idx]
        if vol is None:
            return None

        vol = preprocess_volume(vol, self.cfg)
        vol = self.resize(vol)

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

    return volume


class LatentEncoderAdapter:
    def __init__(self, model, scaling_factor):
        self.model = model
        self.scaling_factor = 1.0 if scaling_factor is None else scaling_factor

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def eval(self):
        self.model.eval()
        return self

    def encode(self, x):
        latent = self.model.encoder(x)
        return latent * self.scaling_factor


def build_model(cfg):
    variant = None
    registry_path = Path(__file__).resolve().parents[2] / "configs" / "models" / "registry.yaml"
    if registry_path.exists():
        with open(registry_path) as f:
            variant = yaml.safe_load(f).get(cfg.model_name)
    model_cfg = create_dc_ae_model_cfg(cfg.model_name, cfg.pretrained_path, variant=variant)
    model = DCAE(model_cfg)

    scaling = model_cfg.scaling_factor
    if scaling is None:
        scaling = getattr(model, "scaling_factor", None)
    if scaling is None:
        scaling = 1.0

    adapter = LatentEncoderAdapter(model, scaling)
    try:
        adapter.model = torch.compile(adapter.model)
    except Exception:
        pass
    return adapter


def save_latent_worker(save_queue, latent_root_path):
    while True:
        item = save_queue.get()
        if item is None:
            break
        lat, sid = item
        path = Path(latent_root_path) / f"{sid}.npy"
        os.makedirs(path.parent, exist_ok=True)
        np.save(path, lat, allow_pickle=False)


def run_worker(rank, world_size, cfg):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    if torch.cuda.is_available():
        dist_init()
        torch.cuda.set_device(rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).eval().to(device)

    dataset = VolumeLatentDataset(cfg)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=DistributedRangedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        ),
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_skip_none,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor,
    )

    if rank == 0:
        os.makedirs(cfg.latent_root_path, exist_ok=True)

    if world_size > 1:
        dist_barrier()

    save_queue = Queue(maxsize=cfg.save_queue_size)
    save_thread = Thread(target=save_latent_worker, args=(save_queue, cfg.latent_root_path), daemon=False)
    save_thread.start()

    compute_dtype = torch.bfloat16 if (cfg.dtype == "bf16" and device.type == "cuda") else torch.float32
    model.to(dtype=compute_dtype)

    for batch in tqdm(loader, disable=(rank != 0)):
        if batch is None:
            continue

        if cfg.resume:
            keep = []
            for i, sid in enumerate(batch["sample_id"]):
                path = Path(cfg.latent_root_path) / f"{sid}.npy"
                if not path.exists():
                    keep.append(i)
            if not keep:
                continue
            batch["image"] = batch["image"][keep]
            batch["sample_id"] = [batch["sample_id"][i] for i in keep]

        images = batch["image"].to(device, dtype=compute_dtype, non_blocking=True)

        with torch.inference_mode():
            latents = model.encode(images)

        latents = latents.float().cpu()
        for lat, sid in zip(latents, batch["sample_id"]):
            save_queue.put((lat.numpy(), sid))

    save_queue.put(None)
    save_thread.join()

    if world_size > 1:
        dist_barrier()


def main():
    cfg = get_config(GenerateLatent3DConfig)

    world_size = torch.cuda.device_count()

    if world_size > 1:
        mp.set_start_method("spawn", force=True)
        mp.spawn(run_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        run_worker(0, 1, cfg)


if __name__ == "__main__":
    main()
