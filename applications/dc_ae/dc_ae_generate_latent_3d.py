import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import MISSING
from torch.utils.data import Dataset
from tqdm import tqdm

from dc_gen.ae_model_zoo import REGISTERED_DCAE_MODEL, REGISTERED_SD_VAE_MODEL, AutoencoderKL, create_dc_ae_model_cfg
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
    sync_tensor,
)
from dc_gen.c2icore.autoencoder import Autoencoder, AutoencoderConfig
from dc_gen.models.utils.network import get_dtype_from_str


@dataclass(frozen=True)
class VolumeSample:
    sample_id: str
    source_type: str
    file_path: Optional[str] = None
    group: Optional[str] = None
    key: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class GenerateLatent3DConfig:
    latent_root_path: str = MISSING
    results_path: Optional[str] = None

    hdf_path: Optional[str] = None
    nifti_dir: Optional[str] = None
    csv_metadata: Optional[str] = None
    path_column: str = "path"
    input_root_path: Optional[str] = None
    group_names: list[str] = field(default_factory=lambda: ["Vol_full"])
    n_slices: Optional[int] = None
    fraction: float = 1.0
    seed: int = 42

    clip_input_range: Optional[list[float]] = field(default_factory=lambda: [-1000.0, 1000.0])
    normalize_mode: Optional[str] = "sample"
    normalize_input_range: Optional[list[float]] = None
    normalize_output_range: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    resize_hw: Optional[list[int]] = field(default_factory=lambda: [128, 128])
    resize_depth: Optional[int] = None

    model_name: str = MISSING
    pretrained_path: Optional[str] = None
    dtype: str = "fp32"
    save_dtype: str = "fp32"
    scaling_factor: Optional[float] = None
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)

    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    save_ext: str = ".npy"

    task_id: int = 0
    num_samples_per_task: Optional[int] = None
    resume: bool = True


class VolumeLatentDataset(Dataset):
    def __init__(self, cfg: GenerateLatent3DConfig):
        super().__init__()
        self.cfg = cfg
        self.samples = self._build_samples()
        self._h5_file = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5_file"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        volume = self._load_volume(sample)
        volume = preprocess_volume(volume, self.cfg)
        return {
            "image": volume,
            "sample_id": sample.sample_id,
        }

    def _build_samples(self) -> list[VolumeSample]:
        source_count = sum(value is not None for value in (self.cfg.hdf_path, self.cfg.nifti_dir, self.cfg.csv_metadata))
        if source_count != 1:
            raise ValueError("Exactly one of hdf_path, nifti_dir, or csv_metadata must be provided")

        if self.cfg.hdf_path is not None:
            return self._build_hdf_samples()
        return self._build_nifti_samples()

    def _build_hdf_samples(self) -> list[VolumeSample]:
        samples: list[VolumeSample] = []
        with h5py.File(self.cfg.hdf_path, "r") as handle:
            for group in self.cfg.group_names:
                if group not in handle:
                    raise ValueError(f"Group '{group}' not found in HDF5 file {self.cfg.hdf_path}")
                for key in sorted(handle[group].keys()):
                    shape = tuple(handle[group][key].shape)
                    if len(shape) == 3:
                        total_slices = shape[0]
                    elif len(shape) == 4:
                        total_slices = shape[1]
                    else:
                        raise ValueError(f"Unsupported HDF5 sample shape {shape} for {group}/{key}")
                    start, end = compute_slice_range(total_slices, self.cfg.n_slices)
                    slice_suffix = build_slice_suffix(start, end, total_slices)
                    sample_id = str(Path(group) / f"{sanitize_name(key)}{slice_suffix}")
                    samples.append(
                        VolumeSample(
                            sample_id=sample_id,
                            source_type="hdf5",
                            group=group,
                            key=key,
                            start=start,
                            end=end,
                        )
                    )
        return maybe_subsample(samples, self.cfg.fraction, self.cfg.seed)

    def _build_nifti_samples(self) -> list[VolumeSample]:
        if self.cfg.csv_metadata is not None:
            file_paths = []
            with open(self.cfg.csv_metadata, newline="") as handle:
                reader = csv.DictReader(handle)
                if self.cfg.path_column not in reader.fieldnames:
                    raise ValueError(
                        f"path_column '{self.cfg.path_column}' not found in CSV columns {reader.fieldnames}"
                    )
                for row in reader:
                    file_paths.append(row[self.cfg.path_column])
        else:
            file_paths = [str(path) for path in Path(self.cfg.nifti_dir).expanduser().rglob("*.nii*")]

        file_paths = sorted(file_paths)
        if not file_paths:
            raise ValueError("No NIfTI files found")

        file_paths = maybe_subsample(file_paths, self.cfg.fraction, self.cfg.seed)

        input_root = resolve_input_root(self.cfg)
        samples: list[VolumeSample] = []
        for file_path in file_paths:
            img = nib.load(file_path)
            if len(img.shape) < 3:
                raise ValueError(f"Expected 3D NIfTI volume, got shape {img.shape} for {file_path}")
            total_slices = int(img.shape[2])
            start, end = compute_slice_range(total_slices, self.cfg.n_slices)
            slice_suffix = build_slice_suffix(start, end, total_slices)
            sample_id = build_nifti_sample_id(file_path, input_root, slice_suffix)
            samples.append(
                VolumeSample(
                    sample_id=sample_id,
                    source_type="nifti",
                    file_path=file_path,
                    start=start,
                    end=end,
                )
            )
        return samples

    def _get_h5_file(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.cfg.hdf_path, "r")
        return self._h5_file

    def _load_volume(self, sample: VolumeSample) -> torch.Tensor:
        if sample.source_type == "hdf5":
            handle = self._get_h5_file()
            data = handle[sample.group][sample.key]
            if data.ndim == 3:
                volume = np.asarray(data[sample.start : sample.end]).copy()[None, ...]
            elif data.ndim == 4:
                volume = np.asarray(data[:, sample.start : sample.end]).copy()
            else:
                raise ValueError(f"Unsupported HDF5 sample rank {data.ndim} for {sample.group}/{sample.key}")
        elif sample.source_type == "nifti":
            img = nib.load(sample.file_path)
            volume = np.asarray(img.dataobj[..., sample.start : sample.end]).copy()
            volume = np.transpose(volume, (2, 0, 1))[None, ...]
        else:
            raise ValueError(f"Unsupported source type {sample.source_type}")
        return torch.from_numpy(volume).float()


def maybe_subsample(items: list, fraction: float, seed: int) -> list:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction >= 1:
        return items
    rng = np.random.default_rng(seed)
    count = max(1, int(len(items) * fraction))
    indices = np.sort(rng.choice(len(items), size=count, replace=False))
    return [items[index] for index in indices]


def sanitize_name(name: str) -> str:
    return name.replace("\\", "/").strip("/").replace("/", "__")


def strip_known_suffixes(path: Path) -> Path:
    if path.name.endswith(".nii.gz"):
        return path.with_name(path.name[: -len(".nii.gz")])
    if path.suffix:
        return path.with_suffix("")
    return path


def build_slice_suffix(start: int, end: int, total_slices: int) -> str:
    if start == 0 and end == total_slices:
        return ""
    return f"__z{start:04d}-{end:04d}"


def resolve_input_root(cfg: GenerateLatent3DConfig) -> Optional[Path]:
    if cfg.input_root_path is not None:
        return Path(cfg.input_root_path).expanduser().resolve()
    if cfg.nifti_dir is not None:
        return Path(cfg.nifti_dir).expanduser().resolve()
    return None


def build_nifti_sample_id(file_path: str, input_root: Optional[Path], slice_suffix: str) -> str:
    path = Path(file_path).expanduser().resolve()
    if input_root is not None:
        try:
            relative = path.relative_to(input_root)
        except ValueError:
            relative = Path("_abs") / Path(*path.parts[1:])
    else:
        relative = Path("_abs") / Path(*path.parts[1:])
    relative = strip_known_suffixes(relative)
    return str(relative.parent / f"{relative.name}{slice_suffix}")


def compute_slice_range(total_slices: int, n_slices: Optional[int]) -> tuple[int, int]:
    if n_slices is None or n_slices >= total_slices:
        return 0, total_slices
    start = (total_slices - n_slices) // 2
    return start, start + n_slices


def preprocess_volume(volume: torch.Tensor, cfg: GenerateLatent3DConfig) -> torch.Tensor:
    if cfg.clip_input_range is not None:
        if len(cfg.clip_input_range) != 2:
            raise ValueError("clip_input_range must contain exactly two values")
        volume = volume.clamp(min=cfg.clip_input_range[0], max=cfg.clip_input_range[1])

    if cfg.normalize_mode is not None:
        if len(cfg.normalize_output_range) != 2:
            raise ValueError("normalize_output_range must contain exactly two values")
        output_min, output_max = cfg.normalize_output_range
        if cfg.normalize_mode == "sample":
            input_min = volume.amin()
            input_max = volume.amax()
        elif cfg.normalize_mode == "fixed":
            if cfg.normalize_input_range is None or len(cfg.normalize_input_range) != 2:
                raise ValueError("normalize_input_range must contain exactly two values when normalize_mode='fixed'")
            input_min, input_max = cfg.normalize_input_range
        else:
            raise ValueError(f"normalize_mode {cfg.normalize_mode} is not supported")

        input_min = float(input_min)
        input_max = float(input_max)
        if input_max > input_min:
            volume = (volume - input_min) / (input_max - input_min)
            volume = volume * (output_max - output_min) + output_min
        else:
            volume = torch.full_like(volume, fill_value=(output_min + output_max) * 0.5)

    target_depth = cfg.resize_depth
    target_hw = cfg.resize_hw
    if target_depth is not None or target_hw is not None:
        if target_hw is None:
            target_hw = [int(volume.shape[-2]), int(volume.shape[-1])]
        if len(target_hw) != 2:
            raise ValueError("resize_hw must contain exactly two values")
        if target_depth is None:
            target_depth = int(volume.shape[-3])
        volume = F.interpolate(
            volume.unsqueeze(0),
            size=(int(target_depth), int(target_hw[0]), int(target_hw[1])),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

    return volume


def sample_id_to_latent_path(sample_id: str, latent_root_path: str, save_ext: str) -> str:
    save_ext = save_ext if save_ext.startswith(".") else f".{save_ext}"
    return str(Path(latent_root_path) / Path(f"{sample_id}{save_ext}"))


def load_saved_latent(latent_path: str) -> torch.Tensor:
    path = Path(latent_path)
    if path.suffix == ".npy":
        tensor = torch.from_numpy(np.load(path))
    elif path.suffix in {".pt", ".pth"}:
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            for key in ("latent", "image", "images"):
                if key in tensor:
                    tensor = tensor[key]
                    break
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
    else:
        raise ValueError(f"Unsupported latent extension {path.suffix}")
    if tensor.ndim != 4:
        raise ValueError(f"Expected latent tensor shaped [C, D, H, W], got {tuple(tensor.shape)}")
    return tensor


class LatentEncoderAdapter:
    def __init__(self, model: torch.nn.Module, scaling_factor: Optional[float] = None, apply_external_scaling: bool = False):
        self.model = model
        self.scaling_factor = 1.0 if scaling_factor is None else scaling_factor
        self.apply_external_scaling = apply_external_scaling

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def eval(self):
        self.model = self.model.eval()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.model.encode(x)
        if self.apply_external_scaling:
            latent = latent * self.scaling_factor
        return latent

    @property
    def spatial_compression_ratio(self):
        return getattr(self.model, "spatial_compression_ratio", None)


def build_model(cfg: GenerateLatent3DConfig) -> LatentEncoderAdapter:
    if cfg.model_name == "autoencoder":
        if cfg.pretrained_path is not None:
            raise ValueError("pretrained_path is only supported with a concrete model_name, not model_name=autoencoder")
        return LatentEncoderAdapter(Autoencoder(cfg.autoencoder), apply_external_scaling=False)

    if cfg.model_name in REGISTERED_DCAE_MODEL:
        model_cfg = create_dc_ae_model_cfg(cfg.model_name, cfg.pretrained_path)
        model = DCAE(model_cfg)
        scaling_factor = cfg.scaling_factor if cfg.scaling_factor is not None else model_cfg.scaling_factor
        return LatentEncoderAdapter(model, scaling_factor=scaling_factor, apply_external_scaling=True)

    if cfg.model_name in REGISTERED_SD_VAE_MODEL:
        model = REGISTERED_SD_VAE_MODEL[cfg.model_name][0](cfg.model_name, REGISTERED_SD_VAE_MODEL[cfg.model_name][1])
        scaling_factor = cfg.scaling_factor if cfg.scaling_factor is not None else getattr(model.cfg, "scaling_factor", None)
        return LatentEncoderAdapter(model, scaling_factor=scaling_factor, apply_external_scaling=True)

    if cfg.model_name in [
        "stabilityai/sd-vae-ft-ema",
        "stabilityai/sdxl-vae",
        "flux-vae",
        "sd3-vae",
        "asymmetric-autoencoder-kl-x-1-5",
        "asymmetric-autoencoder-kl-x-2",
    ]:
        model = AutoencoderKL(cfg.model_name)
        scaling_factor = cfg.scaling_factor
        if scaling_factor is None:
            scaling_factor = model.model.config.scaling_factor
        return LatentEncoderAdapter(model, scaling_factor=scaling_factor, apply_external_scaling=True)

    raise ValueError(f"{cfg.model_name} is not supported for generating 3D latents")


def save_latent(latent: torch.Tensor, latent_path: str, save_ext: str, save_dtype: torch.dtype) -> None:
    os.makedirs(os.path.dirname(latent_path), exist_ok=True)
    if save_ext == ".npy":
        dtype = torch.float32 if save_dtype == torch.bfloat16 else save_dtype
        np.save(latent_path, latent.detach().to(dtype=dtype).cpu().numpy())
    elif save_ext in {".pt", ".pth"}:
        torch.save(latent.detach().to(dtype=save_dtype).cpu(), latent_path)
    else:
        raise ValueError(f"Unsupported save_ext {save_ext}")


def main():
    torch.set_grad_enabled(False)
    cfg = get_config(GenerateLatent3DConfig)

    save_ext = cfg.save_ext if cfg.save_ext.startswith(".") else f".{cfg.save_ext}"
    if save_ext not in {".npy", ".pt", ".pth"}:
        raise ValueError(f"Unsupported save_ext {save_ext}")

    if torch.cuda.is_available():
        dist_init()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(get_dist_local_rank())
        device = torch.device("cuda")
    else:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        device = torch.device("cpu")

    model_dtype = get_dtype_from_str(cfg.dtype)
    save_dtype = get_dtype_from_str(cfg.save_dtype)
    model = build_model(cfg).eval().to(device=device, dtype=model_dtype)

    dataset = VolumeLatentDataset(cfg)

    if cfg.num_samples_per_task is not None:
        num_tasks = (len(dataset) - 1) // cfg.num_samples_per_task + 1
        if is_master():
            print(f"num_tasks {num_tasks}")
        start = min(cfg.num_samples_per_task * cfg.task_id, len(dataset))
        end = min(cfg.num_samples_per_task * (cfg.task_id + 1), len(dataset))
        indices = list(range(start, end))
        dataset = torch.utils.data.Subset(dataset, indices)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        sampler=DistributedRangedSampler(dataset, num_replicas=get_dist_size(), rank=get_dist_rank(), shuffle=False),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    if is_master():
        os.makedirs(cfg.latent_root_path, exist_ok=cfg.resume or cfg.num_samples_per_task is not None)
    dist_barrier()

    latent_total_sum = 0.0
    latent_total_sum_squared = 0.0
    latent_total_cnt = 0

    for batch_idx, batch in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        disable=not is_master(),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        latent_paths = [sample_id_to_latent_path(sample_id, cfg.latent_root_path, save_ext) for sample_id in batch["sample_id"]]

        if cfg.resume:
            skip = True
            for latent_path in latent_paths:
                try:
                    load_saved_latent(latent_path)
                except Exception:
                    skip = False
                    break
            if skip:
                if is_master():
                    print(f"skip batch {batch_idx}")
                continue

        images = batch["image"].to(device=device, dtype=model_dtype, non_blocking=cfg.pin_memory)
        latents = model.encode(images)

        latent_total_sum += latents.float().sum().item()
        latent_total_sum_squared += latents.float().square().sum().item()
        latent_total_cnt += latents.numel()

        for latent, latent_path in zip(latents, latent_paths):
            save_latent(latent, latent_path, save_ext, save_dtype)

    if torch.cuda.is_available():
        latent_total_sum = sync_tensor(torch.tensor(latent_total_sum).cuda(), reduce="sum").cpu().numpy()
        latent_total_sum_squared = sync_tensor(torch.tensor(latent_total_sum_squared).cuda(), reduce="sum").cpu().numpy()
        latent_total_cnt = sync_tensor(torch.tensor(latent_total_cnt).cuda(), reduce="sum").cpu().numpy()

    if latent_total_cnt > 0:
        mean = latent_total_sum / latent_total_cnt
        rms = np.sqrt(latent_total_sum_squared / latent_total_cnt)
        variance = (latent_total_sum_squared - mean * latent_total_sum) / max(latent_total_cnt - 1, 1)
        std = np.sqrt(max(variance, 0))
        if is_master():
            print(f"mean: {mean}, rms: {rms}, std: {std}")

    dist_barrier()

    if cfg.results_path is not None and is_master():
        os.makedirs(cfg.results_path, exist_ok=True)
        with open(os.path.join(cfg.results_path, f"{cfg.num_samples_per_task}_{cfg.task_id}.txt"), "w") as handle:
            handle.write("complete!")


if __name__ == "__main__":
    main()
