import torch
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf


@dataclass
class DatasetConfig:
    name: str = "CTVolume"
    group_names: List[str] = field(default_factory=lambda: ["Vol_full"])
    volume: Optional[str] = None

# @dataclass
# class PathsConfig:
#     hdf_path: str = "/Users/sfkost/storage/ct_rate_train_batch_0_v13.hdf"
#     checkpoint_dir: str = "/Users/sfkost/storage/fun"
#     save_dir: str = "/Users/sfkost/storage/fun"

# @dataclass
# class PathsConfig:
#     hdf_path: str = "/mnt/Volume-eV4BofCN/ct_rate_train_batch_0_v13.hdf"
#     checkpoint_dir: str = "/mnt/checkpoints"
#     save_dir: str = "/mnt/Volume-eV4BofCN/artifacts_3d_33"

@dataclass
class PathsConfig:
    hdf_path: str = "/data/ct_rate_train_batch_0_v13.hdf"
    checkpoint_dir: str = "/data/checkpoints_3"
    save_dir: str = "/data/Volume-eV4BofCN/artifacts_3d_33_3"

@dataclass
class PipelineConfig:
    resize_hw: List[int] = field(default_factory=lambda: [128, 128])
    n_slices: Optional[int] = 32
    resize_depth: Optional[int] = None
    clip_input_range: Optional[List[float]] = field(default_factory=lambda: [-1000.0, 1000.0])
    normalize_mode: str = "sample"
    normalize_output_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    normalize_input_range: Optional[List[float]] = None

@dataclass
class ExperimentConfig:
    name: Optional[str] = None

@dataclass
class ObjectiveConfig:
    loss_fn: str = "l1"
    perceptual_weight: float = 0.25
    detail_weight: float = 0.1

@dataclass
class TrainingConfig:
    num_epochs: int = 350
    batch_size: int = 6
    shuffle_data: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype: str = "float32"
    autocast_dtype: str = "auto"
    resume_from_checkpoint: bool = False
    checkpoint_every: int = 1000
    max_checkpoints: int = 4
    use_autocast: bool = True

@dataclass
class HParamsConfig:
    learning_rate: float = 8e-5
    weight_decay: float = 1e-1

@dataclass
class ModelConfig:
    name: str = "dc-ae-f32c32-in-1.0_3d-depth-last"
    compile: bool = True

@dataclass
class LoggingConfig:
    wandb: bool = True
    save_volumes: bool = True
    viz_every: int = 500

@dataclass
class Config:
    dims: str = "3d"
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    hparams: HParamsConfig = field(default_factory=HParamsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def validate_and_finalize_config(cfg):
    if cfg.dims not in {"2d", "3d"}:
        raise ValueError(f"Unsupported dims={cfg.dims!r}, expected '2d' or '3d'")

    if len(cfg.pipeline.resize_hw) != 2 or any(dim is None or dim <= 0 for dim in cfg.pipeline.resize_hw):
        raise ValueError("pipeline.resize_hw must be a 2-element list of positive integers")

    if cfg.pipeline.normalize_mode not in {"sample", "fixed"}:
        raise ValueError("pipeline.normalize_mode must be one of: 'sample' or 'fixed'")

    if len(cfg.pipeline.normalize_output_range) != 2:
        raise ValueError("pipeline.normalize_output_range must contain exactly two values")

    if cfg.pipeline.clip_input_range is not None:
        if len(cfg.pipeline.clip_input_range) != 2:
            raise ValueError("pipeline.clip_input_range must contain exactly two values")
        if cfg.pipeline.clip_input_range[0] >= cfg.pipeline.clip_input_range[1]:
            raise ValueError("pipeline.clip_input_range must be strictly increasing")

    if cfg.pipeline.normalize_input_range is not None:
        if len(cfg.pipeline.normalize_input_range) != 2:
            raise ValueError("pipeline.normalize_input_range must contain exactly two values")
        if cfg.pipeline.normalize_input_range[0] >= cfg.pipeline.normalize_input_range[1]:
            raise ValueError("pipeline.normalize_input_range must be strictly increasing")

    if cfg.pipeline.normalize_mode == "fixed" and cfg.pipeline.normalize_input_range is None:
        raise ValueError("pipeline.normalize_input_range must be set when normalize_mode='fixed'")

    if cfg.training.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("training.dtype must be one of: 'float32', 'float16', 'bfloat16'")

    if cfg.training.autocast_dtype not in {"auto", "float16", "bfloat16"}:
        raise ValueError("training.autocast_dtype must be one of: 'auto', 'float16', 'bfloat16'")

    if cfg.dims == "2d":
        cfg.pipeline.n_slices = None
        cfg.pipeline.resize_depth = None
        if "_3d" in cfg.model.name:
            raise ValueError(f"2D config cannot use 3D model name: {cfg.model.name}")
    else:
        if "_3d" not in cfg.model.name:
            raise ValueError(f"3D config should use a 3D model variant, got: {cfg.model.name}")
        if cfg.pipeline.n_slices is None and cfg.pipeline.resize_depth is None:
            raise ValueError("3D config must set pipeline.n_slices or pipeline.resize_depth")

    if cfg.experiment.name is None:
        cfg.experiment.name = cfg.model.name

    return cfg

def load_config(yaml_path: str = None):
    cfg = OmegaConf.structured(Config)
    if yaml_path:
        yaml_cfg = OmegaConf.load(yaml_path)
        cfg = OmegaConf.merge(cfg, yaml_cfg)
    cfg = OmegaConf.to_object(cfg)
    return validate_and_finalize_config(cfg)
