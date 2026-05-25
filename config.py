import os
from dataclasses import dataclass, field
from typing import List, Optional, Any
import torch
from omegaconf import OmegaConf

experiment_dirs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "experiments")

@dataclass
class DatasetConfig:
    name: str = "CTVolume"
    group_names: List[str] = field(default_factory=lambda: ["Vol_full"])
    volume: Optional[str] = None


@dataclass
class PathsConfig:
    hdf_path: str = "/data/ct_rate_train_batch_0_v13.hdf"
    checkpoint_dir: str = ""
    save_dir: str = ""
    nifti_dir: str = "/cluster/projects/2025_stmd_VT_diff/processed_256/train"
    nifti_val_dir: str = "/cluster/projects/2025_stmd_VT_diff/processed_256/valid/"


@dataclass
class PipelineConfig:
    resize_hw: List[int] = field(default_factory=lambda: [256, 256])
    n_slices: Optional[int] = 32
    random_slices: Optional[bool] = True
    resize_depth: Optional[int] = None
    clip_input_range: Optional[List[float]] = field(default_factory=lambda: [-1000.0, 1000.0])
    normalize_mode: str = "fixed"
    normalize_output_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    normalize_input_range: Optional[List[float]] = None


@dataclass
class ExperimentConfig:
    name: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ObjectiveConfig:
    loss_fn: str = "l1"
    perceptual_weight: float = 0.25
    detail_weight: float = 0.0
    gan_weight: float = 0.0
    ssim_weight: float = 0.0
    grad_weight: float = 0.0
    gan_loss_type: str = "hinge"
    gan_patch_size: List[int] = field(default_factory=lambda: [32, 32, 16])
    gan_ndf: int = 64
    gan_discriminator_steps: int = 1  # D updates per G update


@dataclass
class TrainingConfig:
    num_epochs: int = 350
    max_steps: Optional[int] = None
    seed: Optional[int] = None
    batch_size: int = 4
    shuffle_data: bool = True
    num_workers: int = 32
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: Optional[int] = None
    device: str = ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dtype: str = "float32"
    autocast_dtype: str = "auto"
    resume_from_checkpoint: bool = True
    load_optimizer_state: bool = True
    checkpoint_every: int = 1000
    max_checkpoints: int = 4
    use_autocast: bool = True
    trainable_ae_params: Optional[List[List[str]]] = None


@dataclass
class HParamsConfig:
    learning_rate: float = 2e-5
    discriminator_learning_rate: Optional[float] = None
    weight_decay: float = 1e-1


@dataclass
class ModelConfig:
    name: str = "dc-ae-f32c32-in-1.0_3d-shallow"
    compile: bool = True


@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "ae_v1"
    wandb_group: Optional[str] = None
    wandb_resume_run: bool = True
    save_volumes: bool = True
    viz_every: int = 500
    validate_every: int = 200000
    val_max_batches: Optional[int] = None
    print_model_arch: bool = False


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
    model_variant: Optional[Any] = None


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
        if (cfg.pipeline.normalize_input_range[0] >= cfg.pipeline.normalize_input_range[1]):
            raise ValueError("pipeline.normalize_input_range must be strictly increasing")

    if (cfg.pipeline.normalize_mode == "fixed" and cfg.pipeline.normalize_input_range is None):
        raise ValueError("pipeline.normalize_input_range must be set when normalize_mode='fixed'")

    if cfg.training.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("training.dtype must be one of: 'float32', 'float16', 'bfloat16'")

    if cfg.training.autocast_dtype not in {"auto", "float16", "bfloat16"}:
        raise ValueError("training.autocast_dtype must be one of: 'auto', 'float16', 'bfloat16'")

    if cfg.dims == "2d":
        cfg.pipeline.n_slices = None
        cfg.pipeline.resize_depth = None
        if cfg.model_variant is None and "_3d" in cfg.model.name:
            raise ValueError(f"2D config cannot use 3D model name: {cfg.model.name}")
    else:
        if cfg.model_variant is None and "_3d" not in cfg.model.name:
            raise ValueError(f"3D config should use a 3D model variant, got: {cfg.model.name}")
        if cfg.pipeline.n_slices is None and cfg.pipeline.resize_depth is None:
            raise ValueError("3D config must set pipeline.n_slices or pipeline.resize_depth")

    if not cfg.paths.save_dir or not cfg.paths.checkpoint_dir:
        raise ValueError(f"No paths configured for model {cfg.model.name!r}. "
            f"Create an experiment YAML in configs/experiments/ or set paths explicitly."
        )

    if cfg.experiment.name is None:
        cfg.experiment.name = cfg.model.name

    return cfg


def load_config(yaml_path: str = None, experiment: str = None, model: str = None):
    import yaml
    _registry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "models", "registry.yaml")
    cfg = OmegaConf.structured(Config)
    if model:
        registry = yaml.safe_load(open(_registry_path))
        if model not in registry:
            raise ValueError(f"Model '{model}' not in registry. Available:\n" + "\n".join(
                f"  {k:40s} lat={v['latent_channels']} enc_depth={v['enc_depth']} width={v['enc_width']}"
                for k, v in registry.items()
            ))
        OmegaConf.set_struct(cfg, False)
        cfg.model_variant = registry[model]
        OmegaConf.set_struct(cfg, True)
    if experiment:
        exp_path = os.path.join(experiment_dirs, f"{experiment}.yaml")
        if not os.path.isfile(exp_path):
            available = [f.removesuffix(".yaml") for f in os.listdir(experiment_dirs) if f.endswith(".yaml")]
            raise FileNotFoundError(f"Experiment '{experiment}' not found at {exp_path}. Available: {available}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_path))
    if yaml_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(yaml_path))
    cfg = OmegaConf.to_object(cfg)
    return validate_and_finalize_config(cfg)


def list_models():
    import yaml
    _registry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "models", "registry.yaml")
    registry = yaml.safe_load(open(_registry_path))
    print("\nAvailable models:")
    for name, v in registry.items():
        print(f"  {name:40s} lat={v['latent_channels']} enc_depth={v['enc_depth']} width={v['enc_width']}")
