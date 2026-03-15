
from omegaconf import OmegaConf

def get_default_config():
    cfg = OmegaConf.create({
        "paths": {
            # "hdf_path": "/mnt/SFS-iCxS1nYm/ct_rate_train_batch_0_v13.hdf",
            "hdf_path": "/data/ct_rate_train_batch_0_v13.hdf",
            "artifact_dir": "artifacts_3d_33",
            "checkpoint_dir": "/data/checkpoints"
        },
        "training": {
            "batch_size": 128,
            "num_epochs": 2000,
            "shuffle_data": True,
            "diff_save_every": 250,
            "checkpoint_every": 1000,
            "max_checkpoints": 5,
            "lr": 8e-5,
            "device": "cuda",
            "dtype": "bfloat16",
            "loss_fn": "l1",
            "perceptual_weight": 0.25,
            "resume_from_checkpoint": False,
            "checkpoint_path": "/mnt/SFS-iCxS1nYm/DC-Gen/3d_experiments/checkpoints/checkpoint_iter4000.pt"
        },
        "dataset": {
            "name": "CTVolume",
            "group_names": ["Vol_full"],
            "volume": True
        },
        "model": {"name": "dc-ae-f32c32-in-1.0_2d"},
        "pipeline": {"n_slices": 32, "resize_hw": [128, 128]},
        "wandb": {"enabled": False, "project": "ct_recon", "run_name": "dc_ae_experiment"},
        "logging": {"save_volumes": True, "save_training_curves": True}
    })
    return cfg

import torch
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf

@dataclass
class DatasetConfig:
    name: str = "CTVolume"
    group_names: List[str] = field(default_factory=lambda: ["Vol_full"])
    volume: Optional[str] = None

@dataclass
class PathsConfig:
    hdf_path: str = "/Users/sfkost/storage/ct_rate_train_batch_0_v13.hdf"
    checkpoint_dir: str = "/Users/sfkost/storage/fun"
    artifact_dir: str = "/Users/sfkost/storage/fun"

@dataclass
class PipelineConfig:
    resize_hw: List[int] = field(default_factory=lambda: [256, 256])
    n_slices: Optional[int] = None

@dataclass
class ObjectiveConfig:
    loss_fn: str = "l1"
    perceptual_weight: float = 0.25

@dataclass
class TrainingConfig:
    num_epochs: int = 5
    batch_size: int = 1
    shuffle_data: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    resume_from_checkpoint: bool = False
    checkpoint_every: int = 100
    max_checkpoints: int = 4
    use_autocast: bool = True

@dataclass
class HParamsConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2

@dataclass
class ModelConfig:
    name: str = "dc-ae-f32c32-in-1.0_2d"
    compile: bool = True

@dataclass
class LoggingConfig:
    wandb: bool = False
    save_volumes: bool = True
    viz_every: int = 100

@dataclass
class Config:
    dims: str = "2d"
    save_dir: str = "/Users/sfkost/storage/fun/"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    hparams: HParamsConfig = field(default_factory=HParamsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

def load_config(yaml_path: str = None):
    cfg = OmegaConf.structured(Config)
    if yaml_path:
        yaml_cfg = OmegaConf.load(yaml_path)
        cfg = OmegaConf.merge(cfg, yaml_cfg)
    return cfg

