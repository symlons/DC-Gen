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
    hdf_path: str = "/data/ct_rate_train_batch_0_v13.hdf"
    checkpoint_root_dir: str = "/cluster/home/kostfab1/DC-GEN/checkpoints/checkpoints_v03"
    checkpoint_dir: Optional[str] = None
    save_root_dir: str = "/cluster/home/kostfab1/DC-Gen/dc_ae_3d_v03"
    save_dir: Optional[str] = None
    nifti_dir: str = "/cluster/projects/ac3t/data/ac3t_ct_rate/processed/train/"
    nifti_val_dir: str = "/cluster/projects/ac3t/data/ac3t_ct_rate/processed/valid/"
    latent_train_dir: str = "/cluster/home/kostfab1/DC-Gen/dc_ae_3d_v03/latents/train"
    latent_val_dir: str = "/cluster/home/kostfab1/DC-Gen/dc_ae_3d_v03/latents/valid"


@dataclass
class ObjectiveConfig:
    eps: float = 1e-5
    mode: str = "uniform"
    mu: float = 0.0
    sigma: float = 1.0
    beta_a: float = 1.0
    beta_b: float = 1.0
    latent_mean: float = 0.0
    latent_std: float = 1.0


@dataclass
class TrainingConfig:
    num_epochs: int = 350
    batch_size: int = 2
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
    variant: str = "custom"
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
    ae_model_name: str = "dc-ae-f32c32-in-1.0_3d-shallow"
    ae_checkpoint_dir: Optional[str] = None


@dataclass
class SamplingConfig:
    enabled: bool = True
    sample_steps: int = 50
    solver: str = "euler"
    time_schedule: str = "linear"
    time_schedule_power: float = 2.0
    num_samples: int = 8
    batch_size: int = 4
    sample_every: Optional[int] = 1000
    output_dir: Optional[str] = None
    save_dtype: str = "float32"
    t_start: float = 0.0
    t_end: float = 1.0


@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "ct_dit_3d_rectified_flow"
    validate_every: Optional[int] = 250
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
