schema Dataset:
    name: str
    group_names: [str]
    volume?: int

    check:
        len(group_names) > 0, "group_names must not be empty"
        volume > 0 if volume, "volume must be positive"

schema Paths:
    hdf_path: str
    checkpoint_dir: str
    save_dir: str

schema Pipeline:
    resize_hw: [int]
    n_slices?: int
    resize_depth?: int
    normalize_mode: "sample" | "fixed"
    normalize_output_range: [float]
    normalize_input_range?: [float]

    check:
        len(resize_hw) == 2, "resize_hw must be [H, W]"
        resize_hw[0] > 0 and resize_hw[1] > 0, "resize_hw dimensions must be positive"
        n_slices > 0 if n_slices, "n_slices must be positive"
        resize_depth > 0 if resize_depth, "resize_depth must be positive"
        len(normalize_output_range) == 2, "normalize_output_range must have two values"
        len(normalize_input_range) == 2 if normalize_input_range, "normalize_input_range must have two values"

schema Experiment:
    name?: str
    description?: str

schema Training:
    num_epochs: int = 10
    batch_size: int = 4
    shuffle_data: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor?: int
    dtype: str = "float32"
    resume_from_checkpoint: bool = False
    checkpoint_every: int = 100
    max_checkpoints: int = 5
    use_autocast: bool = True

    check:
        num_epochs > 0, "num_epochs must be positive"
        batch_size > 0, "batch_size must be positive"
        num_workers >= 0, "num_workers cannot be negative"
        prefetch_factor > 0 if prefetch_factor, "prefetch_factor must be positive"
        dtype in ["float32", "float16", "bfloat16"], "dtype must be float32, float16, or bfloat16"
        checkpoint_every > 0, "checkpoint_every must be positive"
        max_checkpoints > 0, "max_checkpoints must be positive"

schema Objective:
    loss_fn: "l1" | "l2" | "perceptual"
    perceptual_weight: float = 0.0

    check:
        perceptual_weight >= 0.0, "perceptual_weight must be non-negative"
        perceptual_weight <= 1.0, "perceptual_weight must be <= 1.0"

schema HParams:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2

    check:
        learning_rate > 0.0, "learning_rate must be positive"
        learning_rate < 1.0, "learning_rate suspiciously large (>=1.0)"
        weight_decay >= 0.0, "weight_decay must be non-negative"

schema Model:
    name: str
    compile: bool = False

schema Logging:
    wandb: bool = False
    save_volumes: bool = False
    viz_every: int = 100

    check:
        viz_every > 0, "viz_every must be positive"

schema Config:
    dims: "2d" | "3d"
    experiment: Experiment
    dataset: Dataset
    paths: Paths
    pipeline: Pipeline
    training: Training
    objective: Objective
    hparams: HParams
    model: Model
    logging: Logging
