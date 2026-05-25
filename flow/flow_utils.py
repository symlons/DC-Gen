import os
import torch
from .flow_config import TrainDiT3DConfig, TRAIN_TIMESTEP_MODES, SAMPLE_TIME_SCHEDULES
from .dit import DiT_models
from .logging_utils import format_kv_block
from multigpu import rank0_print

DTYPE_NAME_MAP = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}

def _resolve_autocast_dtype(cfg, device):
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

def print_config_summary(cfg: TrainDiT3DConfig, device: torch.device, in_channels: int, spatial_shape: tuple[int, int, int]):
     rows = [
         ("Experiment", cfg.experiment.name),
         ("Device", device),
         ("HDF Path", cfg.paths.hdf_path),
         ("NIfTI Train", cfg.paths.nifti_dir),
         ("NIfTI Val", cfg.paths.nifti_val_dir),
         ("Latent Train", cfg.dataset.train_dir),
         ("Latent Val", cfg.dataset.val_dir),
         ("Channels", in_channels),
         ("Spatial Size", spatial_shape),
         ("Model", cfg.model.variant),
         ("Patch Size", cfg.model.patch_size),
         ("Hidden Size", cfg.model.hidden_size),
         ("Depth", cfg.model.depth),
         ("Heads", cfg.model.num_heads),
         ("Num Classes", max(1, cfg.model.num_classes)),
         ("Objective", f"Rectified Flow ({cfg.objective.mode})"),
         ("Objective Beta", f"({cfg.objective.beta_a}, {cfg.objective.beta_b})"),
         ("TWEO", f"{cfg.tweo.enabled} (weight={cfg.tweo.weight}, tau={cfg.tweo.tau}, p={cfg.tweo.power}, schedule={cfg.tweo.schedule})"),
         ("Transformer Engine", f"{cfg.transformer_engine.enabled} (fp8={cfg.transformer_engine.fp8_autocast}, format={cfg.transformer_engine.recipe_format}, amax={cfg.transformer_engine.amax_history_len})"),
         ("Epochs", cfg.training.num_epochs),
         ("Batch Size", cfg.training.batch_size),
         ("Num Workers", cfg.training.num_workers),
         ("Persistent Workers", cfg.training.persistent_workers),
         ("DType", cfg.training.dtype),
         ("Autocast", f"{cfg.training.use_autocast} ({cfg.training.autocast_dtype} -> {_resolve_autocast_dtype(cfg, device)})"),
         ("LR", cfg.training.learning_rate),
         ("Weight Decay", cfg.training.weight_decay),
         ("EMA Decay", cfg.training.ema_decay),
         ("EMA Warmup", cfg.training.ema_warmup_steps),
         ("Sample Steps", cfg.sampling.sample_steps),
         ("Sampler", cfg.sampling.solver),
         ("Time Schedule", cfg.sampling.time_schedule),
         ("Schedule Power", cfg.sampling.time_schedule_power),
         ("Inspection", cfg.inspection.enabled),
         ("Inspect Every", cfg.inspection.inspect_every),
         ("Inspect Modules", cfg.inspection.module_names or "default"),
         ("Save Dir", cfg.paths.save_dir),
         ("Checkpoints", cfg.paths.checkpoint_dir),
         ("WandB", cfg.logging.wandb),
     ]
     rank0_print(format_kv_block("Config Summary", rows))

def validate_and_finalize_config(cfg: TrainDiT3DConfig) -> TrainDiT3DConfig:
    VALID_TRAIN_DTYPES = {"float32", "float16", "bfloat16"}
    VALID_AUTOCAST_DTYPES = {"auto", "float16", "bfloat16"}
    DTYPE_NAME_MAP = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
    if cfg.dims != "3d":
        raise ValueError("train_dit_3d.py only supports dims='3d'.")
    if cfg.training.dtype not in VALID_TRAIN_DTYPES:
        raise ValueError(f"training.dtype must be one of {sorted(VALID_TRAIN_DTYPES)}.")
    if cfg.training.autocast_dtype not in VALID_AUTOCAST_DTYPES:
        raise ValueError(f"training.autocast_dtype must be one of {sorted(VALID_AUTOCAST_DTYPES)}.")
    if cfg.model.variant != "custom" and cfg.model.variant not in DiT_models:
        raise ValueError(f"Unknown model.variant={cfg.model.variant!r}.")
    if cfg.model.num_classes < 1:
        raise ValueError("model.num_classes must be at least 1.")
    if cfg.objective.mode not in TRAIN_TIMESTEP_MODES:
        raise ValueError(f"objective.mode must be one of {sorted(TRAIN_TIMESTEP_MODES)}.")
    if cfg.objective.beta_a <= 0 or cfg.objective.beta_b <= 0:
        raise ValueError("objective.beta_a and objective.beta_b must be positive.")
    if cfg.sampling.solver not in {"euler", "heun"}:
        raise ValueError("sampling.solver must be one of ['euler', 'heun'].")
    if cfg.sampling.time_schedule not in SAMPLE_TIME_SCHEDULES:
        raise ValueError(f"sampling.time_schedule must be one of {sorted(SAMPLE_TIME_SCHEDULES)}.")
    if cfg.sampling.time_schedule_power <= 0:
        raise ValueError("sampling.time_schedule_power must be positive.")
    if cfg.model.learn_sigma:
        raise ValueError("Rectified-flow training expects model.learn_sigma=False.")
    if cfg.training.ema_decay is not None and not 0.0 <= cfg.training.ema_decay < 1.0:
        raise ValueError("training.ema_decay must be in [0, 1) or None.")
    if cfg.training.ema_warmup_steps < 0:
        raise ValueError("training.ema_warmup_steps must be non-negative.")
    if cfg.inspection.inspect_every is not None and cfg.inspection.inspect_every <= 0:
        raise ValueError("inspection.inspect_every must be positive or None.")
    if cfg.tweo.weight < 0:
        raise ValueError("tweo.weight must be non-negative.")
    if cfg.tweo.tau <= 0:
        raise ValueError("tweo.tau must be positive.")
    if cfg.tweo.power <= 0:
        raise ValueError("tweo.power must be positive.")
    if cfg.tweo.eps < 0:
        raise ValueError("tweo.eps must be non-negative.")
    if cfg.tweo.schedule not in {"constant", "cosine"}:
        raise ValueError("tweo.schedule must be one of ['constant', 'cosine'].")
    if cfg.transformer_engine.recipe_format not in {"E4M3", "E5M2", "HYBRID"}:
        raise ValueError("transformer_engine.recipe_format must be one of ['E4M3', 'E5M2', 'HYBRID'].")
    if cfg.transformer_engine.amax_history_len <= 0:
        raise ValueError("transformer_engine.amax_history_len must be positive.")

    cfg.paths.checkpoint_dir = cfg.paths.checkpoint_dir or os.path.join(
        cfg.paths.checkpoint_root_dir, "flow_matching", cfg.experiment.name
    )
    cfg.paths.save_dir = cfg.paths.save_dir or os.path.join(
        cfg.paths.save_root_dir, "flow", cfg.experiment.name
    )
    cfg.dataset.train_dir = cfg.dataset.train_dir or cfg.paths.latent_train_dir
    cfg.dataset.val_dir = cfg.dataset.val_dir or cfg.paths.latent_val_dir
    cfg.sampling.output_dir = cfg.sampling.output_dir or os.path.join(cfg.paths.save_dir, "samples")
    return cfg
