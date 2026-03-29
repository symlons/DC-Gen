import argparse
import os
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import wandb
from dc_gen.models.utils.network import get_dtype_from_str
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler

from basics import get_autocast_ctx, resolve_autocast_dtype
from checkpointing import load_checkpoint, save_checkpoint
from evaluation import evaluate
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


def resolve_device(rank: int) -> torch.device:
    use_cuda = torch.cuda.is_available()
    return torch.device("mps" if torch.backends.mps.is_available() and not use_cuda else f"cuda:{rank}" if use_cuda else "cpu")


def torch_dtype(name: str) -> torch.dtype:
    return get_dtype_from_str(DTYPE_NAME_MAP[name])


def should_run(step: int, every: Optional[int]) -> bool:
    return every is not None and step > 0 and step % every == 0


def append_log(log_file: Optional[str], *lines: str):
    if not log_file:
        return
    with open(log_file, "a") as f:
        for line in lines:
            f.write(line + "\n")


def log_rank0(message: str, log_file: Optional[str] = None):
    rank0_print(message)
    append_log(log_file, message)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=dtype, non_blocking=True) for key, value in batch.items()}


def make_dataset(cfg: TrainDiT3DConfig, split: str) -> LatentTensorDataset:
    return LatentTensorDataset(
        root_dir=cfg.dataset.train_dir if split == "train" else cfg.dataset.val_dir,
        extensions=tuple(cfg.dataset.extensions),
        recursive=cfg.dataset.recursive,
        fraction=cfg.dataset.fraction if split == "train" else cfg.dataset.val_fraction,
        seed=cfg.dataset.seed,
    )


def make_dataloader(dataset, cfg: TrainDiT3DConfig, sampler=None, shuffle=False, batch_size=None):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size or cfg.training.batch_size,
        "shuffle": shuffle,
        "pin_memory": cfg.training.pin_memory,
        "num_workers": cfg.training.num_workers,
        "persistent_workers": cfg.training.num_workers > 0 and cfg.training.persistent_workers,
        "sampler": sampler,
    }
    if cfg.training.num_workers and cfg.training.prefetch_factor is not None:
        kwargs["prefetch_factor"] = cfg.training.prefetch_factor
    return DataLoader(**kwargs)


def build_model(cfg: TrainDiT3DConfig, in_channels: int, spatial_shape: tuple[int, int, int]) -> DiT:
    common_kwargs = {
        "input_size": spatial_shape,
        "in_channels": in_channels,
        "num_classes": max(1, cfg.model.num_classes),
        "class_dropout_prob": cfg.model.class_dropout_prob,
        "learn_sigma": cfg.model.learn_sigma,
        "mlp_ratio": cfg.model.mlp_ratio,
    }
    if cfg.model.variant != "custom":
        return DiT_models[cfg.model.variant](**common_kwargs)
    return DiT(
        patch_size=cfg.model.patch_size,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        **common_kwargs,
    )


def compute_step_metrics(outputs: dict[str, torch.Tensor], images: torch.Tensor):
    return evaluate(outputs["x1_pred"].float(), images.float(), outputs["loss"].detach().float())


def get_grad_norm(parameters, clip_norm: Optional[float] = None) -> float:
    params = [param for param in parameters if param.grad is not None]
    if not params:
        return 0.0
    if clip_norm is not None:
        return float(clip_grad_norm_(params, clip_norm).item())
    return float(torch.norm(torch.stack([param.grad.detach().float().norm() for param in params])).item())


def run_validation(
    model,
    val_loader,
    objective: RectifiedFlowObjective,
    cfg: TrainDiT3DConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    values = {key: [] for key in ("val_loss/total", "val_metrics/psnr", "val_metrics/ssim", "val_metrics/slice_psnr", "val_metrics/slice_ssim")}
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            if cfg.logging.max_val_batches is not None and batch_idx >= cfg.logging.max_val_batches:
                break
            batch = move_batch(batch, device, dtype)
            with get_autocast_ctx(cfg, device):
                outputs = objective.compute_loss(model, batch["image"])
            for key, value in zip(values, compute_step_metrics(outputs, batch["image"])):
                values[key].append(value)
    if was_training:
        model.train()
    return {key: float(torch.tensor(item).mean().item()) for key, item in values.items() if item}


@torch.no_grad()
def save_sample_batch(
    model,
    cfg: TrainDiT3DConfig,
    spatial_shape: tuple[int, int, int],
    in_channels: int,
    device: torch.device,
    global_step: int,
):
    if not cfg.sampling.enabled:
        return
    os.makedirs(cfg.sampling.output_dir, exist_ok=True)
    save_dtype = torch_dtype(cfg.sampling.save_dtype)
    generated, batch_size, total = [], min(cfg.sampling.batch_size, cfg.sampling.num_samples), cfg.sampling.num_samples
    for start in range(0, total, batch_size):
        noise = torch.randn(min(batch_size, total - start), in_channels, *spatial_shape, device=device)
        generated.append(
            sample_velocity_model(
                model,
                noise,
                cfg.sampling.sample_steps,
                solver=cfg.sampling.solver,
                t_start=cfg.sampling.t_start,
                t_end=cfg.sampling.t_end,
                time_schedule=cfg.sampling.time_schedule,
                time_schedule_power=cfg.sampling.time_schedule_power,
            ).cpu().to(save_dtype)
        )
    torch.save(torch.cat(generated)[:total], os.path.join(cfg.sampling.output_dir, f"latent_samples_step{global_step:07d}.pt"))


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
        ("Epochs", cfg.training.num_epochs),
        ("Batch Size", cfg.training.batch_size),
        ("Num Workers", cfg.training.num_workers),
        ("Persistent Workers", cfg.training.persistent_workers),
        ("DType", cfg.training.dtype),
        ("Autocast", f"{cfg.training.use_autocast} ({cfg.training.autocast_dtype} -> {resolve_autocast_dtype(cfg, device)})"),
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

    cfg.paths.checkpoint_dir = cfg.paths.checkpoint_dir or os.path.join(
        cfg.paths.checkpoint_root_dir, "flow_matching", cfg.experiment.name
    )
    cfg.paths.save_dir = cfg.paths.save_dir or os.path.join(
        cfg.paths.save_root_dir, "flow_matching", cfg.experiment.name
    )
    cfg.dataset.train_dir = cfg.dataset.train_dir or cfg.paths.latent_train_dir
    cfg.dataset.val_dir = cfg.dataset.val_dir or cfg.paths.latent_val_dir
    cfg.sampling.output_dir = cfg.sampling.output_dir or os.path.join(cfg.paths.save_dir, "samples")
    return cfg


def load_config(yaml_path: Optional[str] = None) -> TrainDiT3DConfig:
    cfg = OmegaConf.structured(TrainDiT3DConfig)
    if yaml_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(yaml_path))
    return validate_and_finalize_config(OmegaConf.to_object(cfg))


def main_worker(rank: int, world_size: int, cfg: TrainDiT3DConfig):
    device = resolve_device(rank)
    use_cuda = device.type == "cuda"
    if use_cuda and world_size > 1:
        init_distributed(rank, world_size)

    train_dataset = make_dataset(cfg, "train")
    val_dataset = make_dataset(cfg, "val") if is_main_process() or world_size == 1 else None
    sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=cfg.training.shuffle_data)
        if use_cuda and world_size > 1
        else None
    )
    train_loader = make_dataloader(train_dataset, cfg, sampler=sampler, shuffle=sampler is None and cfg.training.shuffle_data)
    val_loader = (
        make_dataloader(val_dataset, cfg, batch_size=max(1, cfg.training.batch_size))
        if val_dataset is not None
        else None
    )

    in_channels, spatial_shape = infer_latent_shape(
        train_dataset,
        expected_in_channels=cfg.model.in_channels,
        expected_input_size=tuple(cfg.model.input_size) if cfg.model.input_size is not None else None,
    )
    model_dtype = torch_dtype(cfg.training.dtype)
    model = build_model(cfg, in_channels, spatial_shape).to(device=device, dtype=model_dtype)
    ema_model = create_ema_model(model).to(device=device, dtype=model_dtype) if cfg.training.ema_decay is not None else None
    if cfg.training.compile:
        model = torch.compile(model)
    if use_cuda and world_size > 1:
        model = wrap_ddp(model, device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=device.type == "cuda" and cfg.training.use_autocast and resolve_autocast_dtype(cfg, device) == torch.float16,
    )
    objective = RectifiedFlowObjective(
        device=device,
        eps=cfg.objective.eps,
        mode=cfg.objective.mode,
        mu=cfg.objective.mu,
        sigma=cfg.objective.sigma,
        beta_a=cfg.objective.beta_a,
        beta_b=cfg.objective.beta_b,
    )
    inspector = (
        ModelInspector(
            model,
            module_names=cfg.inspection.module_names,
            capture_weights=cfg.inspection.capture_weights,
            capture_gradients=cfg.inspection.capture_gradients,
            capture_activations=cfg.inspection.capture_activations,
        )
        if cfg.inspection.enabled
        else None
    )

    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    global_step = load_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, device, ema_model=ema_model)
    checkpoint_queue = deque()
    log_file = os.path.join(cfg.paths.save_dir, "training_log.txt")

    if is_main_process():
        print_config_summary(cfg, device, in_channels, spatial_shape)
        rank0_print(model)
        if cfg.logging.wandb:
            wandb.init(project=cfg.logging.wandb_project, name=cfg.experiment.name, config=asdict(cfg))

    eval_model = ema_model if ema_model is not None else model
    for epoch in range(cfg.training.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if is_main_process():
            log_rank0(f"[epoch] {format_step_log(global_step, epoch, {'dataset_size': float(len(train_dataset))})}", log_file)

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0 and is_main_process():
                log_rank0(f"[batch] shape={tuple(batch['image'].shape)}", log_file)

            batch = move_batch(batch, device, model_dtype)
            optimizer.zero_grad(set_to_none=True)
            with get_autocast_ctx(cfg, device):
                outputs = objective.compute_loss(model, batch["image"])

            loss = outputs["loss"]
            inspection_stats = None
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if inspector is not None and should_run(global_step, cfg.inspection.inspect_every):
                    inspection_stats = inspector.collect()
                grad_norm = get_grad_norm(model.parameters(), cfg.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if inspector is not None and should_run(global_step, cfg.inspection.inspect_every):
                    inspection_stats = inspector.collect()
                grad_norm = get_grad_norm(model.parameters(), cfg.training.grad_clip_norm)
                optimizer.step()

            if ema_model is not None:
                update_ema_warmup(
                    ema_model,
                    model,
                    global_step + 1,
                    decay=cfg.training.ema_decay,
                    warmup_steps=cfg.training.ema_warmup_steps,
                )

            metrics = dict(
                zip(
                    ("loss", "psnr", "ssim", "slice_psnr", "slice_ssim"),
                    compute_step_metrics(outputs, batch["image"]),
                )
            )
            metrics["t_mean"] = outputs["t"].float().mean().item()
            metrics["grad_norm"] = grad_norm

            if is_main_process() and global_step % cfg.training.log_every == 0:
                log_rank0(f"[train] {format_step_log(global_step, epoch, metrics)}", log_file)
                if cfg.logging.wandb:
                    wandb_metrics = {
                        "loss/total": metrics["loss"],
                        "metrics/psnr": metrics["psnr"],
                        "metrics/ssim": metrics["ssim"],
                        "metrics/slice_psnr": metrics["slice_psnr"],
                        "metrics/slice_ssim": metrics["slice_ssim"],
                        "timestep/mean": metrics["t_mean"],
                        "train/grad_norm": metrics["grad_norm"],
                        **tensor_stats_dict("latent", batch["image"]),
                        **tensor_stats_dict("x_t", outputs["x_t"]),
                        **tensor_stats_dict("v_pred", outputs["v_pred"]),
                        **tensor_stats_dict("x1_pred", outputs["x1_pred"]),
                    }
                    if inspection_stats is not None and cfg.inspection.log_to_wandb:
                        wandb_metrics.update(flatten_inspection_stats(inspection_stats))
                    wandb.log(wandb_metrics, step=global_step)
            elif is_main_process() and inspection_stats is not None and cfg.logging.wandb and cfg.inspection.log_to_wandb:
                wandb.log(flatten_inspection_stats(inspection_stats), step=global_step)

            if is_main_process() and inspection_stats is not None and cfg.inspection.print_summary:
                for line in format_inspection_summary(inspection_stats):
                    log_rank0(f"[model_stats] {line}", log_file)

            if should_run(global_step, cfg.logging.validate_every):
                barrier()
                if is_main_process() and val_loader is not None:
                    val_metrics = run_validation(eval_model, val_loader, objective, cfg, device, model_dtype)
                    log_rank0(f"[val] {format_step_log(global_step, epoch, val_metrics)}", log_file)
                    if cfg.logging.wandb:
                        wandb.log(val_metrics, step=global_step)
                barrier()

            if cfg.sampling.enabled and cfg.logging.save_samples and should_run(global_step, cfg.sampling.sample_every):
                barrier()
                if is_main_process():
                    save_sample_batch(eval_model, cfg, spatial_shape, in_channels, device, global_step)
                    log_rank0(f"[sample] step={global_step} saved_to={cfg.sampling.output_dir}", log_file)
                barrier()

            if should_run(global_step, cfg.training.checkpoint_every):
                barrier()
                if is_main_process():
                    save_checkpoint(
                        cfg,
                        model,
                        optimizer,
                        cfg.paths.checkpoint_dir,
                        global_step,
                        checkpoint_queue,
                        cfg.training.max_checkpoints,
                        ema_model=ema_model,
                    )
                    log_rank0(f"[ckpt] step={global_step} dir={cfg.paths.checkpoint_dir}", log_file)
                barrier()

            global_step += 1

    if inspector is not None:
        inspector.close()
    if is_main_process() and cfg.logging.wandb and wandb.run is not None:
        wandb.finish()
    if use_cuda and world_size > 1:
        cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file to override defaults")
    args = parser.parse_args()

    cfg = load_config(args.config)
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if torch.cuda.is_available() and world_size > 1:
        import torch.multiprocessing as mp

        mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, cfg)


if __name__ == "__main__":
    main()
