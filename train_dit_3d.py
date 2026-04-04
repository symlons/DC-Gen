import argparse
import os
from collections import deque
from dataclasses import asdict

import nibabel as nib
import numpy as np
import torch
import wandb
from dc_gen.ae_model_zoo import DCAE_HF
from torch.utils.data import DistributedSampler

from basics import (
    get_autocast_ctx,
    move_batch,
    resolve_device,
    should_run,
    torch_dtype,
)
from checkpointing import load_checkpoint, save_checkpoint
from flow.dit import DiT, DiT_models
from flow.ema import create_ema_model, update_ema_warmup
from flow.flow_config import TrainDiT3DConfig
from flow.flow_utils import print_config_summary
from flow.inspection import (
    ModelInspector,
    flatten_inspection_stats,
    format_inspection_summary,
    get_grad_norm,
)
from flow.latent_dataset import infer_latent_shape
from flow.logging_utils import format_step_log, log_rank0, tensor_stats_dict
from flow.rectified_flow import RectifiedFlowObjective
from flow_sampling import save_sample_batch
from multigpu import (
    barrier,
    cleanup,
    init_distributed,
    is_main_process,
    rank0_print,
    wrap_ddp,
)
from train_data import make_dataloader, make_dataset
from train_validation import compute_step_metrics, run_validation


def _load_autoencoder(cfg: TrainDiT3DConfig, model_dtype, device):
    """Load autoencoder from checkpoint_dir (if specified) or from model_name."""
    import glob
    from dc_gen.ae_model_zoo import create_dc_ae_model_cfg
    from dc_gen.aecore.models.dc_ae import DCAE
    
    if cfg.model.ae_checkpoint_dir is not None:
        checkpoint_dir = cfg.model.ae_checkpoint_dir
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_iter*.pt")))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint_iter*.pt files found in {checkpoint_dir}")
        latest_checkpoint = checkpoint_files[-1]
        rank0_print(f"Loading autoencoder from checkpoint: {latest_checkpoint}")
        
        ae_cfg = create_dc_ae_model_cfg(cfg.model.ae_model_name)
        ae_cfg.pretrained_path = None
        autoencoder = DCAE(ae_cfg)
        
        try:
            ckpt = torch.load(latest_checkpoint, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            autoencoder.load_state_dict(state_dict)
            rank0_print(f"Autoencoder loaded from global_step {ckpt.get('global_step', '?')}")
        except Exception as e:
            raise RuntimeError(f"Failed to load autoencoder checkpoint {latest_checkpoint}: {e}")
        
        autoencoder = autoencoder.to(dtype=model_dtype, device=device)
    else:
        rank0_print(f"Loading autoencoder from model_name: {cfg.model.ae_model_name}")
        autoencoder = DCAE_HF(model_name=cfg.model.ae_model_name).to(dtype=model_dtype, device=device)
    
    autoencoder.eval()
    return autoencoder


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


def main_worker(rank: int, world_size: int, cfg: TrainDiT3DConfig):
    device = resolve_device(rank)
    use_cuda = device.type == "cuda"
    if use_cuda and world_size > 1:
        init_distributed(rank, world_size)

    train_dataset = make_dataset(cfg, "train")
    val_dataset = make_dataset(cfg, "val")
    sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=cfg.training.shuffle_data)
        if use_cuda and world_size > 1
        else None
    )
    train_loader = make_dataloader(train_dataset, cfg, sampler=sampler, shuffle=sampler is None and cfg.training.shuffle_data)
    val_loader = (make_dataloader(val_dataset, cfg, batch_size=max(1, cfg.training.batch_size)) if val_dataset is not None else None)

    in_channels, spatial_shape = infer_latent_shape(
        train_dataset,
        expected_in_channels=cfg.model.in_channels,
        expected_input_size=tuple(cfg.model.input_size) if cfg.model.input_size is not None else None,
    )
    model_dtype = torch_dtype(cfg.training.dtype)
    model = build_model(cfg, in_channels, spatial_shape).to(device=device, dtype=model_dtype)
    ema_model = create_ema_model(model).to(device=device, dtype=model_dtype) if cfg.training.ema_decay is not None else None
    autoencoder = _load_autoencoder(cfg, model_dtype, device)


    if cfg.training.compile:
        model = torch.compile(model)
    if use_cuda and world_size > 1:
        model = wrap_ddp(model, device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    objective = RectifiedFlowObjective(
        device=device,
        eps=cfg.objective.eps,
        mode=cfg.objective.mode,
        mu=cfg.objective.mu,
        sigma=cfg.objective.sigma,
        beta_a=cfg.objective.beta_a,
        beta_b=cfg.objective.beta_b,
        latent_mean=cfg.objective.latent_mean,
        latent_std=cfg.objective.latent_std,
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
            loss.backward()
            if inspector is not None and should_run(global_step, cfg.inspection.inspect_every):
                inspection_stats = inspector.collect()
            grad_norm = get_grad_norm(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()

            if ema_model is not None: update_ema_warmup(ema_model, model, global_step + 1, decay=cfg.training.ema_decay, warmup_steps=cfg.training.ema_warmup_steps)

            metrics = dict(zip(("loss", "psnr", "ssim", "slice_psnr", "slice_ssim"), compute_step_metrics(outputs, batch["image"])))
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
                    val_metrics = run_validation(eval_model, val_loader, objective, cfg, device, model_dtype, autoencoder)
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
                    save_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, global_step, checkpoint_queue, cfg.training.max_checkpoints, ema_model=ema_model)
                    log_rank0(f"[ckpt] step={global_step} dir={cfg.paths.checkpoint_dir}", log_file)
                barrier()

            global_step += 1

    if inspector is not None:
        inspector.close()
    if is_main_process() and cfg.logging.wandb and wandb.run is not None:
        wandb.finish()
    if use_cuda and world_size > 1:
        cleanup()


def load_config(experiment=None):
    from omegaconf import OmegaConf
    from flow.flow_utils import validate_and_finalize_config
    
    cfg = OmegaConf.structured(TrainDiT3DConfig)
    if experiment:
        yaml_path = os.path.join(os.path.dirname(__file__), "configs", "experiments", f"{experiment}.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Experiment config not found: {yaml_path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(yaml_path))
    return validate_and_finalize_config(OmegaConf.to_object(cfg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None, help="Name of experiment config in configs/experiments/ (e.g. flow_1)")
    args = parser.parse_args()

    cfg = load_config(args.experiment)
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if torch.cuda.is_available() and world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, cfg)

if __name__ == "__main__":
    main()
