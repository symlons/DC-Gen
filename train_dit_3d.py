import argparse
import os
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
import nibabel as nib
from torch.utils.data import DistributedSampler

from dc_gen.ae_model_zoo import DCAE_HF
from basics import get_autocast_ctx, move_batch, resolve_device, should_run, torch_dtype
from checkpointing import load_checkpoint, save_checkpoint

from flow.dit import DiT, DiT_models
from flow.ema import create_ema_model, update_ema_warmup
from flow.flow_config import TrainDiT3DConfig
from flow.flow_utils import print_config_summary
from flow.samplers import euler_sample
from flow.inspection import ModelInspector, flatten_inspection_stats, format_inspection_summary, get_grad_norm
from flow.latent_dataset import infer_latent_shape
from flow.logging_utils import format_step_log, log_rank0, tensor_stats_dict
from flow.rectified_flow import RectifiedFlowObjective

from multigpu import barrier, cleanup, init_distributed, is_main_process, rank0_print, wrap_ddp
from train_data import make_dataloader, make_dataset
from train_validation import run_validation


import sys
import sysconfig
_python_include = sysconfig.get_path("include") # for torch.compile/triton
if _python_include and os.path.isfile(os.path.join(_python_include, "Python.h")):
    os.environ["CPATH"] = _python_include + os.pathsep + os.environ.get("CPATH", "")
else:
    _fallback = f"/opt/python/{sysconfig.get_python_version()}.4/include/python{sysconfig.get_python_version()}"
    if os.path.isfile(os.path.join(_fallback, "Python.h")):
        os.environ["CPATH"] = _fallback + os.pathsep + os.environ.get("CPATH", "")

def load_autoencoder(cfg: TrainDiT3DConfig, model_dtype, device):
    import glob
    from dc_gen.ae_model_zoo import create_dc_ae_model_cfg
    from dc_gen.aecore.models.dc_ae import DCAE
    
    if cfg.model.ae_checkpoint_dir is not None:
        checkpoint_dir = cfg.model.ae_checkpoint_dir
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_iter*.pt")))
        if not checkpoint_files: raise FileNotFoundError(f"No checkpoint_iter*.pt files found in {checkpoint_dir}")
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
    num_classes = max(1, cfg.model.num_classes)
    class_dropout_prob = max(0.0, cfg.model.class_dropout_prob)
    if cfg.model.variant != "custom": return DiT_models[cfg.model.variant](num_classes=num_classes, class_dropout_prob=class_dropout_prob)
    return DiT(
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        patch_size=cfg.model.patch_size,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
    )

def main_worker(rank: int, world_size: int, cfg: TrainDiT3DConfig):
    device = resolve_device(rank)
    use_cuda = device.type == "cuda"
    use_ddp = use_cuda and world_size > 1
    if use_ddp: init_distributed(rank, world_size)

    train_dataset = make_dataset(cfg, "train")
    val_dataset = make_dataset(cfg, "val")
    sampler = (DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True if use_ddp else cfg.training.shuffle_data) if use_ddp else None)
    train_loader = make_dataloader(train_dataset, cfg, sampler=sampler, shuffle=not use_ddp and cfg.training.shuffle_data)
    val_loader = make_dataloader(val_dataset, cfg, batch_size=min(len(val_dataset), max(1, cfg.sampling.batch_size)))
    in_channels, spatial_shape = infer_latent_shape(
        train_dataset,
        expected_in_channels=cfg.model.in_channels,
        expected_input_size=tuple(cfg.model.input_size) if cfg.model.input_size else None,
    )

    model_dtype = torch_dtype(cfg.training.dtype) # (fp32) master weights
    model = build_model(cfg, in_channels, spatial_shape).to(device=device, dtype=model_dtype)
    ema_model = create_ema_model(model).to(device=device, dtype=model_dtype) if cfg.training.ema_decay is not None else None
    autoencoder = load_autoencoder(cfg, model_dtype, device)

    if cfg.training.compile: model = torch.compile(model)
    if use_ddp: model = wrap_ddp(model, device)
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
    inspector = ModelInspector(
        model,
        module_names=cfg.inspection.module_names,
        capture_weights=cfg.inspection.capture_weights,
        capture_gradients=cfg.inspection.capture_gradients,
        capture_activations=cfg.inspection.capture_activations,
    ) if cfg.inspection.enabled else None

    for d in (cfg.paths.save_dir, cfg.sampling.output_dir, cfg.paths.checkpoint_dir): os.makedirs(d, exist_ok=True)
    global_step, _ = load_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, device, ema_model=ema_model)
    checkpoint_queue = deque()
    log_file = os.path.join(cfg.paths.save_dir, "training_log.txt")

    if is_main_process():
        print_config_summary(cfg, device, in_channels, spatial_shape)
        rank0_print(model)
        if cfg.logging.wandb: wandb.init(project=cfg.logging.wandb_project, name=cfg.experiment.name, config=asdict(cfg))

    eval_model = ema_model or model 
    for epoch in range(cfg.training.num_epochs):
        sampler and sampler.set_epoch(epoch)
        log_rank0(f"[epoch] {format_step_log(global_step, epoch, {'dataset_size': float(len(train_dataset))})}", log_file)

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0: log_rank0(f"[batch] shape={tuple(batch['image'].shape)}", log_file)

            batch = move_batch(batch, device, model_dtype)
            optimizer.zero_grad(set_to_none=True)
            with get_autocast_ctx(cfg, device): outputs = objective.compute_loss(model, batch["image"])
            loss = outputs["loss"]
            loss.backward()

            grad_norm = get_grad_norm(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()

            if ema_model is not None: update_ema_warmup(ema_model, model, global_step + 1, decay=cfg.training.ema_decay, warmup_steps=cfg.training.ema_warmup_steps)
            if is_main_process() and global_step % cfg.training.log_every == 0:
                log_rank0(f"[train] {format_step_log(global_step, epoch, {'loss': loss.item(), 't_mean': outputs['t'].float().mean().item(), 'grad_norm': grad_norm})}", log_file)
                if cfg.logging.wandb:
                    wandb.log({
                        "loss/total": loss.item(),
                        "timestep/mean": outputs["t"].float().mean().item(),
                        "train/grad_norm": grad_norm,
                        **tensor_stats_dict("latent", batch["image"]),
                        **tensor_stats_dict("x_t", outputs["x_t"]),
                        **tensor_stats_dict("v_pred", outputs["v_pred"]),
                        **tensor_stats_dict("x1_pred", outputs["x1_pred"]),
                    }, step=global_step)

            # Validation
            if should_run(global_step, cfg.logging.validate_every):
                barrier()
                if is_main_process() and val_loader is not None:
                    autoencoder.to(device)
                    val_metrics = run_validation(eval_model, val_loader, objective, cfg, device, model_dtype, autoencoder)
                    log_rank0(f"[val] {format_step_log(global_step, epoch, val_metrics)}", log_file)
                    if cfg.logging.wandb: wandb.log(val_metrics, step=global_step)
                    autoencoder.to("cpu")
                    torch.cuda.empty_cache()
                barrier()

            # Sampling
            if cfg.sampling.enabled and cfg.logging.save_samples and should_run(global_step, cfg.sampling.sample_every):
                barrier()
                if is_main_process():
                    noise = torch.randn((cfg.sampling.batch_size, in_channels, *spatial_shape), device=device, dtype=model_dtype)
                    x = euler_sample(eval_model, noise, cfg.sampling.sample_steps)
                    x = x * cfg.objective.latent_std + cfg.objective.latent_mean

                    from PIL import Image
                    import nibabel as nib

                    if autoencoder is not None and device is not None:
                        autoencoder.to(device)
                        with torch.no_grad():
                            recon = autoencoder.decode(x.to(device)).to(x.device)
                        autoencoder.to("cpu")
                        torch.cuda.empty_cache()

                    for i in range(recon.shape[0]):
                        img = recon[i, 0, recon.shape[2] // 2]
                        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).byte().cpu().numpy()
                        Image.fromarray(img).save(f"{cfg.sampling.output_dir}/sample_{global_step}_{i}.png")

                        vol = recon[i, 0].detach().cpu().numpy()
                        nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), f"{cfg.sampling.output_dir}/sample_{global_step}_{i}.nii.gz")

                    print(recon.shape)
                    log_rank0(f"[sample] step={global_step} saved_to={cfg.sampling.output_dir}", log_file)
                barrier()

            if should_run(global_step, cfg.training.checkpoint_every):
                barrier()
                if is_main_process():
                    save_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, global_step, checkpoint_queue, cfg.training.max_checkpoints, ema_model=ema_model)
                    log_rank0(f"[ckpt] step={global_step} dir={cfg.paths.checkpoint_dir}", log_file)
                barrier()

            global_step += 1

    if is_main_process() and cfg.logging.wandb and wandb.run is not None: wandb.finish()
    if use_ddp: cleanup()


def load_config(experiment=None):
    from omegaconf import OmegaConf
    from flow.flow_utils import validate_and_finalize_config
    
    cfg = OmegaConf.structured(TrainDiT3DConfig)
    if experiment:
        exp = Path(experiment)
        if not exp.suffix: exp = exp.with_suffix(".yaml")
        if not exp.is_absolute(): exp = Path(__file__).parent / ("configs/experiments" / exp if len(exp.parts) == 1 else exp)
        if not exp.exists(): raise FileNotFoundError(f"Experiment config not found: {exp}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(exp))
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
