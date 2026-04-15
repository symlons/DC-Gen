import torch
import wandb
import argparse
import logging
import os
import re
import signal
from collections import deque

from rich.console import Console

from print_utils import print_config_summary, print_param_group_modules, tensor_stats_dict
from monai.transforms import CenterSpatialCrop, Compose, Resize, ScaleIntensityRange, SpatialPad, RandSpatialCrop
from torch.utils.data import DataLoader, DistributedSampler
from data import collate_fn_skip_none

from basics import get_autocast_ctx, get_git_info, resolve_device, run
from checkpointing import load_checkpoint, save_checkpoint
from config import load_config, list_models
from dc_gen.ae_model_zoo import DCAE_HF, create_dc_ae_model_cfg
from evaluation import evaluate
from multigpu import cleanup, init_distributed, is_main_process, rank0_print, barrier, wrap_ddp, main_process_first
from train_validation import run_validation
from flow.ema import create_ema_model, update_ema_warmup
from ae_utils import compute_loss
from registry import GANLoss, dataset_registry, loss_registry
from viz import Visualize

_shutdown_requested = False
main_pid = os.getpid()
def _shutdown_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    if os.getpid() == main_pid: rank0_print(f"\n[{os.getpid()}] Shutdown signal received...")

def configure_trainable_params(model, trainable_ae_params):
    if trainable_ae_params is None: return [{"params": list(model.parameters())}]
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    named_params = list(base_model.named_parameters())
    used, groups, total_matched = set(), [], 0

    for names in trainable_ae_params:
        params = []
        for pattern_ in names:
            pattern = re.compile(pattern_)
            matched = False
            for p_name, param in named_params:
                if re.search(pattern, p_name):
                    pid = id(param)
                    if pid not in used:
                        params.append(param)
                        used.add(pid)
                        total_matched += param.numel()
                    matched = True
            if not matched:
                print(f"[WARNING] No match for pattern: {pattern_}")
        groups.append({"params": params})

    if total_matched == 0: print("[WARNING] No parameters matched ANY pattern — nothing will be trained.")
    train_params = set(p for g in groups for p in g["params"])

    for p in model.parameters(): p.requires_grad = False
    for p in train_params: p.requires_grad = True
    return groups

def build_pipeline(cfg):
    p = cfg.pipeline; lo, hi = p.normalize_output_range
    n, hw = p.n_slices, tuple(p.resize_hw)
    norm = ScaleIntensityRange(*p.normalize_input_range, b_min=lo, b_max=hi, clip=True)
    crop = (RandSpatialCrop((n,-1,-1), random_size=False, random_center=True) if p.random_slices else CenterSpatialCrop((n,-1,-1)))
    return Compose({"2d": [norm, Resize(hw)], "3d": [norm, SpatialPad((n,-1,-1), value=-1000), crop, Resize((-1,*hw))]}[cfg.dims])

def main_worker(rank: int, world_size: int, cfg):
    global _shutdown_requested
    signal.signal(signal.SIGTERM, _shutdown_handler)

    use_cuda = torch.cuda.is_available()
    device = resolve_device(rank)
    if use_cuda: torch.cuda.set_device(rank)
    if use_cuda and world_size > 1: init_distributed(rank, world_size)

    exp_name = cfg.experiment.name
    log_dir = os.path.join(cfg.paths.save_dir, exp_name, "logs")
    viz_dir = os.path.join(cfg.paths.save_dir, exp_name, "viz")
    log_metrics = cfg.logging.wandb

    if log_metrics and is_main_process():
        git_info = get_git_info()
        cfg_dict = vars(cfg)
        cfg_dict.update(git_info)
        wandb.init(project="ae_v1", name=cfg.experiment.name, config=cfg_dict)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        print_config_summary(cfg, device)

    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        import sys
        log_path = os.path.join(log_dir, "stdout.log")
        log_f = open(log_path, "a", buffering=1)

        class Tee:
            def write(self, data):
                sys.__stdout__.write(data)
                log_f.write(data)
            def flush(self):
                sys.__stdout__.flush()
                log_f.flush()

        sys.stdout = sys.stderr = Tee()

        console = Console()
        exp_root = os.path.join(cfg.paths.save_dir, exp_name)
        print("Experiment root:", exp_root)
        print("Checkpoints:", cfg.paths.checkpoint_dir)

    num_workers = cfg.training.num_workers
    if use_cuda and world_size > 1: num_workers = max(1, num_workers // world_size)

    pipeline = build_pipeline(cfg)
    dataset_cls = dataset_registry[cfg.dataset.name]
    rank0_print("[setup] Building datasets (rank 0 first to populate index cache)...")
    with main_process_first():
        dataset = dataset_cls(nifti_dir=cfg.paths.nifti_dir, group_names=cfg.dataset.group_names, dims=cfg.dims, n_slices=None, transform=pipeline)
        val_dataset = dataset_cls(nifti_dir=cfg.paths.nifti_val_dir, group_names=cfg.dataset.group_names, dims=cfg.dims, n_slices=None, transform=pipeline)
    rank0_print(f"[setup] Datasets ready. Train: {len(dataset)}, Val: {len(val_dataset)}")

    if use_cuda and world_size > 1: sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=cfg.training.shuffle_data, drop_last=True) 
    else: sampler = None


    dl_kw = dict(pin_memory=cfg.training.pin_memory, num_workers=num_workers,
                 prefetch_factor=cfg.training.prefetch_factor, persistent_workers=num_workers > 0)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_cuda and world_size > 1 else None
    loader      = DataLoader(dataset,     batch_size=cfg.training.batch_size,   shuffle=(sampler is None and cfg.training.shuffle_data), sampler=sampler, **dl_kw)
    val_loader  = DataLoader(val_dataset, batch_size=cfg.training.batch_size*2, shuffle=False, sampler=val_sampler, collate_fn=collate_fn_skip_none, **dl_kw)

    # todo: this needs to be cleand up
    rank0_print("[setup] Loading model...")
    dtype = getattr(torch, cfg.training.dtype)
    kwargs = {"model_cfg": create_dc_ae_model_cfg(variant=cfg.model_variant)} if getattr(cfg, "model_variant", None) else {"model_name": cfg.model.name}
    if "model_cfg" in kwargs: rank0_print(f"[setup] Model variant: {cfg.model_variant}")
    model = DCAE_HF(**kwargs).to(dtype=dtype, device=device)
    param_groups = configure_trainable_params(model, cfg.training.trainable_ae_params)
    if cfg.logging.print_model_arch: print_param_group_modules(model, param_groups)

    if getattr(cfg.model, "compile", False):
        rank0_print("[setup] Compiling model with torch.compile...")
        try: model = torch.compile(model)
        except Exception as e: rank0_print(f"[setup] torch.compile failed: {e}")

    if use_cuda and world_size > 1: model = wrap_ddp(model)
    model.train()
    ema_model = create_ema_model(model)
    rank0_print("[setup] Model ready.")

    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    viz = Visualize(viz_type=cfg.dims)

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.hparams.learning_rate, weight_decay=cfg.hparams.weight_decay)
    global_step, wandb_run_id = load_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, device, ema_model=ema_model)
    rank0_print(f"[setup] Checkpoint loaded, starting at global_step={global_step}, wandb_run_id={wandb_run_id}")

    # Loss init
    loss_fns = []
    loss_fns.append(("recon", loss_registry[cfg.objective.loss_fn], 1.0))
    perceptual = loss_registry["perceptual"](device)
    for p in perceptual.parameters(): p.requires_grad = False
    loss_fns.append(("perceptual", perceptual, cfg.objective.perceptual_weight))
    loss_fns.append(("ssim_loss", loss_registry["ssim_loss"], cfg.objective.ssim_weight))
    loss_fns.append(("grad", loss_registry["grad"](), cfg.objective.grad_weight))
    gan = GANLoss(cfg, device) if cfg.objective.gan_weight > 0 else None

    checkpoint_queue = deque()
    num_epochs = cfg.training.num_epochs

    rank0_print(f"[setup] Starting training loop (global_step={global_step})...")
    for epoch in range(num_epochs):
        if _shutdown_requested:
            rank0_print(f"\n[{rank}] Shutdown requested, exiting training loop")
            break
        if sampler: sampler.set_epoch(epoch)
        rank0_print(f"epoch: {epoch} out of {num_epochs}")
        train_size = len(dataset)
        val_size = len(val_dataset)
        rank0_print(f"train set: {train_size}, val set: {val_size}")

        for i, batch in enumerate(loader):
            if _shutdown_requested:
                rank0_print(f"\n[{rank}] Shutdown requested, finishing epoch")
                break
            if i == 0: rank0_print(f"  batch shape: {batch.shape}")
            save_diff = (cfg.logging.save_volumes and global_step % cfg.logging.viz_every == 0)
            save_ckpt = global_step % cfg.training.checkpoint_every == 0
            do_validation = (cfg.logging.validate_every is not None and global_step > 0 and global_step % cfg.logging.validate_every == 0)

            batch = batch.to(dtype=dtype, device=device, non_blocking=True)
            if hasattr(batch, "as_tensor"): batch = batch.as_tensor()

            with get_autocast_ctx(cfg, device):
                recon, c_prime = model(batch, use_mask=True)
            real, fake = batch.float(), recon.float()
            loss_terms, loss = compute_loss(fake, real, loss_fns, gan, global_step, cfg)
            optimizer.zero_grad(); loss.backward(); grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            update_ema_warmup(ema_model, model, global_step, decay=0.9999, warmup_steps=2000)

            if is_main_process():
                with torch.no_grad(): results_dict = evaluate(recon.detach(), batch.detach())
                fmt = lambda k, v: f"{k}={v:.3f}" if not torch.is_tensor(v) else f"{k}={[round(x.item(), 3) for x in v.flatten()[torch.randperm(v.numel())[:min(4, v.numel())]]]}"
                metrics_str = ", ".join(fmt(k, v) for k, v in results_dict.items())

                rank0_print(f"iter {global_step}: {metrics_str}")

                if log_metrics:
                    wandb.log({
                        "loss/total": loss.item(),
                        **{f"loss/{k}": v.item() for k, v in loss_terms.items()},
                        **{f"train/{k}": v for k, v in results_dict.items()},
                        "grad_norm": grad_norm,
                        "epoch": epoch,
                        **tensor_stats_dict("batch", batch),
                        **tensor_stats_dict("recon", recon),
                    }, step=global_step)

            if save_diff: barrier(); viz.save(batch, recon, viz_dir, global_step) if is_main_process() else None; barrier()
            if do_validation:
                rank0_print(f"\n[val] Starting validation at step {global_step}...")
                val_result = run_validation(
                    ema_model, val_loader, device, dtype, cfg,
                    forward_fn=lambda m, x: m(x),
                    loss_fn=lambda recon, real: compute_loss(recon.float(), real.float(), loss_fns, None, global_step, cfg)[1],
                    global_step=global_step,
                    log_metrics=log_metrics,
                    shutdown_flag=lambda: _shutdown_requested,
                )
                if is_main_process() and val_result:
                    val_str = ", ".join(fmt(k, v) for k, v in val_result.items())

                    if log_metrics:
                        wandb.log({
                            **{
                                f"val/{k}": (vv.item() if torch.is_tensor(vv) and vv.numel() == 1
                                    else vv.detach().cpu().numpy() if torch.is_tensor(vv)
                                    else vv
                                )
                                for k, v in val_result.items() if not k.endswith("_full")
                                for vv in [(v.as_tensor() if hasattr(v, "as_tensor") else v)]
                            },
                            **{
                                f"val/{k}_hist": wandb.Histogram(v.detach().cpu().numpy())
                                for k, v in val_result.items() if k.endswith("_full")
                            }
                        }, step=global_step)


            if save_ckpt:
                barrier()
                if is_main_process(): save_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, global_step, checkpoint_queue, cfg.training.max_checkpoints, ema_model=ema_model)
                barrier()
            global_step += 1

    rank0_print(f"\n[{rank}] Finalizing training...")
    if log_metrics and wandb.run is not None: wandb.finish()
    if use_cuda and world_size > 1: barrier(); cleanup()
    rank0_print(f"[{rank}] Training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Name of model in registry.yaml")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models: list_models(); exit(0)

    cfg = load_config(yaml_path=args.config, experiment=args.experiment, model=args.model)
    run(cfg, main_worker)
