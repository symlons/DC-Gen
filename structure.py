import argparse
import io
import logging
import os
import re
import signal
import subprocess
import sys
import sysconfig
from collections import deque
from monai.data import set_track_meta
set_track_meta(False)

_python_include = sysconfig.get_path("include") # for torch.compile/triton
if _python_include and os.path.isfile(os.path.join(_python_include, "Python.h")):
    os.environ["CPATH"] = _python_include + os.pathsep + os.environ.get("CPATH", "")
else:
    _fallback = f"/opt/python/{sysconfig.get_python_version()}.4/include/python{sysconfig.get_python_version()}"
    if os.path.isfile(os.path.join(_fallback, "Python.h")):
        os.environ["CPATH"] = _fallback + os.pathsep + os.environ.get("CPATH", "")

import torch
import wandb
from print_utils import print_config_summary, print_param_group_modules, tensor_stats_dict
from monai.losses import PerceptualLoss
from monai.transforms import (
    CenterSpatialCrop,
    Compose,
    Resize,
    ScaleIntensity,
    ScaleIntensityRange,
    SpatialPad,
)
from torch.utils.data import DataLoader, DistributedSampler
from data import collate_fn_skip_none

from basics import get_autocast_ctx, resolve_autocast_dtype, get_git_info
from checkpointing import load_checkpoint, save_checkpoint
from config import load_config
from dc_gen.ae_model_zoo import DCAE_HF
from evaluation import evaluate
from multigpu import (
    aggregate_metrics, cleanup, init_distributed, is_main_process,
    rank0_print, barrier, get_rank, get_world_size, wrap_ddp,
    main_process_first,
)
from registry import dataset_registry, loss_registry
from viz import Visualize
from gan_loss import LocalPatchGAN

_shutdown_requested = False


class WandBPrintCapture:
    """Captures print statements and logs them to wandb."""
    def __init__(self, log_metrics=False):
        self.log_metrics = log_metrics
        self.buffer = []
        self.original_stdout = sys.stdout
        self.current_epoch = 0
        self.current_step = 0

    def set_epoch(self, epoch):
        """Update current epoch."""
        self.current_epoch = epoch

    def set_step(self, step):
        """Update current step."""
        self.current_step = step

    def write(self, message):
        self.original_stdout.write(message)
        if message.strip() and self.log_metrics:
            self.buffer.append(message.strip())
            # Log to wandb in batches
            if len(self.buffer) >= 10:
                self.flush_to_wandb()

    def flush(self):
        self.original_stdout.flush()
        if self.buffer and self.log_metrics:
            self.flush_to_wandb()

    def flush_to_wandb(self):
        if self.buffer and wandb.run is not None:
            log_text = "\n".join(self.buffer)
            try:
                wandb.log({
                    "logs/print": wandb.Html(f"<pre>{log_text}</pre>"),
                    "epoch": self.current_epoch,
                }, commit=False)
            except Exception:
                pass
            self.buffer = []

    def isatty(self):
        return self.original_stdout.isatty()

def _shutdown_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    rank0_print(f"\n[{os.getpid()}] Shutdown signal received. Finishing current batch...")


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


class ClampIntensity:
    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value
        self._pending_ranges: list[tuple[float, float]] = []

    def __call__(self, tensor):
        self._pending_ranges.append((float(tensor.min().item()), float(tensor.max().item())))
        return torch.clamp(tensor, min=self.min_value, max=self.max_value)

    def pop_batch_ranges(self, batch_size: int) -> dict[str, object] | None:
        if batch_size <= 0 or len(self._pending_ranges) < batch_size:
            return None

        sample_ranges = self._pending_ranges[:batch_size]
        del self._pending_ranges[:batch_size]
        mins = [sample_min for sample_min, _ in sample_ranges]
        maxs = [sample_max for _, sample_max in sample_ranges]
        return {
            "batch_min": min(mins),
            "batch_max": max(maxs),
            "sample_ranges": sample_ranges,
        }


def finite_difference_loss(
    recon: torch.Tensor, target: torch.Tensor, dims: tuple[int, ...]
) -> torch.Tensor:
    losses = []
    for dim in dims:
        recon_diff = torch.diff(recon, dim=dim)
        target_diff = torch.diff(target, dim=dim)
        losses.append(torch.nn.functional.l1_loss(recon_diff, target_diff))
    return torch.stack(losses).mean()


def build_pipeline(cfg):
    output_min, output_max = cfg.pipeline.normalize_output_range
    transforms = []
    clip_transform = None

    if cfg.pipeline.clip_input_range: transforms.append(ClampIntensity(*cfg.pipeline.clip_input_range))
    transforms.append(ScaleIntensityRange(*cfg.pipeline.normalize_input_range, b_min=output_min, b_max=output_max, clip=True))

    if cfg.dims == "2d":
        spatial_size = tuple(cfg.pipeline.resize_hw)
        transforms.append(Resize(spatial_size=spatial_size))
    elif cfg.dims == "3d":
        n_slices = cfg.pipeline.n_slices
        transforms.append(SpatialPad(spatial_size=(n_slices, -1, -1), value=-1000))
        transforms.append(CenterSpatialCrop(roi_size=(n_slices, -1, -1)))

        spatial_size = (-1, *cfg.pipeline.resize_hw)
        transforms.append(Resize(spatial_size=spatial_size))

    return Compose(transforms), clip_transform

def initialize_gan(cfg, device):
    gan_module = None
    discriminator_optimizer = None
    gan_module = LocalPatchGAN(
        in_channels=1,
        patch_size=tuple(cfg.objective.gan_patch_size),
        ndf=cfg.objective.gan_ndf,
        loss_type=cfg.objective.gan_loss_type,
    ).to(device)
    disc_lr = cfg.hparams.discriminator_learning_rate or (cfg.hparams.learning_rate * 0.1)
    discriminator_optimizer = torch.optim.Adam(
        gan_module.discriminator.parameters(),
        lr=disc_lr,
        betas=(0.5, 0.999),
    )
    rank0_print(f"[GAN] Initialized patch discriminator on {device}")
    return gan_module, discriminator_optimizer


def main_worker(rank: int, world_size: int, cfg):
    global _shutdown_requested

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    use_cuda = torch.cuda.is_available()
    device = torch.device("mps" if torch.backends.mps.is_available() and not use_cuda else f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(rank)
    if use_cuda and world_size > 1:
        init_distributed(rank, world_size)

    num_workers = cfg.training.num_workers
    if use_cuda and world_size > 1: num_workers = max(1, num_workers // world_size)

    pipeline, clip_transform = build_pipeline(cfg)
    dataset_cls = dataset_registry[cfg.dataset.name]
    rank0_print("[setup] Building datasets (rank 0 first to populate index cache)...")
    with main_process_first():
        dataset = dataset_cls(
            nifti_dir=cfg.paths.nifti_dir,
            group_names=cfg.dataset.group_names,
            dims=cfg.dims,
            n_slices=cfg.pipeline.n_slices,
            transform=pipeline,
        )

        val_dataset = dataset_cls(
            nifti_dir=cfg.paths.nifti_val_dir,
            group_names=cfg.dataset.group_names,
            dims=cfg.dims,
            n_slices=cfg.pipeline.n_slices,
            transform=pipeline,
        )
    rank0_print(f"[setup] Datasets ready. Train: {len(dataset)}, Val: {len(val_dataset)}")

    if use_cuda and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=cfg.training.shuffle_data,
        )
    else:
        sampler = None

    mp_context = "fork" if use_cuda and world_size > 1 else None

    loader = DataLoader(
         dataset,
         batch_size=cfg.training.batch_size,
         shuffle=(sampler is None and cfg.training.shuffle_data),
         pin_memory=cfg.training.pin_memory,
         num_workers=num_workers,
         prefetch_factor=cfg.training.prefetch_factor,
         sampler=sampler,
         persistent_workers=num_workers > 0,
         multiprocessing_context=mp_context,
     )

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_cuda and world_size > 1 else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        pin_memory=cfg.training.pin_memory,
        num_workers=num_workers,
        prefetch_factor=cfg.training.prefetch_factor,
        sampler=val_sampler,
        collate_fn=collate_fn_skip_none,
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_context,
    )

    rank0_print("[setup] Loading model...")
    dtype = getattr(torch, cfg.training.dtype)
    model = DCAE_HF(model_name=cfg.model.name).to(dtype=dtype, device=device)
    param_groups = configure_trainable_params(model, cfg.training.trainable_ae_params)
    print_param_group_modules(model, param_groups)

    if getattr(cfg.model, "compile", False):
        rank0_print("[setup] Compiling model with torch.compile...")
        try: model = torch.compile(model)
        except Exception as e: rank0_print(f"[setup] torch.compile failed: {e}")
    if use_cuda and world_size > 1:
        model = wrap_ddp(model)
    model.train()
    rank0_print("[setup] Model ready.")

    perceptual_loss_fn = None
    if cfg.objective.perceptual_weight > 0:
        if cfg.dims == "3d":
            perceptual_loss_fn = PerceptualLoss(spatial_dims=3, network_type="vgg", is_fake_3d=True).to(device=device)
        elif cfg.dims == "2d":
            perceptual_loss_fn = PerceptualLoss(spatial_dims=2, network_type="vgg").to(device=device)
        perceptual_loss_fn.eval()
    perceptual_weight = cfg.objective.perceptual_weight
    detail_weight = cfg.objective.detail_weight

    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    viz = Visualize(viz_type=cfg.dims)

    # Initialize GAN if enabled
    if cfg.objective.gan_enable and cfg.objective.gan_weight > 0: 
        gan_module, discriminator_optimizer = initialize_gan(cfg, device)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.hparams.learning_rate,
        weight_decay=cfg.hparams.weight_decay,
    )
    global_step, wandb_run_id = load_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, device)

    loss_history, psnr_history, ssim_history = [], [], []
    checkpoint_queue = deque()
    log_file = os.path.join(cfg.paths.save_dir, "training_log.txt")
    num_epochs = cfg.training.num_epochs
    log_metrics = cfg.logging.wandb

    if log_metrics:
        git_info = get_git_info()
        cfg_dict = vars(cfg)
        cfg_dict.update(git_info)

        wandb.init(project="ct_ae", config=cfg_dict)

        print_capture = WandBPrintCapture(log_metrics=True) # todo
        sys.stdout = print_capture

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger() # todo: is this used?
        wandb_handler = logging.StreamHandler()
        wandb_handler.setLevel(logging.ERROR)
        logger.addHandler(wandb_handler)
    else: print_capture = None
    print_config_summary(cfg, device)

    if cfg.logging.print_model_arch: print(model)

    import time as _tm
    rank0_print("[setup] Warming up dataloader workers...")
    _warmup_t = _tm.time()
    _train_iter = iter(loader)
    _warmup_batch = next(_train_iter)
    rank0_print(f"[setup] Train loader warm: {_tm.time()-_warmup_t:.2f}s")
    _warmup_t = _tm.time()
    _val_iter = iter(val_loader)
    _warmup_batch_val = next(_val_iter)
    rank0_print(f"[setup] Val loader warm: {_tm.time()-_warmup_t:.2f}s")
    del _train_iter, _val_iter, _warmup_batch, _warmup_batch_val

    rank0_print(f"[setup] Starting training loop (global_step={global_step})...")
    for epoch in range(num_epochs):
        if _shutdown_requested:
            rank0_print(f"\n[{rank}] Shutdown requested, exiting training loop")
            break

        if log_metrics and print_capture is not None: print_capture.set_epoch(epoch)

        if sampler: sampler.set_epoch(epoch)

        rank0_print(f"epoch: {epoch} out of {num_epochs}")
        train_size = len(dataset)
        val_size = len(val_dataset)
        rank0_print(f"train set: {train_size}, val set: {val_size}")

        for i, batch in enumerate(loader):
            if _shutdown_requested:
                rank0_print(f"\n[{rank}] Shutdown requested, finishing epoch")
                break
            if i == 0:
                rank0_print(f"  batch shape: {batch.shape}")
                import time as _tm
            save_diff = (cfg.logging.save_volumes and global_step % cfg.logging.viz_every == 0)
            save_ckpt = global_step % cfg.training.checkpoint_every == 0
            do_validation = (cfg.logging.validate_every is not None and global_step > 0 and global_step % cfg.logging.validate_every == 0)
            raw_batch_range = None
            if clip_transform is not None: raw_batch_range = clip_transform.pop_batch_ranges(len(batch))

            if i < 2: _t0 = _tm.time()
            batch = batch.to(dtype=dtype, device=device, non_blocking=True)
            if hasattr(batch, "as_tensor"):
                batch = batch.as_tensor()
            if i < 2: rank0_print(f"  Iter {i}: to_device {_tm.time()-_t0:.2f}s")

            if i < 2: _t0 = _tm.time()
            with get_autocast_ctx(cfg, device):
                recon = model(batch)
            if i < 2: rank0_print(f"  Iter {i}: forward {_tm.time()-_t0:.2f}s")

            if i < 2: _t0 = _tm.time()
            recon_loss = loss_registry[cfg.objective.loss_fn](recon, batch)
            if perceptual_loss_fn is not None:
                perc_loss = perceptual_loss_fn(recon.float(), batch.float())
            else:
                perc_loss = recon_loss.new_zeros(())

            detail_dims = (-2, -1)
            detail_loss = finite_difference_loss(recon.float(), batch.float(), dims=detail_dims)

            gan_loss = recon_loss.new_zeros(())
            if cfg.objective.gan_enable and gan_module is not None:
                real, fake = batch.float(), recon.float()
                if global_step % (cfg.objective.gan_discriminator_steps + 1) == 0:
                    gan_loss = gan_module.compute_generator_loss(real, fake)
                else:
                    discriminator_optimizer.zero_grad()
                    gan_module.compute_discriminator_loss(real, fake.detach()).backward()
                    discriminator_optimizer.step()

            loss = (recon_loss + perceptual_weight * perc_loss + detail_weight * detail_loss)

            if cfg.objective.gan_enable: loss = loss + cfg.objective.gan_weight * gan_loss
            if cfg.objective.ssim_weight > 0: loss = loss + cfg.objective.ssim_weight * loss_registry["ms_ssim"](recon, batch)

            if i < 2: rank0_print(f"  Iter {i}: losses {_tm.time()-_t0:.2f}s")
            if i < 2: _t0 = _tm.time()
            # --- backward + step (all ranks must participate for DDP sync)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()
            if i < 2: rank0_print(f"  Iter {i}: backward {_tm.time()-_t0:.2f}s")

            if is_main_process():
                with torch.no_grad(): loss_value, psnr_value, ssim_value, slice_psnr_value, slice_ssim_value = evaluate(recon.detach(), batch.detach(), loss.detach())
                for h, v in zip([loss_history, psnr_history, ssim_history], [loss_value, psnr_value, ssim_value]): h.append(v)
                metrics_str = f"loss={loss_value:.6f}, PSNR={psnr_value:.6f}, SSIM={ssim_value:.6f}, slice_PSNR={slice_psnr_value:.6f}, slice_SSIM={slice_ssim_value:.6f}"
                try: open(log_file, "a").write(f"iter {global_step}: {metrics_str}\n")
                except Exception as e: print(f"[WARNING] Failed to write to log file {log_file}: {e}")

                range_text = wandb_range_text = ""
                if raw_batch_range:
                    bmin, bmax = raw_batch_range["batch_min"], raw_batch_range["batch_max"]
                    sample_text = ", ".join(f"s{i}: [{lo:.3f}, {hi:.3f}]" for i, (lo, hi) in enumerate(raw_batch_range["sample_ranges"]))
                    range_text = f", raw_min={bmin:.3f}, raw_max={bmax:.3f}"
                    wandb_range_text = f"step {global_step}: raw batch range before clip/normalize [{bmin:.3f}, {bmax:.3f}]" + (f"; per-sample {sample_text}" if sample_text else "")
                print(f"Iter {global_step}: {metrics_str}{range_text}")
            if log_metrics:
                wandb.log({
                    "loss/total": loss.item(), "loss/recon": recon_loss.item(),
                    "loss/perceptual": perc_loss.item(), "loss/detail": detail_loss.item(),
                    "loss/gan": gan_loss.item(), "metrics/psnr": psnr_value,
                    "metrics/ssim": ssim_value, "metrics/slice_psnr": slice_psnr_value,
                    "metrics/slice_ssim": slice_ssim_value, "grad_norm": grad_norm, "epoch": epoch,
                    **tensor_stats_dict("batch", batch), **tensor_stats_dict("recon", recon),
                }, step=global_step)
                if wandb.run is not None and wandb_range_text is not None: wandb.run.summary["raw_input_range_text"] = wandb_range_text

            if save_diff: barrier(); viz.save(batch, recon, cfg.paths.save_dir, global_step) if is_main_process() else None; barrier()

            if do_validation:
                _tv = _tm.time()
                barrier()
                rank0_print(f"\n[val] Starting validation at step {global_step}... (barrier {_tm.time()-_tv:.2f}s)")
                sys.stdout.flush()
                val_psnrs, val_ssims = [], []
                val_slice_psnrs, val_slice_ssims = [], []
                val_max = cfg.logging.val_max_batches
                total = min(len(val_loader), val_max) if val_max else len(val_loader)

                raw_model = model.module if hasattr(model, "module") else model

                with torch.inference_mode():
                    raw_model.eval()
                    _tv_loop = _tm.time()
                    for vi, val_batch in enumerate(val_loader):
                        if vi == 0: rank0_print(f"  [val] First batch: {_tm.time()-_tv_loop:.2f}s")
                        if val_batch is None: continue
                        if val_max and vi >= val_max: break
                        val_batch = val_batch.to(dtype=dtype, device=device, non_blocking=True)
                        with get_autocast_ctx(cfg, device):
                            val_recon = raw_model(val_batch)
                        _, vp, vs, vsp, vss = evaluate(val_recon, val_batch, None)
                        val_psnrs.append(vp); val_ssims.append(vs)
                        val_slice_psnrs.append(vsp); val_slice_ssims.append(vss)
                        if is_main_process(): print(f"\rValidation [{vi + 1}/{total}]", end="", flush=True)
                    raw_model.train()

                    def _mean(vals): return torch.tensor(vals).mean().item() if vals else 0.0
                    val_metrics = aggregate_metrics({
                        "val_psnr": _mean(val_psnrs), "val_ssim": _mean(val_ssims),
                        "val_slice_psnr": _mean(val_slice_psnrs), "val_slice_ssim": _mean(val_slice_ssims),
                    })

                if is_main_process():
                    vm = val_metrics
                    print(f"\nValidation PSNR: {vm['val_psnr']:.6f}, SSIM: {vm['val_ssim']:.6f}, "
                          f"Slice PSNR: {vm['val_slice_psnr']:.6f}, Slice SSIM: {vm['val_slice_ssim']:.6f}")
                    if log_metrics:
                        wandb.log({
                            "val_metrics/psnr": vm["val_psnr"], "val_metrics/ssim": vm["val_ssim"],
                            "val_metrics/slice_psnr": vm["val_slice_psnr"], "val_metrics/slice_ssim": vm["val_slice_ssim"],
                            "epoch": epoch,
                        }, step=global_step)
                rank0_print(f"[val] Validation complete at step {global_step}")
            if save_ckpt:
                barrier()
                if is_main_process(): save_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, global_step, checkpoint_queue, cfg.training.max_checkpoints)
                barrier()
            global_step += 1

        rank0_print(f"\n[{rank}] Finalizing training...")
        if log_metrics and print_capture is not None:
            print_capture.flush()
            if wandb.run is not None: wandb.finish()
        if use_cuda and world_size > 1: barrier(); cleanup()
        rank0_print(f"[{rank}] Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None, help="Name of experiment config in configs/experiments/ (e.g. dc-ae-f32c32-in-1.0_3d-shallow)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file to override defaults")
    args = parser.parse_args()
    cfg = load_config(yaml_path=args.config, experiment=args.experiment)

    use_cuda = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 1

    if not (use_cuda and world_size > 1): main_worker(0, 1, cfg)
    else:
        import socket, torch.multiprocessing as mp
        if 'MASTER_PORT' not in os.environ:
            for port in [8972, 8973, 8974, 8975]:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('', port)); os.environ['MASTER_PORT'] = str(port)
                        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
                        os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
                        break
                    except OSError: continue
            else:
                raise RuntimeError("No free port in assigned range [8972-8975]")
        mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True)
