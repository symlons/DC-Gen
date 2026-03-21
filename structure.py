import argparse
from collections import deque
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from monai.transforms import Compose, Resize, ScaleIntensity, ScaleIntensityRange
from monai.losses import PerceptualLoss

from registry import dataset_registry, loss_registry
from dc_gen.ae_model_zoo import DCAE_HF
from checkpointing import load_checkpoint, save_checkpoint
from evaluation import evaluate
from config import load_config
from multigpu import main_process_only, init_distributed, cleanup
from viz import Visualize
from basics import get_autocast_ctx, resolve_autocast_dtype
import wandb
import re

def configure_trainable_params(model, trainable_ae_params):
    if trainable_ae_params is None:
        return [{"params": list(model.parameters())}]

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    named_params = list(base_model.named_parameters())

    used = set()
    groups = []
    total_matched = 0

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

    if total_matched == 0:
        print("[WARNING] No parameters matched ANY pattern — nothing will be trained.")

    train_params = set(p for g in groups for p in g["params"])

    for p in model.parameters():
        p.requires_grad = False

    for p in train_params:
        p.requires_grad = True

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


def print_config_summary(cfg, device):
    resolved_autocast_dtype = resolve_autocast_dtype(cfg, device)
    print("Config summary")
    print(f"  Experiment   : {cfg.experiment.name}")
    print(f"  Dims         : {cfg.dims}")
    print(f"  Model        : {cfg.model.name}")
    print(f"  Compile      : {cfg.model.compile}")
    print(f"  Dataset      : {cfg.dataset.name}")
    print(f"  Groups       : {cfg.dataset.group_names}")
    print(f"  HDF Path     : {cfg.paths.hdf_path}")
    print(f"  Save Dir     : {cfg.paths.save_dir}")
    print(f"  Checkpoints  : {cfg.paths.checkpoint_dir}")
    print(f"  Resize H/W   : {cfg.pipeline.resize_hw}")
    print(f"  Resize Depth : {cfg.pipeline.resize_depth}")
    print(f"  N Slices     : {cfg.pipeline.n_slices}")
    print(f"  Clip In      : {cfg.pipeline.clip_input_range}")
    print(f"  Norm Mode    : {cfg.pipeline.normalize_mode}")
    print(f"  Norm Out     : {cfg.pipeline.normalize_output_range}")
    print(f"  Norm In      : {cfg.pipeline.normalize_input_range}")
    print(f"  Epochs       : {cfg.training.num_epochs}")
    print(f"  Batch Size   : {cfg.training.batch_size}")
    print(f"  Num Workers  : {cfg.training.num_workers}")
    print(f"  DType        : {cfg.training.dtype}")
    print(f"  Autocast     : {cfg.training.use_autocast}")
    print(f"  AMP DType    : {cfg.training.autocast_dtype} -> {resolved_autocast_dtype}")
    print(f"  LR           : {cfg.hparams.learning_rate}")
    print(f"  Weight Decay : {cfg.hparams.weight_decay}")
    print(f"  Loss         : {cfg.objective.loss_fn}")
    print(f"  Perc Weight  : {cfg.objective.perceptual_weight}")
    print(f"  Detail Weight: {cfg.objective.detail_weight}")
    print(f"  WandB        : {cfg.logging.wandb}")
    print(f"  Viz Every    : {cfg.logging.viz_every}")
    print(f"  Validate Every: {cfg.logging.validate_every}")
    print()

def print_param_group_modules(model, param_groups):
    named_params = dict(model.named_parameters())

    id_to_module = {}
    for name, param in named_params.items():
        module_name = ".".join(name.split(".")[:-1])
        id_to_module[id(param)] = module_name

    total_params = sum(p.numel() for p in model.parameters())
    train_params = set(p for g in param_groups for p in g["params"])
    trainable_params = sum(p.numel() for p in train_params)

    if trainable_params == 0:
        raise ValueError("No trainable parameters selected")

    pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    print("Parameter summary:")
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")
    print(f"  Trainable %      : {pct:.4f}%")

    print("\nParameter groups (module structure):")

    modules_dict = dict(model.named_modules())

    for i, group in enumerate(param_groups):
        params = group["params"]
        num_elements = sum(p.numel() for p in params)
        group_pct = 100.0 * num_elements / total_params if total_params > 0 else 0.0

        print(f"\nGroup {i}: {num_elements:,} elements ({group_pct:.4f}%)")

        printed_modules = set()

        for p in params:
            module_name = id_to_module.get(id(p))
            if module_name is None:
                continue

            parts = module_name.split(".")
            key = ".".join(parts[:2])

            if key in printed_modules:
                continue

            printed_modules.add(key)

            try:
                submodule = modules_dict[key]
                print(f"\n--- {key} ---")
                print(submodule)
            except KeyError:
                print(f"[WARNING] Module not found: {key}")

def tensor_stats_dict(name: str, tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.detach().float()
    quantiles = torch.quantile(tensor.flatten(), torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=tensor.device))
    return {
        f"{name}/mean": tensor.mean().item(),
        f"{name}/std": tensor.std(unbiased=False).item(),
        f"{name}/min": tensor.min().item(),
        f"{name}/max": tensor.max().item(),
        f"{name}/q01": quantiles[0].item(),
        f"{name}/q05": quantiles[1].item(),
        f"{name}/median": quantiles[2].item(),
        f"{name}/q95": quantiles[3].item(),
        f"{name}/q99": quantiles[4].item(),
    }


def finite_difference_loss(recon: torch.Tensor, target: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
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

    if cfg.pipeline.clip_input_range is not None:
        clip_min, clip_max = cfg.pipeline.clip_input_range
        clip_transform = ClampIntensity(min_value=clip_min, max_value=clip_max)
        transforms.append(clip_transform)

    if cfg.pipeline.normalize_mode == "fixed":
        input_min, input_max = cfg.pipeline.normalize_input_range
        normalize = ScaleIntensityRange(
            a_min=input_min,
            a_max=input_max,
            b_min=output_min,
            b_max=output_max,
            clip=True,
        )
    else:
        normalize = ScaleIntensity(minv=output_min, maxv=output_max)

    transforms.append(normalize)

    if cfg.dims == "2d":
        spatial_size = tuple(cfg.pipeline.resize_hw)
    else:
        resize_depth = cfg.pipeline.resize_depth if cfg.pipeline.resize_depth is not None else cfg.pipeline.n_slices
        spatial_size = (resize_depth, *cfg.pipeline.resize_hw)

    transforms.append(Resize(spatial_size=spatial_size))
    return Compose(transforms), clip_transform

def main_worker(rank: int, world_size: int, cfg):
    use_cuda = torch.cuda.is_available()
    device = torch.device("mps" if torch.backends.mps.is_available() and not use_cuda else f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda and world_size > 1: init_distributed(rank, world_size)

    pipeline, clip_transform = build_pipeline(cfg)
    dataset_cls = dataset_registry[cfg.dataset.name]
    dataset = dataset_cls(
        nifti_dir=cfg.paths.nifti_dir,
        group_names=cfg.dataset.group_names,
        dims=cfg.dims,
        n_slices=cfg.pipeline.n_slices,
        transform=pipeline
    )

    val_dataset = dataset_cls(
        nifti_dir=cfg.paths.nifti_val_dir,
        group_names=cfg.dataset.group_names,
        dims=cfg.dims,
        n_slices=cfg.pipeline.n_slices,
        transform=pipeline
    )

    if use_cuda and world_size > 1: sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=cfg.training.shuffle_data)
    else: sampler = None

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(sampler is None and cfg.training.shuffle_data),
        pin_memory=cfg.training.pin_memory,
        num_workers=cfg.training.num_workers,
        prefetch_factor=cfg.training.prefetch_factor,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        pin_memory=cfg.training.pin_memory,
        num_workers=cfg.training.num_workers,
        prefetch_factor=cfg.training.prefetch_factor,
    )

    dtype = getattr(torch, cfg.training.dtype)
    model = DCAE_HF(model_name=cfg.model.name).to(dtype=dtype, device=device)
    trainable_ae_params = [ ["encoder.project_out.*", "decoder.project_in.*"] ]
    # trainable_ae_params = None
    param_groups = configure_trainable_params(model, trainable_ae_params)
    print_param_group_modules(model, param_groups)

    if getattr(cfg.model, "compile", False): model = torch.compile(model)
    if use_cuda and world_size > 1: model = wrap_ddp(model, device, rank, world_size)
    model.train()

    perceptual_loss_fn = None
    if cfg.objective.perceptual_weight > 0:
        if cfg.dims == "3d":
            perceptual_loss_fn = PerceptualLoss(spatial_dims=3, network_type="vgg", is_fake_3d=True).to(device=device)
        elif cfg.dims == "2d":
            perceptual_loss_fn = PerceptualLoss(spatial_dims=2, network_type="vgg").to(device=device)
        perceptual_loss_fn.eval()
    perceptual_weight = cfg.objective.perceptual_weight
    detail_weight = cfg.objective.detail_weight

    viz = Visualize(viz_type=cfg.dims)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.hparams.learning_rate, weight_decay=cfg.hparams.weight_decay)
    global_step = load_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, device)

    loss_history, psnr_history, ssim_history = [], [], []
    checkpoint_queue = deque()
    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    log_file = os.path.join(cfg.paths.save_dir, "training_log.txt")
    num_epochs = cfg.training.num_epochs
    log_metrics = cfg.logging.wandb
    if log_metrics: wandb.init(project="ct_retcon", config=vars(cfg))
    print_config_summary(cfg, device)
    print(model)

    for epoch in range(num_epochs):
        if sampler: sampler.set_epoch(epoch)

        if rank == 0:
            print(f"epoch: {epoch} out of {num_epochs}")
            print(f"dataset size: {len(dataset)}")

        for i, batch in enumerate(loader):
            if i == 0: print(f"  batch shape: {batch.shape}")
            save_diff = cfg.logging.save_volumes and global_step % cfg.logging.viz_every == 0
            save_ckpt = global_step % cfg.training.checkpoint_every == 0
            do_validation = cfg.logging.validate_every is not None and global_step % cfg.logging.validate_every == 0
            raw_batch_range = None
            if clip_transform is not None:
                raw_batch_range = clip_transform.pop_batch_ranges(len(batch))

            batch = batch.to(dtype=dtype, device=device, non_blocking=True)

            with get_autocast_ctx(cfg, device): recon = model.decoder(model.encoder(batch))

            recon_loss = loss_registry[cfg.objective.loss_fn](recon, batch)
            if perceptual_loss_fn is not None:
                perc_loss = perceptual_loss_fn(recon.float(), batch.float())
            else:
                perc_loss = recon_loss.new_zeros(())

            detail_dims = (-2, -1)
            detail_loss = finite_difference_loss(recon.float(), batch.float(), dims=detail_dims)
            loss = recon_loss + perceptual_weight * perc_loss + detail_weight * detail_loss
            # --- Metircs
            with main_process_only():
                with torch.no_grad():
                    loss_value, psnr_value, ssim_value, slice_psnr_value, slice_ssim_value = evaluate(
                        recon.detach(), batch.detach(), loss.detach()
                    )
                for h, v in zip([loss_history, psnr_history, ssim_history], [loss_value, psnr_value, ssim_value]): h.append(v)

                try:
                    with open(log_file, "a") as f:
                        f.write(
                            f"iter {global_step}: loss={loss.item():.6f}, "
                            f"PSNR={psnr_value:.6f}, SSIM={ssim_value:.6f}, "
                            f"slice_PSNR={slice_psnr_value:.6f}, slice_SSIM={slice_ssim_value:.6f}\n"
                        )
                        f.flush()
                except Exception as e:
                    print(f"[WARNING] Failed to write to log file {log_file}: {e}")

                range_text = ""
                wandb_range_text = None
                if raw_batch_range is not None:
                    range_text = (
                        f", raw_min={raw_batch_range['batch_min']:.3f}, "
                        f"raw_max={raw_batch_range['batch_max']:.3f}"
                    )
                    sample_text = ", ".join(
                        f"s{i}: [{sample_min:.3f}, {sample_max:.3f}]"
                        for i, (sample_min, sample_max) in enumerate(raw_batch_range["sample_ranges"])
                    )
                    wandb_range_text = (
                        f"step {global_step}: raw batch range before clip/normalize "
                        f"[{raw_batch_range['batch_min']:.3f}, {raw_batch_range['batch_max']:.3f}]"
                    )
                    if sample_text:
                        wandb_range_text = f"{wandb_range_text}; per-sample {sample_text}"

                print(
                    f"Iter {global_step}: loss={loss.item():.6f}, "
                    f"PSNR={psnr_value:.6f}, SSIM={ssim_value:.6f}, "
                    f"slice_PSNR={slice_psnr_value:.6f}, slice_SSIM={slice_ssim_value:.6f}"
                    f"{range_text}"
                )
                if save_diff: viz.save(batch, recon, cfg.paths.save_dir, global_step)
                if log_metrics:
                    wandb_metrics = {
                        "loss/total": loss.item(),
                        "loss/recon": recon_loss.item(),
                        "loss/perceptual": perc_loss.item(),
                        "loss/detail": detail_loss.item(),
                        "metrics/psnr": psnr_value,
                        "metrics/ssim": ssim_value,
                        "metrics/slice_psnr": slice_psnr_value,
                        "metrics/slice_ssim": slice_ssim_value,
                    }
                    wandb_metrics.update(tensor_stats_dict("batch", batch))
                    wandb_metrics.update(tensor_stats_dict("recon", recon))
                    wandb.log(wandb_metrics, step=global_step)
                    if wandb.run is not None and wandb_range_text is not None:
                        wandb.run.summary["raw_input_range_text"] = wandb_range_text
                if do_validation:
                    print("Running validation...")
                    val_losses, val_psnrs, val_ssims, val_slice_psnrs, val_slice_ssims = [], [], [], [], []
                    total = len(val_loader)

                    with torch.inference_mode():
                        model.eval()
                        for i, batch in enumerate(val_loader):
                            batch = batch.to(dtype=dtype, device=device, non_blocking=True)

                            with get_autocast_ctx(cfg, device): recon = model.decoder(model.encoder(batch))

                            loss_value, psnr_value, ssim_value, slice_psnr_value, slice_ssim_value = evaluate(recon, batch, loss)

                            val_losses.append(loss_value)
                            val_psnrs.append(psnr_value)
                            val_ssims.append(ssim_value)
                            val_slice_psnrs.append(slice_psnr_value)
                            val_slice_ssims.append(slice_ssim_value)

                            print(f"\rValidation [{i+1}/{total}]", end="")
                    print()

                    val_loss = torch.tensor(val_losses).mean().item()
                    val_psnr = torch.tensor(val_psnrs).mean().item()
                    val_ssim = torch.tensor(val_ssims).mean().item()
                    val_slice_psnr = torch.tensor(val_slice_psnrs).mean().item()
                    val_slice_ssim = torch.tensor(val_slice_ssims).mean().item()

                    print(f"Validation Loss: {val_loss:.6f}, PSNR: {val_psnr:.6f}, SSIM: {val_ssim:.6f}, Slice PSNR: {val_slice_psnr:.6f}, Slice SSIM: {val_slice_ssim:.6f}")
                    
                    if log_metrics:
                        wandb.log({
                            "val_loss/total": val_loss,
                            "val_metrics/psnr": val_psnr,
                            "val_metrics/ssim": val_ssim,
                            "val_metrics/slice_psnr": val_slice_psnr,
                            "val_metrics/slice_ssim": val_slice_ssim,
                            }, step=global_step)
                if save_ckpt: save_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, global_step, checkpoint_queue, cfg.training.max_checkpoints)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            global_step += 1

    if use_cuda and world_size > 1:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file to override defaults")
    args = parser.parse_args()

    cfg = load_config(args.config)

    use_cuda = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 1

    if use_cuda and world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, cfg)