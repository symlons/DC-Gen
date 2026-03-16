import argparse
from collections import deque
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from monai.transforms import Compose, ScaleIntensity, Resize
from monai.losses import PerceptualLoss

from registry import dataset_registry, loss_registry
from dc_gen.ae_model_zoo import DCAE_HF
from checkpointing import load_checkpoint, save_checkpoint
from evaluation import evaluate
from config import load_config
from multigpu import main_process_only, init_distributed, cleanup
from viz import Visualize
from basics import get_autocast_ctx
import wandb

def main_worker(rank: int, world_size: int, cfg):
    use_cuda = torch.cuda.is_available()
    device = torch.device("mps" if torch.backends.mps.is_available() and not use_cuda else f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda and world_size > 1: init_distributed(rank, world_size)

    pipeline = Compose([ScaleIntensity(minv=-1, maxv=1), Resize(spatial_size=cfg.pipeline.resize_hw)])
    dataset_cls = dataset_registry[cfg.dataset.name]
    dataset = dataset_cls(
        hdf_path=cfg.paths.hdf_path,
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

    dtype = getattr(torch, cfg.training.dtype)
    model = DCAE_HF(model_name=cfg.model.name).to(dtype=dtype, device=device)

    if getattr(cfg.model, "compile", False): model = torch.compile(model)
    if use_cuda and world_size > 1: model = wrap_ddp(model, device, rank, world_size)
    model.train()

    if cfg.dims == "3d": perceptual_loss_fn = PerceptualLoss(spatial_dims=3, network_type="vgg", is_fake_3d=True).to(device=device)
    elif cfg.dims == "2d": perceptual_loss_fn = PerceptualLoss( spatial_dims=2, network_type="vgg").to(device=device)
    perceptual_loss_fn.eval()
    perceptual_weight = cfg.objective.perceptual_weight

    viz = Visualize(viz_type=cfg.dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.hparams.learning_rate, weight_decay=cfg.hparams.weight_decay)
    global_step = load_checkpoint(cfg, model, optimizer, cfg.paths.checkpoint_dir, device)

    loss_history, psnr_history, ssim_history = [], [], []
    checkpoint_queue = deque()
    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    log_file = os.path.join(cfg.paths.save_dir, "training_log.txt")
    num_epochs = cfg.training.num_epochs
    log_metrics = cfg.logging.wandb
    if log_metrics: wandb.init(project="ct_retcon", config=vars(cfg))
    print("Compiling: ", cfg.model.compile)
    print(model)

    for epoch in range(num_epochs):
        if sampler: sampler.set_epoch(epoch)

        if rank == 0:
            print(f"epoch: {epoch} out of {num_epochs}")
            print(f"dataset size: {len(dataset)}")

        for batch in loader:
            # print(batch.shape)
            save_diff = cfg.logging.save_volumes and global_step % cfg.logging.viz_every == 0
            save_ckpt = global_step % cfg.training.checkpoint_every == 0

            batch = batch.to(dtype=dtype, device=device, non_blocking=True)

            with get_autocast_ctx(cfg, device): recon = model.decoder(model.encoder(batch))

            recon_loss = loss_registry[cfg.objective.loss_fn](recon, batch)
            perc_loss = perceptual_loss_fn(recon.float(), batch.float())
            loss = recon_loss + perceptual_weight * perc_loss

            with main_process_only():
                loss_value, psnr_value, ssim_value = evaluate(recon, batch, loss)
                for h, v in zip([loss_history, psnr_history, ssim_history], [loss_value, psnr_value, ssim_value]): h.append(v)

                try:
                    with open(log_file, "a") as f:
                        f.write(f"iter {global_step}: loss={loss.item():.6f}, " f"PSNR={psnr_value:.6f}, SSIM={ssim_value:.6f}\n")
                        f.flush()
                except Exception as e:
                    print(f"[WARNING] Failed to write to log file {log_file}: {e}")

                print(f"Iter {global_step}: loss={loss.item():.6f}, " f"PSNR={psnr_value:.6f}, SSIM={ssim_value:.6f}")
                if save_diff: viz.save(batch, recon, cfg.paths.save_dir, global_step)
                if log_metrics: wandb.log({"loss": loss.item(), "PSNR": psnr_value, "SSIM": ssim_value}, step=global_step)
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
