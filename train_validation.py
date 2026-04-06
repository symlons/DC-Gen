import torch
from evaluation import evaluate
from itertools import islice


def compute_step_metrics(outputs, images, autoencoder=None, device=None):
    loss = outputs["loss"].detach()
    x1_pred = outputs["x1_pred"].detach()
    if autoencoder is not None and device is not None:
        with torch.no_grad():
            recon = autoencoder.decode(x1_pred.to(device)).to(x1_pred.device)
            images_decoded = autoencoder.decode(images.to(device)).to(images.device)
    loss_value, psnr, ssim, slice_psnr, slice_ssim = evaluate(recon, images_decoded, loss)
    return (loss_value, psnr, ssim, slice_psnr, slice_ssim)


def run_validation(model, val_loader, objective, cfg, device, model_dtype, autoencoder):
    if val_loader is None: return {}
    model.eval()
    metrics_accumulator = {}
    num_batches = 0

    print("Running validation...")
    max_batches = cfg.logging.max_val_batches or len(val_loader)
    total = min(len(val_loader), max_batches)

    with torch.no_grad():
        for i, batch in enumerate(islice(val_loader, max_batches)):
            print(f"\rval: {i+1}/{total} | left: {total-(i+1)}", end="", flush=True)

            batch = {k: v.to(device, dtype=model_dtype) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = objective.compute_loss(model, batch["image"])
            metrics = dict(zip(("loss", "psnr", "ssim", "slice_psnr", "slice_ssim"), compute_step_metrics(outputs, batch["image"], autoencoder, device)))
            for k, v in metrics.items(): metrics_accumulator[k] = metrics_accumulator.get(k, 0.0) + v
            num_batches += 1
    print()
    model.train()

    if num_batches == 0: return {}
    return {k: v / num_batches for k, v in metrics_accumulator.items()}
