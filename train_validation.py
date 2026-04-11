from itertools import islice
import torch
import wandb
from ae_utils import compute_loss
from basics import get_autocast_ctx, move_batch
from evaluation import evaluate
from multigpu import aggregate_metrics, is_main_process, rank0_print


def run_validation(model, val_loader, device, dtype, cfg, *, forward_fn, decode_fn=None, loss_fn=None, global_step=0, log_metrics=False, shutdown_flag=None):
    if val_loader is None: return {}

    model.eval()
    accum, loss_total, count = {}, 0.0, 0
    max_batches = getattr(cfg.logging, "val_max_batches", None) or len(val_loader)
    total = min(len(val_loader), max_batches)

    with torch.inference_mode():
        for i, batch in enumerate(islice(val_loader, max_batches)):
            if shutdown_flag and shutdown_flag(): break
            if batch is None: continue
            if is_main_process(): print(f"\r[val] {i+1}/{total}", end="", flush=True)

            batch = move_batch(batch, device, dtype)
            with get_autocast_ctx(cfg, device): recon = forward_fn(model, batch)

            pixel_recon = decode_fn(recon) if decode_fn is not None else recon
            pixel_real  = decode_fn(batch)  if decode_fn is not None else batch

            for k, v in evaluate(pixel_recon, pixel_real).items(): accum[k] = accum.get(k, 0.0) + v
            if loss_fn is not None: loss_total += loss_fn(recon, batch)
            count += 1

    print()
    model.train()
    if count == 0: return {}

    result = aggregate_metrics({k: v / count for k, v in accum.items()})
    if loss_fn is not None: result["loss"] = aggregate_metrics({"loss": loss_total / count})["loss"]

    if is_main_process():
        rank0_print(f"[val] Validation complete at step {global_step}")
        for k, v in result.items(): rank0_print(f"val/{k}: {v:.4f}" if not torch.is_tensor(v) else f"val/{k}: {v.mean().item():.4f}")
        if log_metrics: wandb.log({ **{f"val/{k}": v if not torch.is_tensor(v) else v.mean().item() for k, v in result.items()}}, step=global_step)
    return result
