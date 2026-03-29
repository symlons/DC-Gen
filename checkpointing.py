import os
import glob
from collections import deque
import torch
import wandb

def _get_base_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def save_checkpoint(cfg, model, optimizer, save_dir, it, checkpoint_queue: deque, max_checkpoints: int):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"checkpoint_iter{it}.pt")

    base_model = _get_base_model(model)
    checkpoint = {
        'global_step': it,
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    # Save wandb run ID if logging
    if wandb.run is not None:
        checkpoint['wandb_run_id'] = wandb.run.id

    try:
        torch.save(checkpoint, ckpt_path)
    except Exception as e:
        print(f"Failed to save checkpoint {ckpt_path}: {e}")
        return

    if getattr(cfg, "wandb", None) and getattr(cfg.wandb, "enabled", False):
        try:
            wandb.save(ckpt_path)
        except Exception as e:
            print(f"Failed to save checkpoint to WandB: {e}")

    checkpoint_queue.append(ckpt_path)
    while len(checkpoint_queue) > max_checkpoints:
        old_ckpt = checkpoint_queue.popleft()
        if os.path.isfile(old_ckpt):
            try:
                os.remove(old_ckpt)
            except Exception as e:
                print(f"Failed to delete old checkpoint {old_ckpt}: {e}")


def load_checkpoint(cfg, model, optimizer, checkpoint_dir, device):
    resume = getattr(cfg.training, "resume_from_checkpoint", False)
    if not resume:
        return 0, None

    ckpt_path = getattr(cfg.training, "resume_from_checkpoint_path", None)
    if not (ckpt_path and os.path.isfile(ckpt_path)):
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_iter*.pt")))
        ckpt_path = checkpoint_files[-1] if checkpoint_files else None

    if not ckpt_path:
        print("No checkpoints found, starting from scratch")
        return 0, None

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        base_model = _get_base_model(model)
        base_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        return 0, None

    global_step = ckpt.get("global_step", ckpt.get("iteration", 0))
    wandb_run_id = ckpt.get("wandb_run_id", None)
    print(f"Resumed from checkpoint {ckpt_path} at global_step {global_step}")
    if wandb_run_id:
        print(f"WandB run ID: {wandb_run_id}")
    return global_step, wandb_run_id
