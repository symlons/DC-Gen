import glob
import os
from collections import deque

import torch
import wandb


def _get_base_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def save_checkpoint(cfg, model, optimizer, save_dir, it, checkpoint_queue: deque, max_checkpoints: int, ema_model=None):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"checkpoint_iter{it}.pt")

    base_model = _get_base_model(model)
    checkpoint = {
        "global_step": it,
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if ema_model is not None: checkpoint["ema_state_dict"] = ema_model.state_dict()
    if wandb.run is not None: checkpoint['wandb_run_id'] = wandb.run.id

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


def load_checkpoint(cfg, model, optimizer, checkpoint_dir, device, ema_model=None):
    resume = getattr(cfg.training, "resume_from_checkpoint", False)
    if not resume: return 0, None

    ckpt_path = getattr(cfg.training, "resume_from_checkpoint_path", None)
    if not (ckpt_path and os.path.isfile(ckpt_path)):
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_iter*.pt")))
        ckpt_path = checkpoint_files[-1] if checkpoint_files else None

    if not ckpt_path:
        print("No checkpoints found, starting from scratch")
        return 0, None

    load_optimizer_state = getattr(cfg.training, "load_optimizer_state", True)

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"]
        
        base_model = _get_base_model(model)
        
        has_orig_mod = any(k.startswith("_orig_mod.") for k in state_dict.keys())
        current_has_orig_mod = any(k.startswith("_orig_mod.") for k in base_model.state_dict().keys())
        
        if has_orig_mod and not current_has_orig_mod: state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        elif not has_orig_mod and current_has_orig_mod: state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
        base_model.load_state_dict(state_dict)
        
        if load_optimizer_state:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (RuntimeError, KeyError, ValueError) as e:
                if "parameter group" in str(e) or "size mismatch" in str(e) or "The size of tensor" in str(e):
                    print(f"[WARNING] Skipped loading optimizer state due to shape/parameter mismatch: {e}")
                    print("Optimizer state will be reinitialized with fresh buffers.")
                else:
                    raise
        else:
            print("[INFO] Skipped loading optimizer state (load_optimizer_state=False)")
            
        if ema_model is not None and "ema_state_dict" in ckpt: ema_model.load_state_dict(ckpt["ema_state_dict"])
        else: print("Could not find an ema model within the checkpoint.")
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        return 0, None

    global_step = ckpt.get("global_step", ckpt.get("iteration", 0))
    wandb_run_id = ckpt.get("wandb_run_id", None)
    
    if isinstance(global_step, tuple): global_step = global_step[0] if global_step else 0
    
    if wandb_run_id: print(f"WandB run ID: {wandb_run_id}")
    global_step += 1
    return global_step, wandb_run_id
