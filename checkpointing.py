import os
import glob
from collections import deque
import torch
import wandb

def save_checkpoint(cfg, model, optimizer, artifact_dir, it, checkpoint_queue: deque, max_checkpoints: int):
    os.makedirs(artifact_dir, exist_ok=True)
    ckpt_path = os.path.join(artifact_dir, f"checkpoint_iter{it}.pt")

    checkpoint = {
        'global_step': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

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
        return 0

    ckpt_path = getattr(cfg.training, "resume_from_checkpoint_path", None)
    if not (ckpt_path and os.path.isfile(ckpt_path)):
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_iter*.pt")))
        ckpt_path = checkpoint_files[-1] if checkpoint_files else None

    if not ckpt_path:
        print("No checkpoints found, starting from scratch")
        return 0

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        return 0

    global_step = ckpt.get("global_step", ckpt.get("iteration", 0))
    print(f"Resumed from checkpoint {ckpt_path} at global_step {global_step}")
    return global_step
