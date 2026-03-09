import torch
import numpy as np
import os
import wandb
import glob

def to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()

def save_checkpoint(cfg, model, optimizer, artifact_dir, it, checkpoint_queue, max_checkpoints):
    os.makedirs(artifact_dir, exist_ok=True)
    ckpt_path = os.path.join(artifact_dir, f"checkpoint_iter{it}.pt")
    torch.save({
        'global_step': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, ckpt_path)
    if cfg.wandb.enabled:
        wandb.save(ckpt_path)
    checkpoint_queue.append(ckpt_path)
    while len(checkpoint_queue) > max_checkpoints:
        old_ckpt = checkpoint_queue.popleft()
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

def load_checkpoint(cfg, model, optimizer, checkpoint_dir, device):
    if not getattr(cfg.training, "resume_from_checkpoint", False):
        return 0

    user_ckpt = getattr(cfg.training, "resume_from_checkpoint_path", None)
    if user_ckpt and os.path.exists(user_ckpt):
        ckpt_path = user_ckpt
    else:
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_iter*.pt")))
        if not checkpoint_files:
            print("no checkpoints found, starting from scratch")
            return 0
        ckpt_path = checkpoint_files[-1]

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    global_step = ckpt.get("global_step", ckpt.get("iteration", 0))
    print(f"resumed from checkpoint {ckpt_path} at global_step {global_step}")
    return global_step