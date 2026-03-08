import torch
import numpy as np
import os
import wandb

def to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()

def save_checkpoint(cfg, model, optimizer, artifact_dir, it, checkpoint_queue, max_checkpoints):
    os.makedirs(artifact_dir, exist_ok=True)
    ckpt_path = os.path.join(artifact_dir, f"checkpoint_iter{it}.pt")
    torch.save({
        'iteration': it,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, ckpt_path)
    if cfg.wandb.enabled: wandb.save(ckpt_path)
    checkpoint_queue.append(ckpt_path)
    while len(checkpoint_queue) > max_checkpoints:
        old_ckpt = checkpoint_queue.popleft()
        if os.path.exists(old_ckpt): os.remove(old_ckpt)