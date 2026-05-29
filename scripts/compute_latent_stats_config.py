from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf

base_path = Path(sys.argv[1])
runtime_path = Path(sys.argv[2])
cfg = OmegaConf.load(base_path)
train_dir = Path(cfg.paths.latent_train_dir)
paths = sorted(train_dir.rglob("*.npy"))
if not paths:
    raise SystemExit(f"No latent files found under {train_dir}")

total = 0
sum_ = 0.0
sumsq = 0.0
for path in paths:
    arr = np.load(path, mmap_mode="r")
    data = arr.astype(np.float64, copy=False)
    total += data.size
    sum_ += float(data.sum())
    sumsq += float((data * data).sum())

mean = sum_ / total
var = max((sumsq - total * mean * mean) / max(1, total - 1), 0.0)
std = var ** 0.5
cfg.objective.latent_mean = float(mean)
cfg.objective.latent_std = float(std)
runtime_path.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, runtime_path)
print(f"Computed latent stats from {len(paths)} files: mean={mean:.8f} std={std:.8f}")
print(f"Wrote runtime config: {runtime_path}")
