from registry import dataset_registry
from torch.utils.data import DataLoader
from structure import build_pipeline
import os
from viz import Visualize

dataset_name = "CTVolume"
dataset_cls = dataset_registry[dataset_name]

class Cfg:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, Cfg(v) if isinstance(v, dict) else v)
cfg = {
    "paths": {
        "nifti_dir": "/cluster/projects/ac3t/data/ac3t_ct_rate/processed/train/"
    },
    "dataset": {
        "group_names": ["train"]
    },
    "pipeline": {
        "n_slices": 32,
        "resize_hw": [256, 256],
        "normalize_input_range": [-1000, 1000],
        "normalize_output_range": [-1.0, 1.0]
    },
    "dims": "3d",
    "training": {
        "batch_size": 8,
        "shuffle_data": False,
        "pin_memory": True,
        "prefetch_factor": None,
        "num_workers": 0
    }
}

cfg = Cfg(cfg)
pipeline = build_pipeline(cfg)
dataset = dataset_cls(
    nifti_dir=cfg.paths.nifti_dir,
    group_names=cfg.dataset.group_names,
    dims=cfg.dims,
    n_slices=cfg.pipeline.n_slices,
    transform=pipeline
)
dataset_no_transform = dataset_cls(
    nifti_dir=cfg.paths.nifti_dir,
    group_names=cfg.dataset.group_names,
    dims=cfg.dims,
    n_slices=cfg.pipeline.n_slices,
    transform=None
)
print(dataset_cls)
loader = DataLoader(
    dataset,
    batch_size=cfg.training.batch_size,
    shuffle=cfg.training.shuffle_data,
    pin_memory=cfg.training.pin_memory,
    num_workers=cfg.training.num_workers,
    prefetch_factor=cfg.training.prefetch_factor if cfg.training.num_workers > 0 else None,
    persistent_workers=cfg.training.num_workers > 0,
    multiprocessing_context=None
)
loader_no_transform = DataLoader(
    dataset_no_transform,
    batch_size=cfg.training.batch_size,
    shuffle=cfg.training.shuffle_data,
    pin_memory=cfg.training.pin_memory,
    num_workers=cfg.training.num_workers,
    prefetch_factor=cfg.training.prefetch_factor if cfg.training.num_workers > 0 else None,
    persistent_workers=cfg.training.num_workers > 0,
    multiprocessing_context=None
)

viz = Visualize(viz_type="3d")
save_dir = "./debug_viz"
os.makedirs(save_dir, exist_ok=True)

for batch in loader:
    print(batch.shape)
    print(batch.min(), batch.max(), batch.mean(), batch.median())
    viz.save(gt=batch, recon=batch, save_dir=save_dir, global_step=0)
    break
# for batch in loader_no_transform:
#     print(batch.shape)
#     print(batch.min(), batch.max(), batch.mean(), batch.median())
#     viz.save(gt=batch, recon=batch, save_dir=save_dir, global_step=0)
#     break