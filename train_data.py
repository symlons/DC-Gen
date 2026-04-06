from torch.utils.data import DataLoader
from flow.latent_dataset import LatentTensorDataset


def make_dataset(cfg, split: str):
    if split not in ("train", "val"): raise ValueError(f"Invalid split: {split}")

    dataset_dir = getattr(cfg.dataset, f"{split}_dir")
    fraction = getattr(cfg.dataset, f"{split}_fraction", cfg.dataset.fraction)

    if dataset_dir is None: return None

    return LatentTensorDataset(
        root_dir=dataset_dir,
        extensions=tuple(cfg.dataset.extensions),
        recursive=cfg.dataset.recursive,
        fraction=fraction,
        seed=cfg.dataset.seed,
    )

def make_dataloader(dataset, cfg, batch_size=None, sampler=None, shuffle=False):
    if dataset is None: return None

    batch_size = batch_size or cfg.training.batch_size
    num_workers = cfg.training.num_workers

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        persistent_workers=cfg.training.persistent_workers if num_workers > 0 else False,
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor if num_workers > 0 else None,
    )
