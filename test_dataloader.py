import torch
from torch.utils.data import DataLoader

from config import load_config
from registry import dataset_registry


def main():
    cfg = load_config()

    dataset = dataset_registry[cfg.dataset.name](
        nifti_dir=cfg.paths.nifti_dir,
        group_names=cfg.dataset.group_names,
        dims=cfg.dims,
        n_slices=None,
        transform=None,
    )

    if len(dataset) == 0:
        raise ValueError("empty dataset")

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"dataset: {len(dataset)}")

    for i, x in enumerate(loader):
        print(f"{i}: shape={tuple(x.shape)} dtype={x.dtype}")

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("invalid values")

        if i == 2:
            break


if __name__ == "__main__":
    main()
