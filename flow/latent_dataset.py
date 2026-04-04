from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

TENSOR_DICT_KEYS = ("image", "images", "latent")


class LatentTensorDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        extensions: tuple[str, ...] = (".npy", ".pt", ".pth"),
        recursive: bool = True,
        fraction: float = 1.0,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.extensions = tuple(ext.lower() for ext in extensions)
        if not self.root_dir.exists(): raise FileNotFoundError(f"Latent root dir does not exist: {self.root_dir}")

        pattern = "**/*" if recursive else "*"
        paths = sorted(path for path in self.root_dir.glob(pattern) if path.is_file() and path.suffix.lower() in self.extensions)
        if not paths: raise ValueError(f"No latent files found under {self.root_dir} with extensions {self.extensions}")

        if fraction < 1.0:
            rng = np.random.default_rng(seed)
            count = max(1, int(len(paths) * fraction))
            indices = np.sort(rng.choice(len(paths), size=count, replace=False))
            paths = [paths[index] for index in indices]

        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.load_tensor(self.paths[index])}

    @staticmethod
    def load_tensor(path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            tensor = torch.from_numpy(np.load(path))
        elif suffix in {".pt", ".pth"}:
            data = torch.load(path, map_location="cpu")
            if isinstance(data, dict):
                for key in TENSOR_DICT_KEYS:
                    if key in data:
                        data = data[key]
                        break
                else:
                    raise ValueError(f"Unsupported tensor dict keys in {path}: {list(data.keys())}")
            tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
        else:
            raise ValueError(f"Unsupported latent extension: {suffix}")

        tensor = tensor.float()
        if tensor.ndim != 4:
            raise ValueError(f"Expected latent tensor shaped [C, D, H, W], got {tuple(tensor.shape)} from {path}")
        return tensor


def infer_latent_shape(
    dataset: Dataset,
    expected_in_channels: Optional[int] = None,
    expected_input_size: Optional[tuple[int, int, int]] = None,
) -> tuple[int, tuple[int, int, int]]:
    sample = dataset[0]["image"]
    if sample.ndim != 4: raise ValueError(f"Expected latent sample [C, D, H, W], got {tuple(sample.shape)}")

    in_channels = int(sample.shape[0])
    spatial_shape = tuple(int(dim) for dim in sample.shape[1:])

    if expected_in_channels is not None and expected_in_channels != in_channels: raise ValueError(f"Configured model.in_channels={expected_in_channels} does not match latent channels={in_channels}.")
    if expected_input_size is not None and expected_input_size != spatial_shape: raise ValueError(f"Configured model.input_size={expected_input_size} does not match latent shape={spatial_shape}.")

    return in_channels, spatial_shape
