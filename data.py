import os
from collections import OrderedDict
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.stack(batch)


class HDF5Backend:
    def __init__(self, hdf_path, group_names=["Vol_full"], dims="2d", n_slices=None):
        self.hdf_path = hdf_path
        self.group_names = group_names
        self.load_volumes = dims == "3d"
        self.n_slices = n_slices
        self.index_map = []
        self._h5_file = h5py.File(self.hdf_path, "r")
        self._build_index_map()

    def _build_index_map(self):
        for group in self.group_names:
            if group not in self._h5_file:
                raise ValueError(f"Group '{group}' not found in HDF5 file.")
            for key in self._h5_file[group].keys():
                shape = self._h5_file[group][key].shape
                total_slices = shape[0] if len(shape) == 3 else shape[1]
                start, end = 0, total_slices
                if self.n_slices is not None and self.n_slices < total_slices:
                    start = (total_slices - self.n_slices) // 2
                    end = start + self.n_slices

                if self.load_volumes:
                    self.index_map.append((group, key, (start, end), shape))
                else:
                    for s in range(start, end):
                        self.index_map.append((group, key, s, shape))

    def get_volume(self, idx):
        group, key, slice_idx, shape = self.index_map[idx]
        dataset = self._h5_file[group][key]
        if self.load_volumes:
            start, end = slice_idx
            vol = dataset[start:end]
            if vol.ndim == 3:
                vol = vol[None, ...]
        else:
            if len(dataset.shape) == 3:
                vol = dataset[slice_idx]
            else:
                vol = dataset[0, slice_idx]
            vol = vol[None, ...]
        return torch.from_numpy(vol.copy()).float()

    def index_map_preview(self):
        return [(g, k, s, sh) for g, k, s, sh in self.index_map]

    def close(self):
        self._h5_file.close()


class NiftiBackend:
    def __init__(
        self,
        nifti_dir=None,
        csv_metadata=None,
        dims="2d",
        n_slices=None,
        fraction=1.0,
        seed=42,
        cache_size=8,
    ):
        import json
        import os
        import hashlib

        self.nifti_dir = nifti_dir
        self.csv_metadata = csv_metadata
        self.load_volumes = dims == "3d"
        self.n_slices = n_slices
        self.index_map = []

        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.seed = seed
        self.fraction = fraction

        mode = "3d" if self.load_volumes else "2d"
        if self.nifti_dir:
            dir_hash = hashlib.md5(self.nifti_dir.encode()).hexdigest()[:8]
        else:
            dir_hash = "metadata"
        frac_str = f"_{fraction:.2f}" if fraction < 1.0 else ""
        self.cache_path = os.path.expanduser(f"~/DC-Gen/index_map_{mode}_{dir_hash}{frac_str}.json")

        if Path(self.cache_path).exists():
            with open(self.cache_path, "r") as f:
                self.index_map = json.load(f)
            return

        self.files = self._collect_files(fraction, seed)
        self._build_index_map()

        with open(self.cache_path, "w") as f:
            json.dump(self.index_map, f)

    def _collect_files(self, fraction, seed):
        if self.csv_metadata:
            import pandas as pd

            df = pd.read_csv(self.csv_metadata)
            files = [str(Path(row["path"])) for _, row in df.iterrows()]
        else:
            files = [str(p) for p in Path(self.nifti_dir).rglob("*.nii*")]

        rng = np.random.default_rng(seed)
        n_select = int(len(files) * fraction)
        return rng.choice(files, n_select, replace=False).tolist()

    def _get_shape_fast(self, file):
        img = nib.load(file, mmap=True)
        return img.header.get_data_shape()

    def _build_index_map(self):
        rng = np.random.default_rng(self.seed)

        pbar = tqdm(
            self.files,
            desc="Indexing NIfTI",
            unit="file",
            dynamic_ncols=True,
            smoothing=0.05,
            miniters=1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for file in pbar:
            pbar.set_postfix_str(Path(file).name[:40])
            shape = self._get_shape_fast(file)
            total_slices = shape[2]

            start, end = 0, total_slices
            if self.n_slices is not None and self.n_slices < total_slices:
                start = (total_slices - self.n_slices) // 2
                end = start + self.n_slices

            file = str(file)

            if self.load_volumes:
                self.index_map.append([file, [start, end]])
            else:
                available = np.arange(start, end)
                num_samples = min(32, len(available))
                sampled = rng.choice(available, size=num_samples, replace=False)

                for s in sampled:
                    self.index_map.append([file, int(s)])

    def _get_cached_img(self, file):
        if file in self.cache:
            img = self.cache.pop(file)
            self.cache[file] = img
            return img

        img = nib.load(file, mmap=True)

        self.cache[file] = img
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return img

    def get_volume(self, idx):
        entry = self.index_map[idx]
        file = entry[0]

        try:
            img = self._get_cached_img(file)
            data = np.asarray(img.dataobj)
        except (EOFError, OSError) as e:
            raise RuntimeError(f"Corrupted file: {file}. Error: {e}")

        if self.load_volumes:
            start, end = entry[1]
            end = min(end, data.shape[2])
            vol = data[..., start:end]
            vol = np.transpose(vol, (2, 0, 1))
            vol = vol[None, ...]
        else:
            s = entry[1]
            s = min(int(s), data.shape[2] - 1)

            vol = data[..., s]
            vol = vol[None, ...]

        return torch.from_numpy(vol.copy()).float()

    def index_map_preview(self):
        return self.index_map[:10]


class CTVolumeDataset(Dataset):
    def __init__(
        self,
        hdf_path=None,
        nifti_dir=None,
        csv_metadata=None,
        group_names=["Vol_full"],
        dims="2d",
        n_slices=None,
        fraction=1.0,
        transform=None,
    ):
        self.transform = transform
        self.dims = dims
        self.load_volumes = dims == "3d"
        self.n_slices = n_slices

        if hdf_path:
            self.backend = HDF5Backend(hdf_path, group_names, dims, n_slices)
        elif nifti_dir or csv_metadata:
            self.backend = NiftiBackend(
                nifti_dir, csv_metadata, dims, n_slices, fraction
            )
        else:
            raise ValueError("Must provide either hdf_path or nifti_dir/csv_metadata")

    def __len__(self):
        return len(self.backend.index_map)

    def __getitem__(self, idx):
        try:
            vol = self.backend.get_volume(idx)
        except (EOFError, OSError, RuntimeError) as e:
            print(f"Skipping corrupted sample at idx {idx}: {e}")
            return None
        if self.transform:
            vol = self.transform(vol)
        return vol

    def index_map_preview(self):
        return self.backend.index_map_preview()

    def batch_shape_preview(self, batch_size=4):
        preview_shapes = [entry[-1] for entry in self.backend.index_map[:batch_size]]
        if self.load_volumes:
            batch_shape = (len(preview_shapes), 1, *preview_shapes[0])
        else:
            batch_shape = (len(preview_shapes), 1, *preview_shapes[0][1:])
        return batch_shape

    def close(self):
        if isinstance(self.backend, HDF5Backend):
            self.backend.close()
