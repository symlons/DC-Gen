import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import os

class CTVolumeDataset(Dataset):
    def __init__(self, hdf_path, group_names=["Vol_full"], volume=False, n_slices=None, transform=None):
        self.hdf_path = hdf_path
        self.group_names = group_names
        self.volume = volume
        self.n_slices = n_slices
        self.transform = transform
        self.index_map = []

        with h5py.File(self.hdf_path, "r") as f:
            for group_name in group_names:
                if group_name not in f:
                    raise ValueError(f"Group '{group_name}' not found in HDF5 file.")
                for key in f[group_name].keys():
                    vol = f[group_name][key][()]
                    if volume:
                        self.index_map.append((group_name, key, None))
                    else:
                        total_slices = vol.shape[0] if vol.ndim == 3 else vol.shape[1]
                        if self.n_slices is not None and self.n_slices < total_slices:
                            start = (total_slices - self.n_slices) // 2
                            end = start + self.n_slices
                        else:
                            start, end = 0, total_slices
                        for s in range(start, end):
                            self.index_map.append((group_name, key, s))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        group_name, key, slice_idx = self.index_map[idx]
        with h5py.File(self.hdf_path, "r") as f:
            vol = f[group_name][key][()]
        if self.volume:
            if slice_idx is not None:
                vol = vol[slice_idx]
            if vol.ndim == 3:
                vol = vol[None, ...]
        else:
            if vol.ndim == 3:
                vol = vol[slice_idx]
            elif vol.ndim == 4:
                vol = vol[0, slice_idx]
            vol = vol[None, ...]
        vol = torch.from_numpy(vol).float()
        if self.transform:
            vol = self.transform(vol)
        return vol