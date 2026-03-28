import os
import argparse

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


def find_all_nifti(root, ext):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(ext) or f.endswith(ext + ".gz"):
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def load_nifti(path):
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    np.clip(data, -1000, 1000, out=data)
    return data.astype(np.int16, copy=False)


def process_volume(vol, n_slices):
    d = vol.shape[2]

    if d >= n_slices:
        start = (d - n_slices) // 2
        vol = vol[:, :, start:start + n_slices]
    else:
        pad = n_slices - d
        pad_before = pad // 2
        pad_after = pad - pad_before
        vol = np.pad(
            vol,
            ((0, 0), (0, 0), (pad_before, pad_after)),
            constant_values=-1000,
        )

    return np.transpose(vol, (2, 0, 1))


def write_hdf(nifti_paths, out_file, n_slices):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    total = len(nifti_paths)
    print(f"Writing {total} volumes to {out_file}")

    sample = process_volume(load_nifti(nifti_paths[0]), n_slices)
    D, H, W = sample.shape

    with h5py.File(out_file, "w") as f:
        data_ds = f.create_dataset(
            "data",
            shape=(total, D, H, W),
            dtype=np.int16,
            chunks=(1, D, H, W),
        )

        for i, p in enumerate(tqdm(nifti_paths, desc="Processing")):
            vol = load_nifti(p)
            vol = process_volume(vol, n_slices)
            data_ds[i] = vol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ext", default=".nii")
    parser.add_argument("--n_slices", type=int, default=32)
    args = parser.parse_args()

    files = find_all_nifti(args.root, args.ext)
    print(f"Found {len(files)} files in {args.root}")

    write_hdf(files, args.out, args.n_slices)
