import os
import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


def find_all_nifti(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".nii.gz"):
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def load_nifti(path):
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    np.clip(data, -1000, 1000, out=data)
    return data.astype(np.int16, copy=False)


def get_written_count(shape_ds):
    if "written" in shape_ds.attrs:
        return int(shape_ds.attrs["written"])
    return 0


def set_written_count(shape_ds, count):
    shape_ds.attrs["written"] = count


def write_shards(nifti_paths, out_dir, shard_size=256):
    os.makedirs(out_dir, exist_ok=True)
    total = len(nifti_paths)

    with tqdm(total=total, desc="Total progress") as pbar_total:
        for shard_idx, i in enumerate(range(0, total, shard_size)):
            shard_paths = nifti_paths[i : i + shard_size]
            shard_file = os.path.join(out_dir, f"ct_shard_{shard_idx:04d}.h5")

            dt = h5py.vlen_dtype(np.dtype("int16"))

            if os.path.exists(shard_file):
                f = h5py.File(shard_file, "a")
                data_ds = f["data"]
                shape_ds = f["shape"]
                written = get_written_count(shape_ds)

                if written >= len(shard_paths):
                    pbar_total.update(len(shard_paths))
                    f.close()
                    continue
            else:
                f = h5py.File(shard_file, "w")
                data_ds = f.create_dataset("data", (len(shard_paths),), dtype=dt)
                shape_ds = f.create_dataset("shape", (len(shard_paths), 3), dtype=np.int32)
                written = 0
                set_written_count(shape_ds, 0)

            print(f"\nWriting {shard_file} from index {written}/{len(shard_paths)}")
            pbar_total.update(written)

            for j in tqdm(
                range(written, len(shard_paths)),
                desc=f"Shard {shard_idx}",
                leave=False,
            ):
                p = shard_paths[j]

                try:
                    vol = load_nifti(p)
                except:
                    continue

                data_ds[j] = vol.flatten()
                shape_ds[j] = vol.shape

                written += 1
                set_written_count(shape_ds, written)
                pbar_total.update(1)

            f.close()


if __name__ == "__main__":
    root = "/scratch/train"
    out = "/cluster/projects/2025_stmd_VT_diff/hdfs2"

    files = find_all_nifti(root)
    print(f"Found {len(files)} files")

    write_shards(files, out, shard_size=256)
