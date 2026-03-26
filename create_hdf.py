import os
import h5py
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import threading
import queue
import argparse

TARGET_DEPTH   = 256
PAD_VALUE      = -1000
HU_MIN, HU_MAX = -1000, 1000
QUEUE_MAXSIZE  = 32


def find_all_nifti(root, ext=".nii.gz"):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in sorted(filenames):
            if f.endswith(ext):
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def load_and_process(path):
    """Runs in worker process — pure CPU, no HDF5 touches."""
    try:
        import SimpleITK as sitk
        sitk_img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(sitk_img)   # (Z, X, Y) float32
        data = data.astype(np.float32)
        np.clip(data, HU_MIN, HU_MAX, out=data)
        data = data.astype(np.int16)

        z = data.shape[0]
        if z >= TARGET_DEPTH:
            start = (z - TARGET_DEPTH) // 2
            data  = data[start : start + TARGET_DEPTH]
        else:
            pad  = TARGET_DEPTH - z
            data = np.pad(
                data,
                ((pad // 2, pad - pad // 2), (0, 0), (0, 0)),
                mode="constant",
                constant_values=PAD_VALUE,
            )

        return path, data   # (TARGET_DEPTH, X, Y) int16

    except Exception as e:
        print(f"[WARN] {path}: {e}")
        return path, None


def writer_thread(q, data_ds, paths_ds, f, total, checkpoint_every=50):
    """Single thread that owns all HDF5 writes."""
    written = int(f.attrs.get("written", 0))
    failed  = int(f.attrs.get("failed",  0))
    pbar    = tqdm(total=total, initial=0, desc="Writing")

    while True:
        item = q.get()
        if item is None:        # poison pill
            break

        idx, path, vol = item
        paths_ds[idx] = path

        if vol is not None:
            data_ds[idx] = vol
        else:
            failed += 1

        written += 1

        if written % checkpoint_every == 0:
            f.attrs["written"] = written
            f.attrs["failed"]  = failed
            f.flush()

        pbar.set_postfix(failed=failed)
        pbar.update(1)
        q.task_done()

    f.attrs["written"] = written
    f.attrs["failed"]  = failed
    f.flush()
    pbar.close()
    print(f"\nDone. {written - failed}/{written} volumes written, {failed} failed.")


def write_hdf(nifti_paths, out_path, num_workers=16):
    N = len(nifti_paths)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, "a", rdcc_nbytes=256 * 1024 * 1024) as f:

        if "data" not in f:
            f.create_dataset(
                "data",
                shape=(N, TARGET_DEPTH, 512, 512),
                dtype=np.int16,
                chunks=(1, 32, 512, 512),
                compression="lzf",
                shuffle=True,
            )
        if "paths" not in f:
            f.create_dataset(
                "paths",
                shape=(N,),
                dtype=h5py.special_dtype(vlen=str),
            )

        data_ds  = f["data"]
        paths_ds = f["paths"]

        written = int(f.attrs.get("written", 0))
        remaining_paths = nifti_paths[written:]
        remaining_idxs  = list(range(written, N))

        if not remaining_paths:
            print("Already complete.")
            return

        print(f"Resuming from {written}/{N}  ({N - written} remaining)")

        q = queue.Queue(maxsize=QUEUE_MAXSIZE)

        wt = threading.Thread(
            target=writer_thread,
            args=(q, data_ds, paths_ds, f, len(remaining_paths)),
            daemon=True,
        )
        wt.start()

        with mp.Pool(processes=num_workers, maxtasksperchild=100) as pool:
            for (path, vol), idx in zip(
                pool.imap(load_and_process, remaining_paths, chunksize=8),
                remaining_idxs,
            ):
                q.put((idx, path, vol))

        q.put(None)   # poison pill
        wt.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    required=True,  help="Root dir with NIfTI files")
    parser.add_argument("--out",     required=True,  help="Output .h5 path")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ext",     default=".nii.gz", help="File extension to search for")
    args = parser.parse_args()

    files = find_all_nifti(args.root, ext=args.ext)
    print(f"Found {len(files)} files with extension '{args.ext}'")

    write_hdf(files, args.out, num_workers=args.workers)
