import os
import time
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm

split = "valid"
fraction = 0.1
shuffle = False

data_root = "/mnt/Volume-eV4BofCN/storage"
out_root  = "/mnt/Volume-eV4BofCN/storage_processed"

HU_MIN = -1000
HU_MAX = 1000

metadata_file = {
    "train": "train_metadata.csv",
    "valid": "validation_metadata.csv"
}

script_dir = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(script_dir, metadata_file[split])

if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

data = pd.read_csv(metadata_path)

if shuffle:
    data = data.sample(frac=fraction, random_state=42).reset_index(drop=True)
else:
    if fraction < 1.0:
        data = data.sample(frac=fraction, random_state=42)
    data["_folder"] = data["VolumeName"].apply(lambda x: "_".join(x.split("_")[:2]))
    data = data.sort_values("_folder").drop(columns="_folder").reset_index(drop=True)

total = len(data)
processed = 0
skipped = 0
failed = 0

start_time = time.time()

SEP = "─" * 52
print(f"\n{SEP}")
print(f"Processing {total} volumes")
print(f"{SEP}\n")

with tqdm(total=total, desc="Processing", unit="vol") as pbar:
    for _, row in data.iterrows():
        name = row["VolumeName"]

        parts = name.split("_")
        folder = parts[0] + "_" + parts[1]
        sub = folder + "_" + parts[2]

        in_path = os.path.join(data_root, "dataset", split, folder, sub, name)
        out_dir = os.path.join(out_root, "dataset", split, folder, sub)
        out_path = os.path.join(out_dir, name)

        try:
            if not os.path.exists(in_path):
                skipped += 1
                pbar.update(1)
                continue

            if os.path.exists(out_path):
                skipped += 1
                pbar.update(1)
                continue

            os.makedirs(out_dir, exist_ok=True)

            slope = float(row["RescaleSlope"])
            intercept = float(row["RescaleIntercept"])

            nii = nib.load(in_path)
            raw = np.asarray(nii.dataobj)

            vol = raw.astype(np.float32) * slope + intercept
            vol = np.clip(vol, HU_MIN, HU_MAX)
            vol = vol.astype(np.int16)

            new_img = nib.Nifti1Image(vol, nii.affine, nii.header)
            new_img.set_data_dtype(np.int16)
            new_img.header["scl_slope"] = 1.0
            new_img.header["scl_inter"] = 0.0

            nib.save(new_img, out_path)

            processed += 1

        except Exception as e:
            failed += 1
            print(f"Error with {name}: {e}")

        pbar.update(1)

elapsed = time.time() - start_time

print(f"\n{SEP}")
print(f"Processed: {processed}")
print(f"Skipped:   {skipped}")
print(f"Failed:    {failed}")
print(f"Time:      {elapsed/60:.1f} min")
print(f"{SEP}\n")