import torch
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, ScaleIntensity, Resize
import os
from monai.metrics import PSNRMetric
from monai.metrics.utils import MetricReduction
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dc_gen.ae_model_zoo import DCAE_HF

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
                        if n_slices is not None and vol.ndim == 3 and n_slices < vol.shape[0]:
                            step = vol.shape[0] / n_slices
                            indices = [int(step * i + step / 2) for i in range(n_slices)]
                            self.index_map.append((group_name, key, indices))
                        else:
                            self.index_map.append((group_name, key, None))
                    else:
                        total_slices = vol.shape[0] if vol.ndim == 3 else vol.shape[1]
                        slice_indices = range(total_slices)
                        if n_slices is not None:
                            step = max(1, total_slices // n_slices)
                            slice_indices = slice_indices[::step][:n_slices]
                        for s in slice_indices:
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

hdf_path = "/workspace/ct_rate_train_batch_0_v13.hdf"
n_slices = 32
pipeline_3d = Compose([
    ScaleIntensity(minv=0.0, maxv=1.0),
    Resize(spatial_size=(n_slices, 128, 128))  # D, H, W
])
dataset_3d = CTVolumeDataset(hdf_path, group_names=["Vol_full"], volume=True, n_slices=n_slices, transform=pipeline_3d)
loader_3d = DataLoader(dataset_3d, batch_size=1, shuffle=False)

batch_3d = next(iter(loader_3d))
print("3D batch shape:", batch_3d.shape)

artifact_dir = "artifacts_4"
os.makedirs(artifact_dir, exist_ok=True)

device = torch.device("cuda")
model_name = "dc-ae-f32c32-in-1.0"
model = DCAE_HF(model_name=model_name).to(dtype=torch.bfloat16, device=device)
model.train()

lr = 1e-5
batch_size = loader_3d.batch_size if hasattr(loader_3d, 'batch_size') else 1
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

batch_3d = next(iter(loader_3d))
if batch_3d.ndim == 4:
    batch_3d = batch_3d.unsqueeze(1)
batch_3d = batch_3d.to(dtype=torch.bfloat16, device=device)

max_val = float(batch_3d.max() - batch_3d.min())
psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.MEAN)

loss_history = []
psnr_history = []

log_file = os.path.join(artifact_dir, "training_log.txt")
with open(log_file, "w") as f:
    pbar = tqdm(range(8000))
    for it in pbar:
        latent = model.encoder(batch_3d)
        recon = model.decoder(latent)

        loss = torch.nn.functional.l1_loss(recon, batch_3d)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        psnr_metric(recon, batch_3d)
        psnr_value = psnr_metric.aggregate().item()
        psnr_history.append(psnr_value)

        pbar.set_postfix({'loss': loss.item(), 'PSNR': psnr_value})
        f.write(f"iter {it}: loss={loss.item():.6f}, PSNR={psnr_value:.6f}\n")

        if it % 50 == 0:
            recon_cpu = recon.detach().cpu().float().numpy()  # [B, C, D, H, W]
            gt_cpu = batch_3d.detach().cpu().float().numpy()
            diff_cpu = np.abs(gt_cpu - recon_cpu)

            for b in range(recon_cpu.shape[0]):
                # Save reconstructed volume
                nii_recon = nib.Nifti1Image(recon_cpu[b, 0], affine=np.eye(4))
                nib.save(nii_recon, os.path.join(artifact_dir, f"recon_iter{it}_sample{b}.nii.gz"))

                # Save ground truth volume
                nii_gt = nib.Nifti1Image(gt_cpu[b, 0], affine=np.eye(4))
                nib.save(nii_gt, os.path.join(artifact_dir, f"gt_iter{it}_sample{b}.nii.gz"))

                # Save diff volume
                nii_diff = nib.Nifti1Image(diff_cpu[b, 0], affine=np.eye(4))
                nib.save(nii_diff, os.path.join(artifact_dir, f"diff_iter{it}_sample{b}.nii.gz"))

            # Central slice images for first sample
            depth_center = recon_cpu.shape[2] // 2
            recon_slice = recon_cpu[0, 0, depth_center]
            gt_slice = gt_cpu[0, 0, depth_center]
            diff_slice = diff_cpu[0, 0, depth_center]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(gt_slice, cmap='gray'); axes[0].axis('off'); axes[0].set_title("Ground Truth")
            axes[1].imshow(recon_slice, cmap='gray'); axes[1].axis('off'); axes[1].set_title("Reconstruction")
            axes[2].imshow(diff_slice, cmap='hot'); axes[2].axis('off'); axes[2].set_title("Difference")
            plt.tight_layout()
            plt.savefig(os.path.join(artifact_dir, f"central_slice_iter{it}.png"), dpi=300)
            plt.close()

# Plot loss and PSNR curves
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(loss_history, 'r-', label='Loss')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss', color='r')
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()
ax2.plot(psnr_history, 'b-', label='PSNR')
ax2.set_ylabel('PSNR', color='b')
ax2.tick_params(axis='y', labelcolor='b')

fig.tight_layout()
plt.title(f'Training Curves (lr={lr}, batch_size={batch_size})')
plt.savefig(os.path.join(artifact_dir, f'training_curves_lr{lr}_bs{batch_size}.png'), dpi=300)
plt.close()