import os
import torch
from tqdm import tqdm
from monai.metrics import PSNRMetric
from monai.metrics.utils import MetricReduction
import matplotlib.pyplot as plt
from data import CTVolumeDataset
from dc_gen.ae_model_zoo import DCAE_HF
from torch.utils.data import DataLoader
from monai.transforms import Compose, ScaleIntensity
import numpy as np
from viz import plot_training_curves, diff_visualization
from basics import to_numpy

artifact_dir = "artifacts_4"
hdf_path = "/workspace/ct_rate_train_batch_0_v13.hdf"
batch_size = 1
n_slices = 16
shuffle_data = False
model_name = "dc-ae-f32c32-in-1.0"
device = torch.device("cuda")
dtype = torch.bfloat16
lr = 1e-5
num_iters = 8000
diff_save_every = 500

pipeline_2d = Compose([ScaleIntensity(minv=-1.0, maxv=1.0)])
dataset_2d = CTVolumeDataset(hdf_path, group_names=["Vol_full"], volume=False, n_slices=n_slices, transform=pipeline_2d)
loader_2d = DataLoader(dataset_2d, batch_size=batch_size, shuffle=shuffle_data)
batch_2d = next(iter(loader_2d))

os.makedirs(artifact_dir, exist_ok=True)

model = DCAE_HF(model_name=model_name).to(dtype=dtype, device=device)
model.train()
model = torch.compile(model)

if batch_2d.ndim == 3:
    batch_2d = batch_2d.unsqueeze(1)
batch_2d = batch_2d.to(dtype=dtype, device=device)

max_val = float(batch_2d.max() - batch_2d.min())
psnr_metric = PSNRMetric(max_val=max_val, reduction=MetricReduction.MEAN)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_history = []
psnr_history = []

log_file = os.path.join(artifact_dir, "training_log.txt")
with open(log_file, "w") as f:
    pbar = tqdm(range(num_iters))
    for it in pbar:
        latent = model.encoder(batch_2d)
        recon = model.decoder(latent)
        loss = torch.nn.functional.l1_loss(recon, batch_2d)
        loss_history.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        psnr_metric(recon, batch_2d)
        psnr_value = psnr_metric.aggregate().item()
        psnr_history.append(psnr_value)
        pbar.set_postfix({'loss': loss.item(), 'PSNR': psnr_value})
        f.write(f"iter {it}: loss={loss.item():.6f}, PSNR={psnr_value:.6f}\n")
        if it % diff_save_every == 0:
            save_path = os.path.join(artifact_dir, f"reconstruction_diff_red_blue_iter{it}.png")
            diff_visualization(to_numpy(batch_2d[0, 0]), to_numpy(recon[0, 0]), save_path, title_suffix=f"Iter {it}")

curve_path = os.path.join(artifact_dir, f'training_curves_lr{lr}_bs{batch_size}.png')
plot_training_curves(loss_history, psnr_history, curve_path, lr, batch_size)