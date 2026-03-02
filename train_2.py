import os
import math
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dc_gen.ae_model_zoo import DCAE_HF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_numpy
import nibabel as nib
import pandas as pd

cfg = {
    "device": "cuda",
    "out_dir": "3d_dc_ae_v3",
    "file_path": "/cluster/projects/ac3t/data/ac3t_ct_rate/size_256/v13/final_hdfs_/nproj_491/ct_rate_train_batch_2_v13.hdf",
    "batch_size": 1,
    "num_steps": 10000,
    "save_every": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "model_name": "dc-ae-f32c32-in-1.0",
    "hu_range": [-1000, 1000],
}
os.makedirs(cfg["out_dir"], exist_ok=True)
device = torch.device(cfg["device"])

with h5py.File(cfg["file_path"], "r") as f:
    key = list(f["Vol_full"].keys())[0]
    vol = f["Vol_full"][key][()]

D, H, W = vol.shape
vol_hu = vol.astype(np.float32).clip(*cfg["hu_range"])
x = torch.from_numpy((vol_hu + 1000.0)/2000.0).unsqueeze(0).unsqueeze(0).to(device, torch.bfloat16)

dc_ae = DCAE_HF(model_name=cfg["model_name"]).to(torch.bfloat16).to(device)
dc_ae = torch.compile(dc_ae)
dc_ae.train()

def save_volume_nifti(path, vol_hu):
    nii = nib.Nifti1Image(vol_hu.astype(np.float32), affine=np.eye(4))
    nib.save(nii, path)

def save_volume_slices_triplet(gt_vol, recon_vol, path):
    D = gt_vol.shape[0]
    fig, axes = plt.subplots(D, 3, figsize=(12, 4*D))
    for i in range(D):
        diff = recon_vol[i] - gt_vol[i]
        for ax, img, title, cmap in zip(axes[i], [gt_vol[i], recon_vol[i], diff], ["GT","Recon","Diff"], ["gray","gray","bwr"]):
            ax.imshow(img, cmap=cmap, vmin=-1000, vmax=1000)
            ax.axis('off'); ax.set_title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def compute_ssim(gt_vol, recon_vol):
    return float(np.mean([ssim_numpy(gt_vol[i], recon_vol[i], data_range=2000.0) for i in range(gt_vol.shape[0])]))

optimizer = torch.optim.AdamW(dc_ae.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_steps"], eta_min=1e-6)

losses, psnrs, ssims = [], [], []

dc_ae.eval()
with torch.no_grad():
    latent_init = dc_ae.encode(x)
    recon_init = dc_ae.decode(latent_init).squeeze(0).squeeze(0).float().cpu().numpy()
save_volume_nifti(os.path.join(cfg["out_dir"], "step_0000_gt.nii"), vol_hu)
save_volume_nifti(os.path.join(cfg["out_dir"], "step_0000_recon.nii"), recon_init)
save_volume_slices_triplet(vol_hu, recon_init, os.path.join(cfg["out_dir"], "step_0000_triplet.png"))

pbar = tqdm(range(1, cfg["num_steps"]+1))
for step in pbar:
    optimizer.zero_grad()
    latent = dc_ae.encode(x)
    recon = dc_ae.decode(latent)
    loss = F.mse_loss(recon, x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dc_ae.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())

    if step % cfg["save_every"] == 0 or step == cfg["num_steps"]:
        dc_ae.eval()
        with torch.no_grad():
            latent_eval = dc_ae.encode(x)
            recon_eval = dc_ae.decode(latent_eval).squeeze(0).squeeze(0).float().cpu().numpy()
        dc_ae.train()

        mse_val = float(np.mean((recon_eval - vol_hu)**2))
        psnr_val = 10 * math.log10(2000.0**2 / mse_val) if mse_val > 0 else float("inf")
        ssim_val = compute_ssim(vol_hu, recon_eval)

        psnrs.append(psnr_val)
        ssims.append(ssim_val)

        step_dir = os.path.join(cfg["out_dir"], f"step_{step:04d}")
        os.makedirs(step_dir, exist_ok=True)
        save_volume_nifti(os.path.join(step_dir,"gt.nii"), vol_hu)
        save_volume_nifti(os.path.join(step_dir,"recon.nii"), recon_eval)
        save_volume_slices_triplet(vol_hu, recon_eval, os.path.join(step_dir,"triplet.png"))

        tqdm.write(f"step {step}  MSE={mse_val:.2f}  PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}")

dc_ae.eval()
with torch.no_grad():
    latent_final = dc_ae.encode(x)
    final_np = dc_ae.decode(latent_final).squeeze(0).squeeze(0).float().cpu().numpy()
mse_final = float(np.mean((final_np - vol_hu)**2))
psnr_final = 10*math.log10(2000.0**2 / mse_final)
ssim_final = compute_ssim(vol_hu, final_np)

save_volume_nifti(os.path.join(cfg["out_dir"],"final_gt.nii"), vol_hu)
save_volume_nifti(os.path.join(cfg["out_dir"],"final_recon.nii"), final_np)
save_volume_slices_triplet(vol_hu, final_np, os.path.join(cfg["out_dir"],"final_triplet.png"))

df = pd.DataFrame({"step": range(cfg["save_every"], cfg["num_steps"]+1, cfg["save_every"]), "psnr": psnrs, "ssim": ssims})
df.to_csv(os.path.join(cfg["out_dir"],"metrics.csv"), index=False)

plt.figure(figsize=(6,4))
plt.plot(losses,label="MSE Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(cfg["out_dir"],"loss_curve.png"))
plt.close()

print(f"Final MSE: {mse_final:.2f}  PSNR: {psnr_final:.2f} dB  SSIM: {ssim_final:.4f}")
print(f"All outputs saved to: {cfg['out_dir']}/")