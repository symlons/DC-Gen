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

# ---------------- CONFIG ----------------
cfg = {
    "device": "cuda",
    "out_dir": "2d_dc_ae_v4_matplotlib",
    "file_path": "/cluster/projects/ac3t/data/ac3t_ct_rate/size_256/v13/final_hdfs_/nproj_491/ct_rate_train_batch_2_v13.hdf",
    "batch_size": 8,
    "num_steps": 10000,
    "save_every": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "model_name": "dc-ae-f32c32-in-1.0",
    "hu_range": [-1000, 1000],
}
os.makedirs(cfg["out_dir"], exist_ok=True)

# ---------------- DATA ----------------
with h5py.File(cfg["file_path"], "r") as f:
    key = list(f["Vol_full"].keys())[0]
    vol = f["Vol_full"][key][()]

D, H, W = vol.shape
assert H % 32 == 0 and W % 32 == 0
slice_indices = np.linspace(D//8, D-D//8, cfg["batch_size"], dtype=int)
gt_hu_batch = vol[slice_indices].astype(np.float32).clip(*cfg["hu_range"])
x = torch.from_numpy((gt_hu_batch + 1000.0)/2000.0).unsqueeze(1).to(device=cfg["device"], dtype=torch.bfloat16)

# ---------------- MODEL ----------------
dc_ae = DCAE_HF(model_name=cfg["model_name"]).to(torch.bfloat16).to(cfg["device"])
dc_ae = torch.compile(dc_ae)  # use TorchDynamo for speed
dc_ae.train()

# ---------------- HELPERS ----------------
def save_gray(path, img_hu, cmap="gray"):
    plt.figure(figsize=(4,4))
    plt.imshow(img_hu, cmap=cmap, vmin=cfg["hu_range"][0], vmax=cfg["hu_range"][1])
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_diff(path, diff_hu, cmap="bwr"):
    plt.figure(figsize=(4,4))
    plt.imshow(diff_hu, cmap=cmap, vmin=-abs(diff_hu).max(), vmax=abs(diff_hu).max())
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_triplet(gt, recon, diff, path):
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    for ax, img, title, cmap in zip(axes, [gt, recon, diff], ["Ground Truth","Reconstruction","Difference"], ["gray","gray","bwr"]):
        ax.imshow(img, cmap=cmap, vmin=cfg["hu_range"][0], vmax=cfg["hu_range"][1])
        ax.axis('off'); ax.set_title(title, fontsize=12)
    fig.subplots_adjust(top=0.85)
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_comparisons(gt_batch, recon_batch, step):
    step_dir = os.path.join(cfg["out_dir"], f"step_{step:04d}")
    os.makedirs(step_dir, exist_ok=True)
    B = gt_batch.shape[0]
    for i in range(B):
        gt_img = gt_batch[i]
        recon_img = recon_batch[i]
        diff_img = recon_img - gt_img
        save_gray(os.path.join(step_dir, f"slice{i:02d}_gt.png"), gt_img)
        save_gray(os.path.join(step_dir, f"slice{i:02d}_recon.png"), recon_img)
        save_diff(os.path.join(step_dir, f"slice{i:02d}_diff.png"), diff_img)
        save_triplet(gt_img, recon_img, diff_img, os.path.join(step_dir, f"slice{i:02d}_triplet.png"))
    # full batch montage
    montage = np.hstack([np.hstack([gt_batch[i], recon_batch[i], (recon_batch[i]-gt_batch[i])]) for i in range(B)])
    plt.figure(figsize=(12,B*4))
    plt.imshow(montage, cmap="gray", vmin=cfg["hu_range"][0], vmax=cfg["hu_range"][1])
    plt.axis('off'); plt.tight_layout(); plt.savefig(os.path.join(step_dir,"batch_montage.png"), dpi=150); plt.close()
    return step_dir

def compute_ssim_batch(gt_batch, recon_batch):
    return float(np.mean([ssim_numpy(gt_batch[i], recon_batch[i], data_range=2000.0) for i in range(gt_batch.shape[0])]))

# ---------------- TRAINING ----------------
optimizer = torch.optim.AdamW(dc_ae.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_steps"], eta_min=1e-6)

losses, psnrs, ssims = [], [], []

gt_tensor = x.float()

# Step 0: untrained model
dc_ae.eval()
with torch.no_grad():
    recon_init = dc_ae.decode(dc_ae.encode(x)).squeeze(1).float().cpu().numpy()
recon_init_hu = (recon_init * 2000.0 - 1000.0).clip(*cfg["hu_range"])
save_comparisons(gt_hu_batch, recon_init_hu, step=0)

# Training loop
pbar = tqdm(range(1, cfg["num_steps"]+1))
for step in pbar:
    optimizer.zero_grad()
    latent = dc_ae.encode(x)
    recon = dc_ae.decode(latent)
    loss = F.mse_loss(recon, x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dc_ae.parameters(),1.0)
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())

    if step % cfg["save_every"] == 0 or step == cfg["num_steps"]:
        dc_ae.eval()
        with torch.no_grad():
            recon_eval = dc_ae.decode(dc_ae.encode(x))
        dc_ae.train()
        recon_np = recon_eval.squeeze(1).float().cpu().numpy()
        recon_hu = (recon_np * 2000.0 - 1000.0).clip(*cfg["hu_range"])

        mse_val = float(np.mean((recon_hu - gt_hu_batch)**2))
        psnr_val = 10 * math.log10(2000.0**2 / mse_val) if mse_val>0 else float("inf")
        ssim_val = compute_ssim_batch(gt_hu_batch, recon_hu)

        psnrs.append(psnr_val)
        ssims.append(ssim_val)

        save_comparisons(gt_hu_batch, recon_hu, step=step)
        tqdm.write(f"step {step}  MSE={mse_val:.2f}  PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}")

# ---------------- FINAL ----------------
dc_ae.eval()
with torch.no_grad():
    final_np = dc_ae.decode(dc_ae.encode(x)).squeeze(1).float().cpu().numpy()
final_hu = (final_np * 2000.0 - 1000.0).clip(*cfg["hu_range"])
mse_final = float(np.mean((final_hu - gt_hu_batch)**2))
psnr_final = 10*math.log10(2000.0**2 / mse_final)
ssim_final = compute_ssim_batch(gt_hu_batch, final_hu)

# Save progress plots
plt.figure(figsize=(6,4)); plt.plot(losses, label="MSE Loss"); plt.xlabel("Step"); plt.ylabel("Loss"); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(cfg["out_dir"],"loss_curve.png")); plt.close()

plt.figure(figsize=(6,4)); plt.plot(range(cfg["save_every"], cfg["num_steps"]+1, cfg["save_every"]), psnrs, label="PSNR (dB)"); plt.xlabel("Step"); plt.ylabel("PSNR"); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(cfg["out_dir"],"psnr_progress.png")); plt.close()

plt.figure(figsize=(6,4)); plt.plot(range(cfg["save_every"], cfg["num_steps"]+1, cfg["save_every"]), ssims, label="SSIM"); plt.xlabel("Step"); plt.ylabel("SSIM"); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(cfg["out_dir"],"ssim_progress.png")); plt.close()

# Save logs
with open(os.path.join(cfg["out_dir"],"training_log.csv"),"w") as f:
    f.write("step,mse_loss,psnr,ssim\n")
    for i, l in enumerate(losses,1):
        ps = psnrs[i//cfg["save_every"]-1] if i % cfg["save_every"]==0 else ""
        ss = ssims[i//cfg["save_every"]-1] if i % cfg["save_every"]==0 else ""
        f.write(f"{i},{l:.8f},{ps},{ss}\n")

print(f"Final MSE: {mse_final:.2f}  PSNR: {psnr_final:.2f} dB  SSIM: {ssim_final:.4f}")
print(f"All outputs saved to: {cfg['out_dir']}/")