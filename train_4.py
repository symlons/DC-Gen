import os
import math
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from tqdm import tqdm
from dc_gen.ae_model_zoo import DCAE_HF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import time

torch.set_float32_matmul_precision("medium")

cfg = {
    "device": "cuda",
    "out_dir": "3d_dc_ae_v3_pl_2",
    "file_path": "/cluster/projects/ac3t/data/ac3t_ct_rate/size_256/v13/final_hdfs_/nproj_491/ct_rate_train_batch_2_v13.hdf",
    "batch_size": 16,
    "num_steps": 100000,
    "save_every": 1000,
    "learning_rate": 1e-5,
    "weight_decay": 1e-2,
    "model_name": "dc-ae-f32c32-in-1.0",
    "hu_range": [-1000, 1000],
    "patch_depth": 8,
    "wandb_project": "3d_AE",
}
os.makedirs(cfg["out_dir"], exist_ok=True)

# class CTVolumeDataset(Dataset):
#     def __init__(self, file_path, hu_range, patch_depth):
#         self.file_path = file_path
#         self.hu_range = hu_range
#         self.patch_depth = patch_depth
#         self.volumes = []
#         with h5py.File(file_path, "r") as f:
#             for key in f["Vol_full"].keys():
#                 vol = f["Vol_full"][key][()]
#                 vol_hu = vol.astype(np.float32).clip(*hu_range)
#                 self.volumes.append(torch.from_numpy((vol_hu + 1000.0)/2000.0).unsqueeze(0))
#         self.total_slices = sum(v.shape[1] - patch_depth + 1 for v in self.volumes)
#         self.D, self.H, self.W = self.volumes[0].shape[1:]

#     def __len__(self):
#         return self.total_slices

#     def __getitem__(self, idx):
#         vol_idx = 0
#         slice_idx = idx
#         while slice_idx >= self.volumes[vol_idx].shape[1] - self.patch_depth + 1:
#             slice_idx -= self.volumes[vol_idx].shape[1] - self.patch_depth + 1
#             vol_idx += 1
#         vol = self.volumes[vol_idx]
#         patch = vol[:, slice_idx:slice_idx+self.patch_depth, :, :]
#         return patch.to(torch.bfloat16)

class CTVolumeDataset(Dataset):
    def __init__(self, file_path, hu_range, patch_depth):
        self.file_path = file_path
        self.hu_range = hu_range
        self.patch_depth = patch_depth
        self.volumes = []
        with h5py.File(file_path, "r") as f:
            for key in f["Vol_full"].keys():
                vol = f["Vol_full"][key][()]
                vol_hu = vol.astype(np.float32).clip(*hu_range)
                vol_tensor = torch.from_numpy((vol_hu + 1000.0)/2000.0).unsqueeze(0)
                D = vol_tensor.shape[1]
                start_idx = (D - patch_depth) // 2
                patch = vol_tensor[:, start_idx:start_idx+patch_depth, :, :]
                self.volumes.append(patch.to(torch.bfloat16))
        self.D, self.H, self.W = self.volumes[0].shape[1:]

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        return self.volumes[idx]

def save_volume_nifti(path, vol_hu):
    nii = nib.Nifti1Image(vol_hu.astype(np.float32), affine=np.eye(4))
    nib.save(nii, path)

def save_volume_slices_triplet(gt_vol, recon_vol, path):
    D = gt_vol.shape[0]
    fig, axes = plt.subplots(D, 2, figsize=(8, 4*D))
    for i in range(D):
        for ax, img, title, cmap in zip(axes[i], [gt_vol[i], recon_vol[i]], ["GT","Recon"], ["gray","gray"]):
            ax.imshow(img, cmap=cmap, vmin=-1000, vmax=1000)
            ax.axis('off'); ax.set_title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def compute_psnr(gt, pred, data_range=2000.0):
    gt_det = gt.float() if isinstance(gt, torch.Tensor) else torch.tensor(gt, dtype=torch.float32)
    pred_det = pred.float() if isinstance(pred, torch.Tensor) else torch.tensor(pred, dtype=torch.float32)
    mse = float(torch.mean((gt_det - pred_det) ** 2).detach())
    return 10 * math.log10((data_range ** 2) / mse) if mse > 0 else float("inf")

class LitDCAE(pl.LightningModule):
    def __init__(self, model_name, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.model = DCAE_HF(model_name=model_name)
        self.lr = learning_rate
        self.wd = weight_decay
        self._batch_start_time = None
        self._last_step_throughput = None
        self._warmup_steps = 20
        self._step_count = 0

    def forward(self, x):
        latent = self.model.encode(x)
        return self.model.decode(latent)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = getattr(self, "_batch_start_time", None)
        if start is None:
            return
        step_time = time.time() - start
        world_size = int(self.trainer.world_size) if self.trainer is not None else 1
        global_batch = batch.shape[0] * max(1, world_size)
        throughput = global_batch / step_time if step_time > 0 else float("inf")
        self._last_step_throughput = throughput
        self._step_count += 1
        if self._step_count > self._warmup_steps:
            self.log("train_throughput", throughput, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        recon = self(batch)
        loss = F.l1_loss(recon.float(), batch.float())
        psnr_val = compute_psnr(batch.float()*2000-1000, recon.float()*2000-1000)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_psnr", psnr_val, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["num_steps"], eta_min=1e-6)
        return [opt], [sched]

class SaveIntermediateCallback(Callback):
    def __init__(self, out_dir, save_every=1000):
        self.out_dir = out_dir
        self.save_every = save_every
        self.psnrs = []
        self._step_counter = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._step_counter += 1
        step = self._step_counter
        if step % self.save_every == 0 or step == cfg["num_steps"]:
            pl_module.eval()
            with torch.no_grad():
                recon = pl_module(batch).float().cpu()
                gt = batch.float().cpu()
                loss_val = float(F.l1_loss(recon, gt).detach())
                psnr_val = compute_psnr(gt*2000-1000, recon*2000-1000)
                throughput = getattr(pl_module, "_last_step_throughput", None)
                self.psnrs.append(psnr_val)

                pl_module.log("intermediate_loss", loss_val, on_step=True, sync_dist=True)
                pl_module.log("intermediate_psnr", psnr_val, on_step=True, sync_dist=True)

                if trainer.is_global_zero:
                    step_dir = os.path.join(self.out_dir, f"step_{step:04d}")
                    os.makedirs(step_dir, exist_ok=True)
                    save_volume_nifti(os.path.join(step_dir,"gt.nii"), (gt[0,0]*2000-1000).detach().numpy())
                    save_volume_nifti(os.path.join(step_dir,"recon.nii"), (recon[0,0]*2000-1000).detach().numpy())
                    save_volume_slices_triplet(
                        (gt[0,0]*2000-1000).detach().numpy(),
                        (recon[0,0]*2000-1000).detach().numpy(),
                        os.path.join(step_dir,"triplet.png")
                    )
                    tqdm.write(f"Step {step} L1={loss_val:.2f} PSNR={psnr_val:.2f} dB Throughput={throughput:.2f} samples/s")
            pl_module.train()

dataset = CTVolumeDataset(cfg["file_path"], cfg["hu_range"], cfg["patch_depth"])
dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

wandb_logger = WandbLogger(
    project=cfg["wandb_project"],
    log_model=False,
    id="f5ckz9ro",
    resume="allow"
)

ckpt_path = '/cluster/home/kostfab1/DC-Gen/3d_dc_ae_v3_pl/last.ckpt'
model = LitDCAE.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    model_name=cfg["model_name"],
    learning_rate=cfg["learning_rate"],
    weight_decay=cfg["weight_decay"]
)

# Compile the model for faster training
model = torch.compile(model)

checkpoint_cb = ModelCheckpoint(
    dirpath=cfg["out_dir"],
    filename="dc_ae-{step}",
    save_last=True,
    every_n_train_steps=cfg["save_every"],
)

save_cb = SaveIntermediateCallback(cfg["out_dir"], save_every=cfg["save_every"])

trainer = pl.Trainer(
    accelerator="gpu",
    devices="auto",
    strategy="ddp",
    precision="bf16-true",
    max_steps=cfg["num_steps"],
    callbacks=[checkpoint_cb, save_cb],
    log_every_n_steps=10,
    logger=wandb_logger,
)

trainer.fit(model, dataloader, ckpt_path=ckpt_path)

df = pd.DataFrame({
    "step": list(range(cfg["save_every"], cfg["num_steps"]+1, cfg["save_every"])),
    "psnr": save_cb.psnrs
})
df.to_csv(os.path.join(cfg["out_dir"],"metrics.csv"), index=False)