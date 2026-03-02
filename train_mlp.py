"""
MLP Implicit Neural Representation (INR) baseline for CT volume overfitting.

Maps (z, y, x) coordinates → HU value, using Fourier feature positional encoding
to help the MLP learn high-frequency detail. This is the NeRF-style approach and
serves as a sanity-check upper bound: if the MLP can overfit well, the DC-AE
should too (with the advantage of a structured latent space).

Usage:
    python overfit_mlp.py
"""

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import math
from tqdm import tqdm

device = torch.device("cuda")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load volume
# ─────────────────────────────────────────────────────────────────────────────
file_path = "/cluster/projects/ac3t/data/ac3t_ct_rate/size_256/v13/final_hdfs_/nproj_491/ct_rate_train_batch_2_v13.hdf"
with h5py.File(file_path, "r") as f:
    key = list(f["Vol_full"].keys())[0]
    vol = f["Vol_full"][key][()]

print(f"Loaded: shape={vol.shape}, dtype={vol.dtype}, "
      f"min={vol.min():.1f}, max={vol.max():.1f}")

vol_f32 = np.clip(vol, -1000, 1000).astype(np.float32) / 1000.0   # [-1, 1]
D, H, W = vol_f32.shape

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build coordinate grid  (N, 3) in [-1, 1]
# ─────────────────────────────────────────────────────────────────────────────
zz = torch.linspace(-1, 1, D)
yy = torch.linspace(-1, 1, H)
xx = torch.linspace(-1, 1, W)
grid_z, grid_y, grid_x = torch.meshgrid(zz, yy, xx, indexing="ij")  # each (D,H,W)
coords = torch.stack([grid_z, grid_y, grid_x], dim=-1).reshape(-1, 3)  # (N, 3)
values = torch.from_numpy(vol_f32).reshape(-1, 1)                       # (N, 1)

N = coords.shape[0]
print(f"Total voxels: {N:,}  ({D}×{H}×{W})")

coords  = coords.to(device)
values  = values.to(device)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fourier feature encoding  (Tancik et al. 2020)
#    Projects coords to a higher-dim sinusoidal space so the MLP can learn
#    high-frequency detail without needing to be very deep.
# ─────────────────────────────────────────────────────────────────────────────
class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 3, num_frequencies: int = 128, sigma: float = 6.0):
        super().__init__()
        # Random Gaussian projection matrix, fixed (not learned)
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        proj = (2 * math.pi * x) @ self.B          # (..., num_frequencies)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)  # (..., 2*F)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MLP model
#    ~4.5 M parameters — large enough to memorise a 256³ volume
# ─────────────────────────────────────────────────────────────────────────────
class CTMLP(nn.Module):
    def __init__(
        self,
        num_frequencies: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 6,
        sigma: float = 6.0,
    ):
        super().__init__()
        self.ff = FourierFeatures(3, num_frequencies, sigma)
        feat_dim = self.ff.out_dim                   # 2 * num_frequencies

        layers = []
        in_dim = feat_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Tanh())                     # output in (-1, 1) matches [-1000, 1000] HU
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.ff(coords)
        return self.net(feats)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


model = CTMLP(num_frequencies=128, hidden_dim=512, num_layers=6, sigma=6.0)
model = model.to(device)
print(f"MLP parameters: {model.count_params():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training — mini-batch over voxels (whole volume doesn't fit in one pass)
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)

num_epochs   = 2000
batch_size   = 2 ** 17   # ~130k voxels per step, fast on A100/V100

pbar = tqdm(range(num_epochs))
for epoch in pbar:
    # Random mini-batch of voxels
    idx    = torch.randperm(N, device=device)[:batch_size]
    c_bat  = coords[idx]
    v_bat  = values[idx]

    optimizer.zero_grad()
    pred  = model(c_bat)
    loss  = F.mse_loss(pred, v_bat)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0 or epoch < 5:
        pbar.set_postfix({
            "mse":  f"{loss.item():.6f}",
            "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
        })

# ─────────────────────────────────────────────────────────────────────────────
# 6. Reconstruct full volume in chunks (all voxels, no gradient)
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
recon_flat = torch.empty(N, 1, device="cpu", dtype=torch.float32)
chunk = 2 ** 18   # ~260k at a time

with torch.no_grad():
    for start in range(0, N, chunk):
        end  = min(start + chunk, N)
        pred = model(coords[start:end]).float().cpu()
        recon_flat[start:end] = pred

recon_np = recon_flat.reshape(D, H, W).numpy()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Metrics
# ─────────────────────────────────────────────────────────────────────────────
gt_np      = vol_f32  # [-1, 1]
mse_norm   = float(np.mean((recon_np - gt_np) ** 2))
recon_hu   = (recon_np  * 1000.0).clip(-1000, 1000).astype(np.float32)
gt_hu      = (gt_np     * 1000.0).clip(-1000, 1000).astype(np.float32)
mse_hu     = float(np.mean((recon_hu - gt_hu) ** 2))
psnr       = 10 * math.log10((2000.0 ** 2) / mse_hu) if mse_hu > 0 else float("inf")

print(f"\n── MLP Overfit Results ──────────────────────────────")
print(f"  MSE (normalised [-1,1]) : {mse_norm:.6f}")
print(f"  MSE (HU)                : {mse_hu:.2f}")
print(f"  PSNR (HU, range 2000)  : {psnr:.2f} dB")
print(f"  Recon HU range          : [{recon_hu.min():.1f}, {recon_hu.max():.1f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Save NIfTI
# ─────────────────────────────────────────────────────────────────────────────
def save_nifti_float32(path: str, data: np.ndarray):
    assert data.dtype == np.float32
    dz, dy, dx = data.shape
    header = bytearray(348)
    struct.pack_into("<i",  header,   0, 348)
    struct.pack_into("<h",  header,  40, 3)
    struct.pack_into("<h",  header,  42, dx)
    struct.pack_into("<h",  header,  44, dy)
    struct.pack_into("<h",  header,  46, dz)
    struct.pack_into("<h",  header,  48, 1)
    struct.pack_into("<h",  header,  50, 1)
    struct.pack_into("<h",  header,  52, 1)
    struct.pack_into("<h",  header,  54, 1)
    struct.pack_into("<h",  header,  70, 16)
    struct.pack_into("<h",  header,  72, 32)
    struct.pack_into("<8f", header,  76, 1.0, 1.0, 1.0, 1.0, 0., 0., 0., 0.)
    struct.pack_into("<f",  header, 108, 352.0)
    struct.pack_into("<h",  header, 252, 1)
    header[344:348] = b'n+1\x00'
    with open(path, "wb") as f:
        f.write(header)
        f.write(b'\x00\x00\x00\x00')
        data.tofile(f)
    print(f"Saved: {path}  ({dz}×{dy}×{dx})")

save_nifti_float32("mlp_reconstruction.nii",  recon_hu)
save_nifti_float32("mlp_ground_truth.nii",    gt_hu)

# Save model checkpoint in case you want to query it later
torch.save({"model_state": model.state_dict(), "shape": (D, H, W)}, "mlp_overfit.pt")
print("Checkpoint saved: mlp_overfit.pt")