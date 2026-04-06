import torch
import numpy as np
from pathlib import Path
import nibabel as nib

from flow.latent_dataset import LatentTensorDataset
from dc_gen.ae_model_zoo import create_dc_ae_model_cfg
from dc_gen.aecore.models.dc_ae import DCAE

device = "cuda" if torch.cuda.is_available() else "cpu"

latent_dir = "/cluster/projects/2025_stmd_VT_diff/latents/latents_v04_shallow_phase1_valid_v2"
ckpt_path = "/cluster/projects/2025_stmd_VT_diff/DC-GEN_backup/checkpoints_v04_shallow_phase1/checkpoint_iter141000.pt"

dataset = LatentTensorDataset(root_dir=latent_dir)
x = dataset[0]["image"]

model_cfg = create_dc_ae_model_cfg("dc-ae-f32c32-in-1.0_3d-shallow")
autoencoder = DCAE(model_cfg)

ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt.get("model_state_dict", ckpt)
autoencoder.load_state_dict(state_dict)

autoencoder = autoencoder.to(device).eval()
with torch.no_grad(): recon = autoencoder.decode(x.unsqueeze(0).to(device))[0]

vol = recon[0].cpu().numpy()
nii = nib.Nifti1Image(vol, affine=np.eye(4))
nib.save(nii, "decoded.nii.gz")
print("latent shape:", x.shape)
print("latent stats:", x.mean().item(), x.std().item())
print("recon shape:", recon.shape)
