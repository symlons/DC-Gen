from dc_gen.ae_model_zoo import DCAE_HF
import torch

device = torch.device("cuda")

model_name = "dc-ae-f32c32-in-1.0"
dc_ae = DCAE_HF(model_name=model_name)
dc_ae = dc_ae.to(device).eval()

# 2D variant [B, C, H, W]
# x2d = torch.randn(1, 3, 1024, 1024).to(device)

# latent2d = dc_ae.encode(x2d)
# print("2D Latent shape:", latent2d.shape)

# recon2d = dc_ae.decode(latent2d)
# print("2D Reconstructed shape:", recon2d.shape)

# 3D variant [B, C, D, H, W]
# x3d = torch.randn(1, 3, 2, 1024, 1024).to(device)
x3d = torch.randn(1, 3, 16, 512, 512).to(device)

latent3d = dc_ae.encode(x3d)
print("3D Latent shape:", latent3d.shape)

recon3d = dc_ae.decode(latent3d)
print("3D Reconstructed shape:", recon3d.shape)