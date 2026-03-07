import torch
from dc_gen.ae_model_zoo import DCAE_HF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "dc-ae-f32c32-in-1.0"
model = DCAE_HF(model_name=model_name).to(dtype=torch.bfloat16, device=device)
model.eval()

# Fake 3D volume: [B, D, H, W]
B, D, H, W = 1, 128, 512, 512
volume = torch.randn(B, D, H, W, dtype=torch.bfloat16, device=device)

# Add channel dimension if needed: [B, C=1, D, H, W]
if volume.ndim == 4:
    volume = volume.unsqueeze(1)

print("Volume.shape", volume.shape)
# Forward pass through encoder and decoder
with torch.no_grad():
    latent = model.encoder(volume)
#recon = model.decoder(latent)

print(f"Input shape: {volume.shape}")
print(f"Latent shape: {latent.shape if isinstance(latent, torch.Tensor) else [l.shape for l in latent]}")
print(f"Reconstruction shape: {recon.shape}")
