import torch
from dc_gen.ae_model_zoo import DCAE_HF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model_name = "dc-ae-f32c32-sana-1.0"

with torch.no_grad():
    model = DCAE_HF(model_name=model_name).to(device=device, dtype=dtype)
    model.eval()

B, C, H, W = 3, 3, 32, 32
volume = torch.randn(B, C, H, W, dtype=dtype, device=device)
# volume = volume.unsqueeze(1)

with torch.no_grad():
    latent = model.encoder(volume)
    recon = model.decoder(latent)

print("Input:", volume.shape)
print("Latent:", latent.shape if isinstance(latent, torch.Tensor) else type(latent))
print("Recon:", recon.shape)
print(model)