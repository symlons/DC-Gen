from dc_gen.ae_model_zoo import DCAE_HF
import torch
import torch.nn.functional as F

device = torch.device("cuda")

model_name = "dc-ae-f32c32-in-1.0"
dc_ae = DCAE_HF(model_name=model_name)

dc_ae = dc_ae.to(torch.bfloat16)
dc_ae = dc_ae.to(device).train()

print("Model parameter dtype:", next(dc_ae.parameters()).dtype)

optimizer = torch.optim.AdamW(
    dc_ae.parameters(),
    lr=1e-4,
    weight_decay=1e-2
)

x3d = torch.randn(1, 1, 64, 256, 256, device=device, dtype=torch.bfloat16)

optimizer.zero_grad()

latent3d = dc_ae.encode(x3d)
recon3d = dc_ae.decode(latent3d)

loss = F.mse_loss(recon3d, x3d)

loss.backward()

optimizer.step()

print("Loss:", loss.item())
print("Latent shape:", latent3d.shape)
print("Recon shape:", recon3d.shape)