import torch
import yaml
from dc_gen.aecore.models.dc_ae import DCAE, build_dcae_cfg_from_variant

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

registry_path = "/cluster/home/kostfab1/DC-Gen/configs/models/registry.yaml"
model_key = "f32c128_shallow_narrow_last_adaptive"

with open(registry_path) as f:
    registry = yaml.safe_load(f)

variant = registry[model_key]
cfg = build_dcae_cfg_from_variant(variant)

model = DCAE(cfg).to(device=device, dtype=dtype)
model.eval()

B, C, D, H, W = 3, 1, 32, 32, 32
volume = torch.randn(B, C, D, H, W, dtype=dtype, device=device)

with torch.no_grad():
    latent_full = model.encoder(volume)
    recon_full = model.decoder(latent_full)

c_choices = list(range(16, cfg.latent_channels + 1, 4))
c_prime = c_choices[len(c_choices) // 2]

with torch.no_grad():
    latent_partial = model.encoder(volume, latent_channels=c_prime)
    recon_partial = model.decoder(latent_partial)

print("Input:", volume.shape)
print("Latent full:", latent_full.shape)
print("Recon full:", recon_full.shape)
print("Latent partial:", latent_partial.shape)
print("Recon partial:", recon_partial.shape)

print("OUT BLOCK:", type(model.encoder.project_out.op_list[0]))
print("IN BLOCK:", type(model.decoder.project_in))
