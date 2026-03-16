import torch
from dc_gen.ae_model_zoo import DCAE_HF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def format_param_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count:,} ({count / 1_000_000_000:.2f}B)"
    if count >= 1_000_000:
        return f"{count:,} ({count / 1_000_000:.2f}M)"
    if count >= 1_000:
        return f"{count:,} ({count / 1_000:.2f}K)"
    return f"{count:,}"


def summarize_model(model_name: str) -> dict[str, str | int]:
    model = DCAE_HF(model_name=model_name)
    model.eval()
    model_total, model_trainable = count_parameters(model)
    encoder_total, encoder_trainable = count_parameters(model.encoder)
    decoder_total, decoder_trainable = count_parameters(model.decoder)
    return {
        "name": model_name,
        "model_total": model_total,
        "model_trainable": model_trainable,
        "encoder_total": encoder_total,
        "encoder_trainable": encoder_trainable,
        "decoder_total": decoder_total,
        "decoder_trainable": decoder_trainable,
    }


comparison_model_names = [
    "dc-ae-f32c32-in-1.0_3d",
    "dc-ae-f32c32-in-1.0_3d-depth-last",
    "dc-ae-f32c32-sana-1.0",
    "dc-ae-f32c32-sana-1.1",
    "dc-ae-lite-f32c32-sana-1.1",
]

print("Parameter comparison")
header = (
    f"{'Model':<34} "
    f"{'Model Total':>18} {'Model Trainable':>18} "
    f"{'Encoder Total':>18} {'Encoder Trainable':>18} "
    f"{'Decoder Total':>18} {'Decoder Trainable':>18}"
)
print(header)
print("-" * len(header))
for summary in [summarize_model(name) for name in comparison_model_names]:
    print(
        f"{summary['name']:<34} "
        f"{format_param_count(summary['model_total']):>18} {format_param_count(summary['model_trainable']):>18} "
        f"{format_param_count(summary['encoder_total']):>18} {format_param_count(summary['encoder_trainable']):>18} "
        f"{format_param_count(summary['decoder_total']):>18} {format_param_count(summary['decoder_trainable']):>18}"
    )

print()

model_name = "dc-ae-f32c32-in-1.0_3d-depth-last"
model = DCAE_HF(model_name=model_name).to(dtype=torch.bfloat16, device=device)
model.eval()

print(f"Detailed architecture for: {model_name}")
print(model)

# Fake 3D volume: [B, D, H, W]
B, D, H, W = 1, 4, 32, 32
volume = torch.randn(B, D, H, W, dtype=torch.bfloat16, device=device)

# Add channel dimension if needed: [B, C=1, D, H, W]
if volume.ndim == 4:
    volume = volume.unsqueeze(1)

print("Volume.shape", volume.shape)
# Forward pass through encoder and decoder
with torch.no_grad():
    latent = model.encoder(volume)
    print(f"Input shape: {volume.shape}")
    print(f"Latent shape: {latent.shape if isinstance(latent, torch.Tensor) else [l.shape for l in latent]}")
    recon = model.decoder(latent)
    print(f"Reconstruction shape: {recon.shape}")
