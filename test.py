import torch
from dc_gen.ae_model_zoo import DCAE_HF
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters, avoiding double-counting shared params.
    """
    seen = set()
    total = 0
    trainable = 0
    for param in module.parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def count_toplevel_only_parameters(model: torch.nn.Module, *submodules: torch.nn.Module) -> tuple[int, int]:
    """
    Count parameters that belong to the model but NOT to any of the given submodules.
    Useful for detecting params outside encoder/decoder.
    """
    sub_param_ids = set()
    for submodule in submodules:
        for param in submodule.parameters():
            sub_param_ids.add(id(param))

    total = 0
    trainable = 0
    seen = set()
    for param in model.parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        if pid not in sub_param_ids:
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
    return total, trainable


def format_param_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count:,} ({count / 1_000_000_000:.2f}B)"
    if count >= 1_000_000:
        return f"{count:,} ({count / 1_000_000:.2f}M)"
    if count >= 1_000:
        return f"{count:,} ({count / 1_000:.2f}K)"
    return f"{count:,}"


def summarize_model(model_name: str) -> dict:
    try:
        print(f"Loading {model_name} ... ", end="", flush=True)
        with torch.no_grad():
            model = DCAE_HF(model_name=model_name).to(device=device, dtype=dtype)
            model.eval()

        model_total, model_trainable = count_parameters(model)
        encoder_total, encoder_trainable = count_parameters(model.encoder)
        decoder_total, decoder_trainable = count_parameters(model.decoder)
        toplevel_total, toplevel_trainable = count_toplevel_only_parameters(
            model, model.encoder, model.decoder
        )

        # Sanity check: encoder + decoder + toplevel-only should equal model total
        accounted = encoder_total + decoder_total + toplevel_total
        if accounted != model_total:
            print(f"\n  WARNING: param count mismatch for {model_name}: "
                  f"encoder({encoder_total}) + decoder({decoder_total}) + toplevel({toplevel_total}) "
                  f"= {accounted} != model_total({model_total})")

        # Clean up immediately
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("done")

        return {
            "name": model_name,
            "model_total": model_total,
            "model_trainable": model_trainable,
            "encoder_total": encoder_total,
            "encoder_trainable": encoder_trainable,
            "decoder_total": decoder_total,
            "decoder_trainable": decoder_trainable,
            "toplevel_total": toplevel_total,
            "toplevel_trainable": toplevel_trainable,
        }
    except Exception as e:
        print(f"FAILED: {e}")
        return {"name": model_name, "error": str(e)}


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
    f"{'Decoder Total':>18} {'Decoder Trainable':>18} "
    f"{'TopLevel Total':>18} {'TopLevel Trainable':>18}"
)
print(header)
print("-" * len(header))

summaries = []
for name in comparison_model_names:
    summary = summarize_model(name)
    summaries.append(summary)

for summary in summaries:
    if "error" in summary:
        print(f"{summary['name']:<34} {'ERROR':>18} {summary.get('error', 'Unknown error')}")
        continue
    print(
        f"{summary['name']:<34} "
        f"{format_param_count(summary['model_total']):>18} {format_param_count(summary['model_trainable']):>18} "
        f"{format_param_count(summary['encoder_total']):>18} {format_param_count(summary['encoder_trainable']):>18} "
        f"{format_param_count(summary['decoder_total']):>18} {format_param_count(summary['decoder_trainable']):>18} "
        f"{format_param_count(summary['toplevel_total']):>18} {format_param_count(summary['toplevel_trainable']):>18}"
    )

print()

# ────────────────────────────────────────────────
# Detailed example with one model

model_name = "dc-ae-f32c32-in-1.0_3d-depth-last"

print(f"Loading detailed model: {model_name}")
with torch.no_grad():
    model = DCAE_HF(model_name=model_name).to(device=device, dtype=dtype)
    model.eval()

print(f"Detailed architecture for: {model_name}")
print(model)

# Fake 3D volume: [B, D, H, W] → add channel → [B, C=1, D, H, W]
B, D, H, W = 1, 4, 32, 32
volume = torch.randn(B, D, H, W, dtype=dtype, device=device)
if volume.ndim == 4:
    volume = volume.unsqueeze(1)  # → [1, 1, 4, 32, 32]

print("Volume.shape:", volume.shape)

# Forward pass
with torch.no_grad():
    latent = model.encoder(volume)
    latent_repr = (
        latent.shape
        if isinstance(latent, torch.Tensor)
        else [l.shape for l in latent] if isinstance(latent, (list, tuple)) else "???"
    )
    print(f"Input shape:       {volume.shape}")
    print(f"Latent shape:      {latent_repr}")

    recon = model.decoder(latent)
    print(f"Reconstruction shape: {recon.shape}")

# Shape check
if isinstance(recon, torch.Tensor) and recon.shape == volume.shape:
    print("→ Reconstruction shape matches input → OK")
else:
    print("→ Shape mismatch!")

# Cleanup
del model, volume, latent, recon
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
