import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from basics import get_autocast_ctx, move_batch, resolve_device, torch_dtype
from flow.dit import DiT, DiT_models
from flow.flow_config import TrainDiT3DConfig
from flow.flow_utils import validate_and_finalize_config
from flow.latent_dataset import infer_latent_shape
from flow.rectified_flow import RectifiedFlowObjective
from flow.transformer_engine import build_fp8_recipe, fp8_autocast_context, wrap_linears
from flow.ema import create_ema_model, update_ema_warmup
from flow.inspection import get_grad_norm
from train_data import make_dataloader, make_dataset


def load_config(path: str) -> TrainDiT3DConfig:
    cfg = OmegaConf.structured(TrainDiT3DConfig)
    yaml_cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(cfg, yaml_cfg)
    return validate_and_finalize_config(OmegaConf.to_object(cfg))


def build_model(cfg: TrainDiT3DConfig, in_channels: int) -> torch.nn.Module:
    num_classes = max(1, cfg.model.num_classes)
    class_dropout_prob = max(0.0, cfg.model.class_dropout_prob)
    if cfg.model.variant != "custom":
        return DiT_models[cfg.model.variant](num_classes=num_classes, class_dropout_prob=class_dropout_prob)
    return DiT(
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        patch_size=cfg.model.patch_size,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
    )


def percentile(values, q):
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def parse_spatial_shape(value: str) -> tuple[int, int, int]:
    value = value.strip().replace("[", "").replace("]", "").replace("x", ",")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected spatial shape with 3 dims, got {value!r}")
    shape = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"Spatial shape dims must be positive, got {shape}")
    return shape


def shape_label(spatial_shape: tuple[int, int, int] | None) -> str:
    if spatial_shape is None:
        return "dataset"
    return "x".join(str(dim) for dim in spatial_shape)


def run_variant(
    label: str,
    config_path: str,
    warmup: int,
    timed: int,
    max_batches: int | None,
    synthetic_shape: tuple[int, int, int] | None = None,
    batch_size_override: int | None = None,
):
    cfg = load_config(config_path)
    if batch_size_override is not None:
        cfg.training.batch_size = batch_size_override
    device = resolve_device(0)
    model_dtype = torch_dtype(cfg.training.dtype)

    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)

    train_dataset = None
    loader = None
    if synthetic_shape is None:
        train_dataset = make_dataset(cfg, "train")
        loader = make_dataloader(train_dataset, cfg, sampler=None, shuffle=False)
        in_channels, spatial_shape = infer_latent_shape(
            train_dataset,
            expected_in_channels=cfg.model.in_channels,
            expected_input_size=tuple(cfg.model.input_size) if cfg.model.input_size else None,
        )
        max_steps = len(loader) * cfg.training.num_epochs
    else:
        in_channels = cfg.model.in_channels
        spatial_shape = synthetic_shape
        max_steps = None

    model = build_model(cfg, in_channels).to(device=device, dtype=model_dtype)
    fp8_recipe = None
    if cfg.transformer_engine.enabled:
        fp8_recipe = build_fp8_recipe(cfg)
        if cfg.transformer_engine.replace_linears:
            model = wrap_linears(model)
    ema_model = create_ema_model(model).to(device=device, dtype=model_dtype) if cfg.training.ema_decay is not None else None
    global_step = 0
    if cfg.training.compile:
        model = torch.compile(model)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    objective = RectifiedFlowObjective(
        device=device,
        eps=cfg.objective.eps,
        mode=cfg.objective.mode,
        mu=cfg.objective.mu,
        sigma=cfg.objective.sigma,
        beta_a=cfg.objective.beta_a,
        beta_b=cfg.objective.beta_b,
        latent_mean=cfg.objective.latent_mean,
        latent_std=cfg.objective.latent_std,
        tweo_enabled=cfg.tweo.enabled,
        tweo_weight=cfg.tweo.weight,
        tweo_tau=cfg.tweo.tau,
        tweo_power=cfg.tweo.power,
        tweo_eps=cfg.tweo.eps,
        tweo_schedule=cfg.tweo.schedule,
        log_block_activations=cfg.tweo.log_activations,
        adaptive_latent_enabled=cfg.adaptive_latent.enabled,
        adaptive_latent_min_channels=cfg.adaptive_latent.min_channels,
        adaptive_latent_step=cfg.adaptive_latent.step,
        max_steps=max_steps,
    )

    total_needed = warmup + timed
    if max_batches is not None:
        total_needed = min(total_needed, max_batches)
    if total_needed <= warmup:
        raise ValueError("Need at least one timed batch")
    timed_steps = total_needed - warmup

    data_iter = iter(loader) if loader is not None else None
    rows = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for step_idx in range(total_needed):
        if synthetic_shape is None:
            assert data_iter is not None and loader is not None
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = move_batch(batch, device, model_dtype)
        else:
            batch = {
                "image": torch.randn(
                    (cfg.training.batch_size, in_channels, *synthetic_shape),
                    device=device,
                    dtype=model_dtype,
                )
            }
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        with get_autocast_ctx(cfg, device):
            with fp8_autocast_context(cfg.transformer_engine.enabled and cfg.transformer_engine.fp8_autocast, fp8_recipe):
                outputs = objective.compute_loss(model, batch["image"], global_step=global_step)
                loss = outputs["loss"]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        grad_norm = get_grad_norm(model.parameters(), cfg.training.grad_clip_norm)
        optimizer.step()
        if ema_model is not None:
            update_ema_warmup(ema_model, model, global_step + 1, decay=cfg.training.ema_decay, warmup_steps=cfg.training.ema_warmup_steps)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        if step_idx >= warmup:
            rows.append({
                "step_ms": (t4 - t0) * 1000.0,
                "zero_ms": (t1 - t0) * 1000.0,
                "forward_ms": (t2 - t1) * 1000.0,
                "backward_ms": (t3 - t2) * 1000.0,
                "opt_ema_clip_ms": (t4 - t3) * 1000.0,
                "loss": float(loss.detach().item()),
                "flow_loss": float(outputs["flow_loss"].detach().item()),
                "tweo_loss": float(outputs["tweo_loss"].detach().item()),
                "tweo_weight": float(outputs["tweo_weight"].detach().item()),
                "block_abs_max": float(outputs["block_activation_abs_max"].detach().item()),
                "block_abs_mean": float(outputs["block_activation_abs_mean"].detach().item()),
                "latent_c_prime": float(outputs["latent_c_prime"].detach().item()),
                "grad_norm": float(grad_norm),
            })
            print(
                f"[{label}] timed_step={len(rows)}/{timed_steps} global_step={global_step} "
                f"step_ms={rows[-1]['step_ms']:.2f} loss={rows[-1]['loss']:.4f} "
                f"mem_alloc={torch.cuda.max_memory_allocated()/1024**3 if device.type == 'cuda' else 0:.2f}GiB",
                flush=True,
            )
        global_step += 1

    step_ms = [r["step_ms"] for r in rows]
    fwd_ms = [r["forward_ms"] for r in rows]
    bwd_ms = [r["backward_ms"] for r in rows]
    opt_ms = [r["opt_ema_clip_ms"] for r in rows]
    batch_size = int(cfg.training.batch_size)
    result = {
        "label": label,
        "config": config_path,
        "experiment": cfg.experiment.name,
        "batch_size": batch_size,
        "model": cfg.model.variant,
        "compile": bool(cfg.training.compile),
        "training_dtype": cfg.training.dtype,
        "autocast_dtype": cfg.training.autocast_dtype,
        "tweo_enabled": bool(cfg.tweo.enabled),
        "tweo_weight": float(cfg.tweo.weight),
        "adaptive_latent_enabled": bool(cfg.adaptive_latent.enabled),
        "adaptive_latent_min_channels": int(cfg.adaptive_latent.min_channels),
        "adaptive_latent_step": int(cfg.adaptive_latent.step),
        "transformer_engine_enabled": bool(cfg.transformer_engine.enabled),
        "fp8_autocast": bool(cfg.transformer_engine.fp8_autocast),
        "timed_steps": len(rows),
        "warmup_steps": warmup,
        "start_global_step": int(global_step - total_needed),
        "end_global_step": int(global_step - 1),
        "step_ms_mean": float(np.mean(step_ms)),
        "step_ms_median": float(np.median(step_ms)),
        "step_ms_p10": percentile(step_ms, 10),
        "step_ms_p90": percentile(step_ms, 90),
        "samples_per_sec_mean": float(batch_size * 1000.0 / np.mean(step_ms)),
        "forward_ms_mean": float(np.mean(fwd_ms)),
        "backward_ms_mean": float(np.mean(bwd_ms)),
        "opt_ema_clip_ms_mean": float(np.mean(opt_ms)),
        "loss_mean": float(np.mean([r["loss"] for r in rows])),
        "flow_loss_mean": float(np.mean([r["flow_loss"] for r in rows])),
        "tweo_loss_mean": float(np.mean([r["tweo_loss"] for r in rows])),
        "block_abs_max_mean": float(np.mean([r["block_abs_max"] for r in rows])),
        "block_abs_mean_mean": float(np.mean([r["block_abs_mean"] for r in rows])),
        "peak_mem_alloc_gib": float(torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else None,
        "peak_mem_reserved_gib": float(torch.cuda.max_memory_reserved() / 1024**3) if device.type == "cuda" else None,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "profile_case": shape_label(synthetic_shape),
        "synthetic": synthetic_shape is not None,
        "spatial_shape": list(spatial_shape),
        "in_channels": int(in_channels),
    }

    del model, optimizer, ema_model, objective, loader, train_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8-tweo-config", default="configs/experiments/flow_fp8_tweo.yaml")
    parser.add_argument("--bf16-config", default="configs/experiments/flow_bf16_baseline.yaml")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--synthetic-shapes",
        nargs="*",
        default=None,
        help="Optional D,H,W latent shapes to profile with synthetic batches, e.g. 8,16,16 16,16,16.",
    )
    parser.add_argument("--out", default="logs/profile_train_compare/results.json")
    args = parser.parse_args()

    results = []
    synthetic_shapes = None
    if args.synthetic_shapes:
        synthetic_shapes = [parse_spatial_shape(value) for value in args.synthetic_shapes]
    cases = synthetic_shapes or [None]
    comparisons = []
    for spatial_shape in cases:
        case = shape_label(spatial_shape)
        fp8_result = run_variant(
            "fp8_tweo_te",
            args.fp8_tweo_config,
            args.warmup,
            args.timed,
            args.max_batches,
            synthetic_shape=spatial_shape,
            batch_size_override=args.batch_size,
        )
        bf16_result = run_variant(
            "bf16_baseline",
            args.bf16_config,
            args.warmup,
            args.timed,
            args.max_batches,
            synthetic_shape=spatial_shape,
            batch_size_override=args.batch_size,
        )
        results.extend([fp8_result, bf16_result])
        comparisons.append({
            "profile_case": case,
            "fp8_tweo_vs_bf16_speedup": bf16_result["step_ms_mean"] / fp8_result["step_ms_mean"],
            "fp8_tweo_step_ms_delta_pct": (fp8_result["step_ms_mean"] / bf16_result["step_ms_mean"] - 1.0) * 100.0,
            "fp8_tweo_mem_alloc_delta_pct": (fp8_result["peak_mem_alloc_gib"] / bf16_result["peak_mem_alloc_gib"] - 1.0) * 100.0,
            "fp8_tweo_mem_reserved_delta_pct": (fp8_result["peak_mem_reserved_gib"] / bf16_result["peak_mem_reserved_gib"] - 1.0) * 100.0,
        })

    payload = {"results": results, "comparisons": comparisons}
    if len(comparisons) == 1:
        payload["comparison"] = comparisons[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n=== PROFILE SUMMARY ===")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
