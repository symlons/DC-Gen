import torch
from basics import resolve_autocast_dtype

def print_config_summary(cfg, device):
    resolved_autocast_dtype = resolve_autocast_dtype(cfg, device)
    print("Config summary")
    print(f"  Experiment   : {cfg.experiment.name}")
    if cfg.experiment.description:
        print(f"  Description  : {cfg.experiment.description}")
    print(f"  Dims         : {cfg.dims}")
    print(f"  Model        : {cfg.model.name}")
    print(f"  Compile      : {cfg.model.compile}")
    print(f"  Dataset      : {cfg.dataset.name}")
    print(f"  Groups       : {cfg.dataset.group_names}")
    print(f"  HDF Path     : {cfg.paths.hdf_path}")
    print(f"  Save Dir     : {cfg.paths.save_dir}")
    print(f"  Checkpoints  : {cfg.paths.checkpoint_dir}")
    print(f"  Resize H/W   : {cfg.pipeline.resize_hw}")
    print(f"  Resize Depth : {cfg.pipeline.resize_depth}")
    print(f"  N Slices     : {cfg.pipeline.n_slices}")
    print(f"  Clip In      : {cfg.pipeline.clip_input_range}")
    print(f"  Norm Mode    : {cfg.pipeline.normalize_mode}")
    print(f"  Norm Out     : {cfg.pipeline.normalize_output_range}")
    print(f"  Norm In      : {cfg.pipeline.normalize_input_range}")
    print(f"  Epochs       : {cfg.training.num_epochs}")
    print(f"  Batch Size   : {cfg.training.batch_size}")
    print(f"  Num Workers  : {cfg.training.num_workers}")
    print(f"  Persistent Workers: {cfg.training.persistent_workers}")
    print(f"  DType        : {cfg.training.dtype}")
    print(f"  Autocast     : {cfg.training.use_autocast}")
    print(
        f"  AMP DType    : {cfg.training.autocast_dtype} -> {resolved_autocast_dtype}"
    )
    print(f"  LR           : {cfg.hparams.learning_rate}")
    print(f"  Weight Decay : {cfg.hparams.weight_decay}")
    print(f"  Loss         : {cfg.objective.loss_fn}")
    print(f"  Perc Weight  : {cfg.objective.perceptual_weight}")
    print(f"  Detail Weight: {cfg.objective.detail_weight}")
    if cfg.objective.gan_weight > 0:
        print(f"  GAN Weight   : {cfg.objective.gan_weight}")
        print(f"  GAN Loss Type: {cfg.objective.gan_loss_type}")
        print(f"  GAN Patch    : {cfg.objective.gan_patch_size}")
        print(f"  GAN Disc D   : {cfg.objective.gan_discriminator_steps}")
    print(f"  WandB        : {cfg.logging.wandb}")
    print(f"  Viz Every    : {cfg.logging.viz_every}")
    print(f"  Validate Every: {cfg.logging.validate_every}")
    print(f"  Val Max Batch : {cfg.logging.val_max_batches}")
    print()

def print_param_group_modules(model, param_groups):
    named_params = dict(model.named_parameters())

    id_to_module = {}
    for name, param in named_params.items():
        module_name = ".".join(name.split(".")[:-1])
        id_to_module[id(param)] = module_name

    total_params = sum(p.numel() for p in model.parameters())
    train_params = set(p for g in param_groups for p in g["params"])
    trainable_params = sum(p.numel() for p in train_params)

    if trainable_params == 0:
        raise ValueError("No trainable parameters selected")

    pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    print("Parameter summary:")
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")
    print(f"  Trainable %      : {pct:.4f}%")

    print("\nParameter groups (module structure):")

    modules_dict = dict(model.named_modules())

    for i, group in enumerate(param_groups):
        params = group["params"]
        num_elements = sum(p.numel() for p in params)
        group_pct = 100.0 * num_elements / total_params if total_params > 0 else 0.0

        print(f"\nGroup {i}: {num_elements:,} elements ({group_pct:.4f}%)")

        printed_modules = set()

        for p in params:
            module_name = id_to_module.get(id(p))
            if module_name is None:
                continue

            parts = module_name.split(".")
            key = ".".join(parts[:2])

            if key in printed_modules:
                continue

            printed_modules.add(key)

            try:
                submodule = modules_dict[key]
                print(f"\n--- {key} ---")
                print(submodule)
            except KeyError:
                print(f"[WARNING] Module not found: {key}")


def tensor_stats_dict(name: str, tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.detach().float()
    quantiles = torch.quantile(
        tensor.flatten(),
        torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=tensor.device),
    )
    return {
        f"{name}/mean": tensor.mean().item(),
        f"{name}/std": tensor.std(unbiased=False).item(),
        f"{name}/min": tensor.min().item(),
        f"{name}/max": tensor.max().item(),
        f"{name}/q01": quantiles[0].item(),
        f"{name}/q05": quantiles[1].item(),
        f"{name}/median": quantiles[2].item(),
        f"{name}/q95": quantiles[3].item(),
        f"{name}/q99": quantiles[4].item(),
    }
