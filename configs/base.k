import .config

cfg: config.Config {
    dims = "2d"
    dataset: config.Dataset {
        name = "CTVolume"
        group_names = ["Vol_full"]
    }
    paths: config.Paths {
        hdf_path = ""
        checkpoint_dir = ""
        save_dir = ""
    }
    pipeline: config.Pipeline { resize_hw = [64, 64] }
    training: config.Training {
        num_epochs = 5
        batch_size = 1
        shuffle_data = False
        num_workers = 0
        pin_memory = False
        dtype = "float32"
        resume_from_checkpoint = False
        checkpoint_every = 100
        max_checkpoints = 4
        use_autocast = True
    }
    objective: config.Objective { loss_fn = "l1", perceptual_weight = 0.25 }
    hparams: config.HParams { learning_rate = 1e-5, weight_decay = 1e-2 }
    model: config.Model { name = "dc-ae-f32c32-in-1.0_2d", compile = True }
    logging: config.Logging { wandb = False, save_volumes = True, viz_every = 100 }
}
