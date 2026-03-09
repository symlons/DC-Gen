
from omegaconf import OmegaConf

def get_default_config():
    cfg = OmegaConf.create({
        "paths": {
            "hdf_path": "/Users/sfkost/storage/ct_rate_train_batch_0_v13.hdf",
            "artifact_dir": "artifacts_7_3d",
            "checkpoint_dir": "checkpoints"
        },
        "training": {
            "batch_size": 1,
            "num_iters": 20000,
            "shuffle_data": False,
            "diff_save_every": 500,
            "checkpoint_every": 2000,
            "max_checkpoints": 5,
            "lr": 1e-5,
            "device": "mps",
            "dtype": "bfloat16"
        },
        "model": {"name": "dc-ae-f32c32-in-1.0"},
        "pipeline": {"n_slices": 32, "resize_hw": [64, 64]},
        "wandb": {"enabled": False, "project": "ct_recon", "run_name": "dc_ae_experiment"},
        "logging": {"save_volumes": True, "save_training_curves": True}
    })
    return cfg
