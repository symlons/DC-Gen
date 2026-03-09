
from omegaconf import OmegaConf

def get_default_config():
    cfg = OmegaConf.create({
        "paths": {
            "hdf_path": "/workspace/ct_rate_train_batch_0_v13.hdf",
            "artifact_dir": "artifacts_8_3d",
            "checkpoint_dir": "/checkpoints"
        },
        "training": {
            "batch_size": 1,
            "num_epochs": 200,
            "shuffle_data": False,
            "diff_save_every": 500,
            "checkpoint_every": 2000,
            "max_checkpoints": 5,
            "lr": 1e-5,
            "device": "cuda",
            "dtype": "bfloat16",
            "loss_fn": "l1",
            "resume_from_checkpoint": True,
            "checkpoint_path": "/workspace/DC-Gen/3d_experiments/checkpoints/checkpoint_iter4000.pt"
        },
        "dataset": {
            "name": "CTVolume",
            "group_names": ["Vol_full"],
            "volume": True
        },
        "model": {"name": "dc-ae-f32c32-in-1.0"},
        "pipeline": {"n_slices": 32, "resize_hw": [64, 64]},
        "wandb": {"enabled": True, "project": "ct_recon", "run_name": "dc_ae_experiment"},
        "logging": {"save_volumes": True, "save_training_curves": True}
    })
    return cfg