import .config

cfg: config.Config {
    dims = "3d"
    pipeline: config.Pipeline { resize_hw = [64, 64], n_slices = 32 }
    model: config.Model { name = "dc-ae-f32c32-in-1.0_3d", compile = True }
}
