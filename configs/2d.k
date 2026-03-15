import .config

cfg: config.Config {
    dims = "2d"
    pipeline: config.Pipeline { resize_hw = [64, 64] }
    model: config.Model { name = "dc-ae-f32c32-in-1.0_2d", compile = True }
}
