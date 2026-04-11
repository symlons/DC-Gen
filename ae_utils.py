import torch

def compute_loss(fake, real, loss_fns, gan, global_step, cfg):
    loss_terms = {}
    loss = sum((loss_terms.update({name: (value := fn(fake, real))}) or 0) + weight * value for name, fn, weight in loss_fns if weight > 0)
    if gan is not None:
        gan_loss, d_loss = gan.step(real, fake, global_step)
        loss_terms |= {"gan": gan_loss, **({"gan_d": d_loss} if d_loss is not None else {})}
        loss += cfg.objective.gan_weight * gan_loss
    return loss_terms, loss
