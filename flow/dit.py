import torch
import torch.nn as nn
import numpy as np
import math

from timm.layers.attention import Attention
from timm.layers.mlp import Mlp

from functools import partial
from typing import cast


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (D, H, W)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.fc1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        x = self.fc1(t_freq)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=cast(type[nn.GELU], partial(nn.GELU, approximate="tanh")),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size ** 3 * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_class_condition=False,
    ):
        super().__init__()
        assert hidden_size % 3 == 0

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.use_class_condition = use_class_condition

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.use_class_condition:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        else:
            self.y_embedder = None

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.fc1.weight, std=0.02)
        nn.init.normal_(self.t_embedder.fc2.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, spatial_shape):
        c = self.out_channels
        p = self.patch_size
        D, H, W = spatial_shape

        x = x.reshape(x.shape[0], D, H, W, p, p, p, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(x.shape[0], c, D * p, H * p, W * p)
        return x

    def forward(self, x, t, y=None):
        x, spatial_shape = self.x_embedder(x)

        pos_embed = get_3d_sincos_pos_embed(x.shape[-1], spatial_shape)
        pos_embed = torch.from_numpy(pos_embed).to(device=x.device, dtype=x.dtype).unsqueeze(0)

        x = x + pos_embed

        t = self.t_embedder(t)

        if self.use_class_condition:
            assert y is not None
            y = self.y_embedder(y, self.training)
            c = t + y
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x, spatial_shape)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        assert self.use_class_condition

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_3d_sincos_pos_embed(embed_dim, grid_size_dhw):
    assert embed_dim % 3 == 0
    D, H, W = grid_size_dhw
    dim_each = embed_dim // 3

    emb_d = get_1d_sincos_pos_embed_from_grid(dim_each, np.arange(D, dtype=np.float32))
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_each, np.arange(H, dtype=np.float32))
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_each, np.arange(W, dtype=np.float32))

    emb_d = np.broadcast_to(emb_d[:, None, None, :], (D, H, W, dim_each)).copy()
    emb_h = np.broadcast_to(emb_h[None, :, None, :], (D, H, W, dim_each)).copy()
    emb_w = np.broadcast_to(emb_w[None, None, :, :], (D, H, W, dim_each)).copy()

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=-1)
    return emb.reshape(D * H * W, embed_dim)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    return np.concatenate([emb_sin, emb_cos], axis=1)


def DiT_XL_1(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


def DiT_L_1(**kwargs):
    return DiT(depth=24, hidden_size=1056, patch_size=1, num_heads=16, in_channels=32, **kwargs)


def DiT_S_1(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/1": DiT_XL_1,
    "DiT-L/1": DiT_L_1,
    "DiT-S/1": DiT_S_1,
}
