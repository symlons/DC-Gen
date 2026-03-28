import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp


def _to_3tuple(value):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        assert len(value) == 3, "Expected a 3-tuple for 3D shapes."
        return tuple(int(v) for v in value)
    value = int(value)
    return (value, value, value)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed3D(nn.Module):
    def __init__(self, input_size, patch_size, in_channels, embed_dim, bias=True):
        super().__init__()
        self.input_size = _to_3tuple(input_size)
        self.patch_size = _to_3tuple(patch_size)

        if any(size % patch != 0 for size, patch in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"Input size {self.input_size} must be divisible by patch size {self.patch_size}."
            )

        self.grid_size = tuple(size // patch for size, patch in zip(self.input_size, self.patch_size))
        self.num_patches = math.prod(self.grid_size)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (d, h, w)


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
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
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
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
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
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        patch_size = _to_3tuple(patch_size)
        patch_volume = math.prod(patch_size)

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_volume * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = _to_3tuple(patch_size)
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed3D(input_size, self.patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.register_buffer(
            "pos_embed",
            torch.zeros(1, self.x_embedder.num_patches, hidden_size),
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

    def _pos_embed_for_shape(self, spatial_shape, device, dtype):
        spatial_shape = tuple(int(v) for v in spatial_shape)
        if spatial_shape == self.x_embedder.grid_size:
            return self.pos_embed.to(device=device, dtype=dtype)

        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], spatial_shape)
        return torch.from_numpy(pos_embed).to(device=device, dtype=dtype).unsqueeze(0)

    def unpatchify(self, x, spatial_shape):
        c = self.out_channels
        pd, ph, pw = self.patch_size
        d, h, w = (int(v) for v in spatial_shape)

        if x.shape[1] != d * h * w:
            raise ValueError(
                f"Token count {x.shape[1]} does not match spatial shape {spatial_shape}."
            )

        x = x.reshape(x.shape[0], d, h, w, pd, ph, pw, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(x.shape[0], c, d * pd, h * ph, w * pw)
        return x

    def forward(self, x, t, y):
        x, spatial_shape = self.x_embedder(x)
        x = x + self._pos_embed_for_shape(spatial_shape, x.device, x.dtype)

        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x, spatial_shape)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    grid_size = _to_3tuple(grid_size)
    grid_d = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing="ij")
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(3, -1)
    return get_3d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    axis_dims = _split_evenly_across_axes(embed_dim, num_axes=3)
    emb_d = get_1d_sincos_pos_embed_from_grid(axis_dims[0], grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(axis_dims[1], grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(axis_dims[2], grid[2])
    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)

    if emb.shape[1] < embed_dim:
        pad = np.zeros((emb.shape[0], embed_dim - emb.shape[1]), dtype=emb.dtype)
        emb = np.concatenate([emb, pad], axis=1)
    return emb


def _split_evenly_across_axes(embed_dim, num_axes):
    if embed_dim <= 0:
        raise ValueError("embed_dim must be positive.")

    dims = [embed_dim // num_axes for _ in range(num_axes)]
    dims = [dim - (dim % 2) for dim in dims]

    remaining = embed_dim - sum(dims)
    axis_idx = 0
    while remaining >= 2:
        dims[axis_idx] += 2
        remaining -= 2
        axis_idx = (axis_idx + 1) % num_axes

    return dims


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, max_period=10000):
    pos = np.asarray(pos).reshape(-1)
    if embed_dim == 0:
        return np.zeros((pos.shape[0], 0), dtype=np.float32)
    if embed_dim % 2 != 0:
        raise ValueError(f"1D sin-cos embedding dimension must be even, got {embed_dim}.")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (max_period**omega)

    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb.astype(np.float32)


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


def DiT_L_1(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-S/2": DiT_S_2,
    "DiT-S/8": DiT_S_8,
    "DiT-L/1": DiT_L_1,
}
