# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# You may obtain a copy of the Apache 2 License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ----------------------------------------------------------------------
# Commons Clause Restriction Notice:
# PortPy is licensed under Apache 2.0 with the Commons Clause.
# You may use, modify, and share the code for non-commercial
# academic and research purposes only.
# Commercial use — including offering PortPy as a service,
# or incorporating it into a commercial product — requires
# a separate commercial license.
# ----------------------------------------------------------------------
"""## 3D Vision Transformer (ViT) baseline for beamlet dose prediction.

Encoder is a pure ViT (Dosovitskiy et al., ICLR 2021) lifted to 3D: non-overlapping
patch embedding -> learned positional embedding -> `depth` pre-norm multi-head
self-attention blocks. There is NO convolution in the encoder.

The decoder is deliberately identical to UNet3D's decoder (same DoubleConv, same
GroupNorm/LeakyReLU, same skip topology and channel widths), so that a comparison
against the UNet3D baseline isolates the encoder: conv encoder vs. transformer
encoder, everything else held fixed.

Sizing (measured at the (96, 192, 224) data grid, batch 2, precision 32):
    backbone                      params    peak mem   step
    UNet3D (baseline)             22.58 M   21.37 GB   0.715 s
    BeamletViT3D d288 L8  (matched)  22.86 M   16.35 GB   0.671 s
    BeamletViT3D d768 L12 (ViT-B)   110.24 M   18.32 GB   0.720 s
The ViT is *cheaper* in memory than the U-Net because its encoder runs at 1/16
resolution, while the U-Net's full-resolution 32-channel enc1 dominates.

Default is ViT-B scale (110 M), matching the ~115 M reported by UNetrDose
(Phys. Med. Biol. 2026) for transformer-based beamlet dose prediction.
For a parameter-matched run: embed_dim=288, depth=8, num_heads=8.
Set use_skips=False for the pure-ViT / SETR-naive ablation (no encoder skips).

Vendored verbatim from the research training script
ANet_First_Train_GJ_d1_d2_save_light_fix_sampling_paper.py (class ``ViTDose3D`` and
its helpers, renamed here to ``BeamletViT3D`` for consistency with the other beamlet
models), so the PortPy inference workflow is self-contained. ``DoubleConv`` is imported
from :mod:`portpy.ai.models.beamlet_unet3d` rather than duplicated -- the definitions
are byte-identical, which is what lets a research checkpoint load here with
``strict=True``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .beamlet_unet3d import DoubleConv


class PatchEmbed3D(nn.Module):
    """Non-overlapping linear projection of 3D patches to tokens (the ViT tokenizer)."""

    def __init__(self, in_channels=4, embed_dim=288, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, E, d, h, w
        grid = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B, N, E
        return x, grid


class Attention3D(nn.Module):
    """Standard multi-head self-attention (uses PyTorch SDPA / flash kernels)."""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        p = self.attn_drop if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class ViTBlock(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention3D(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _vit_up_block(in_ch, out_ch, n_up, num_groups=8):
    """n_up successive x2 transposed-conv upsamples, projecting tokens back to a feature map."""
    layers = []
    c = in_ch
    for _ in range(n_up):
        layers += [
            nn.ConvTranspose3d(c, out_ch, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        c = out_ch
    return nn.Sequential(*layers)


class BeamletViT3D(nn.Module):
    """3D Vision Transformer encoder + UNet3D-identical convolutional decoder."""

    def __init__(self, in_channels=4, out_channels=1, img_size=(96, 192, 224),
                 patch_size=16, embed_dim=288, depth=8, num_heads=8, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0, base_features=32, num_groups=8,
                 use_skips=True):
        super().__init__()
        self.patch_size = patch_size
        self.use_skips = use_skips
        self.grid = tuple(s // patch_size for s in img_size)

        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        num_patches = self.grid[0] * self.grid[1] * self.grid[2]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # intermediate ViT features at evenly spaced depths feed the decoder skips
        self.skip_idx = [max(1, round(depth * f)) - 1 for f in (0.25, 0.5, 0.75)]

        bf = base_features
        self.proj_bottleneck = nn.Sequential(
            nn.Conv3d(embed_dim, bf * 16, kernel_size=1),
            nn.GroupNorm(num_groups, bf * 16),
            nn.LeakyReLU(0.01, inplace=True))
        if use_skips:
            self.up_to_8 = _vit_up_block(embed_dim, bf * 8, 1, num_groups)  # 1/16 -> 1/8
            self.up_to_4 = _vit_up_block(embed_dim, bf * 4, 2, num_groups)  # 1/16 -> 1/4
            self.up_to_2 = _vit_up_block(embed_dim, bf * 2, 3, num_groups)  # 1/16 -> 1/2
            self.enc1 = DoubleConv(in_channels, bf, num_groups=num_groups)  # full res

        # ---- decoder: identical to UNet3D ----
        self.upconv4 = nn.ConvTranspose3d(bf * 16, bf * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(bf * 16 if use_skips else bf * 8, bf * 8, num_groups=num_groups)
        self.upconv3 = nn.ConvTranspose3d(bf * 8, bf * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(bf * 8 if use_skips else bf * 4, bf * 4, num_groups=num_groups)
        self.upconv2 = nn.ConvTranspose3d(bf * 4, bf * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(bf * 4 if use_skips else bf * 2, bf * 2, num_groups=num_groups)
        self.upconv1 = nn.ConvTranspose3d(bf * 2, bf, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(bf * 2 if use_skips else bf, bf, num_groups=num_groups)
        self.conv_out = nn.Conv3d(bf, out_channels, kernel_size=1)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _get_pos_embed(self, grid):
        """Trilinearly interpolate the positional embedding if the input grid differs."""
        if tuple(grid) == tuple(self.grid):
            return self.pos_embed
        pe = self.pos_embed.reshape(1, *self.grid, -1).permute(0, 4, 1, 2, 3)
        pe = F.interpolate(pe, size=tuple(grid), mode='trilinear', align_corners=False)
        return pe.permute(0, 2, 3, 4, 1).reshape(1, -1, pe.shape[1])

    @staticmethod
    def _tokens_to_map(t, grid):
        B, N, C = t.shape
        return t.transpose(1, 2).reshape(B, C, *grid)

    def forward(self, x):
        tokens, grid = self.patch_embed(x)
        tokens = self.pos_drop(tokens + self._get_pos_embed(grid))

        feats = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i in self.skip_idx:
                feats.append(tokens)
        z = self.norm(tokens)

        bottleneck = self.proj_bottleneck(self._tokens_to_map(z, grid))

        if self.use_skips:
            enc1 = self.enc1(x)
            enc2 = self.up_to_2(self._tokens_to_map(feats[0], grid))
            enc3 = self.up_to_4(self._tokens_to_map(feats[1], grid))
            enc4 = self.up_to_8(self._tokens_to_map(feats[2], grid))
            dec4 = self.dec4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
            dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc3), dim=1))
            dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc2), dim=1))
            dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))
        else:
            dec4 = self.dec4(self.upconv4(bottleneck))
            dec3 = self.dec3(self.upconv3(dec4))
            dec2 = self.dec2(self.upconv2(dec3))
            dec1 = self.dec1(self.upconv1(dec2))

        return self.conv_out(dec1)
