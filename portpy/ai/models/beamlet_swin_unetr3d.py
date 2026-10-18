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
"""## 3D Swin Transformer (Swin UNETR style) baseline for beamlet dose prediction.

As with BeamletViT3D, the decoder is identical to UNet3D's so that a
comparison against the UNet3D baseline isolates the encoder.

Window size (6, 6, 7) is chosen so that it divides EVERY stage resolution of the
(96, 192, 224) data grid exactly:
    patch_size=2 -> (48,96,112) -> (24,48,56) -> (12,24,28) -> (6,12,14)
No padding or cropping is therefore needed anywhere in the encoder, which removes
the main source of subtle bugs in 3D shifted-window attention. At patch_size=2 the
decoder resolutions line up with UNet3D's exactly (no interpolation needed).

Sizing (measured at (96, 192, 224), batch 2 per GPU, precision 32):
    config                                                params   peak mem
    feature_size=48,  depths=(2,2,2,2),  heads=(3,6,12,24)  14.4 M   30.1 GB
    feature_size=96,  depths=(2,2,2,2),  heads=(3,6,12,24)  31.1 M   39.7 GB
    feature_size=96,  depths=(2,2,6,2),  heads=(3,6,12,24)  38.3 M   41.8 GB
    feature_size=72,  depths=(2,2,18,2), heads=(3,6,12,24)  37.7 M   41.5 GB
    feature_size=96,  depths=(2,2,18,2), heads=(3,6,12,24)  59.8 M   48.0 GB  <- default
    feature_size=128, depths=(2,2,18,2), heads=(4,8,16,32)  99.2 M   58.7 GB
Note feature_size must be divisible by every entry of num_heads (so f64 with
heads=(3,...) is invalid).

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .beamlet_unet3d import DoubleConv


def window_partition_3d(x, ws):
    """x: B,D,H,W,C -> (B*nW, wd*wh*ww, C)"""
    B, D, H, W, C = x.shape
    wd, wh, ww = ws
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wd * wh * ww, C)


def window_reverse_3d(windows, ws, B, D, H, W):
    """(B*nW, wd*wh*ww, C) -> B,D,H,W,C"""
    wd, wh, ww = ws
    C = windows.shape[-1]
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, C)


def _adjust_window(window_size, shift_size, resolution):
    """Collapse any axis whose resolution is <= the window into one unshifted window."""
    ws, ss = list(window_size), list(shift_size)
    for i in range(3):
        if resolution[i] <= ws[i]:
            ws[i] = resolution[i]
            ss[i] = 0
    return tuple(ws), tuple(ss)


class WindowAttention3D(nn.Module):
    """Window-based multi-head self-attention with 3D relative position bias."""

    def __init__(self, dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        wd, wh, ww = window_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads))

        coords = torch.stack(torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing='ij'))  # 3,wd,wh,ww
        cf = coords.flatten(1)  # 3,N
        rel = (cf[:, :, None] - cf[:, None, :]).permute(1, 2, 0).contiguous()  # N,N,3
        rel[:, :, 0] += wd - 1
        rel[:, :, 1] += wh - 1
        rel[:, :, 2] += ww - 1
        rel[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        rel[:, :, 1] *= (2 * ww - 1)
        self.register_buffer("relative_position_index", rel.sum(-1), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, nW=None):
        """x: (B*nW, N, C); mask: (nW, N, N) additive, or None.

        The bias/mask is broadcast over the batch rather than materialised per
        batch element -- at stage-0 resolutions that alone saves several GB.
        Verified to match an explicit softmax-attention reference to 1.7e-16 (fp64).
        """
        Bn, N, C = x.shape
        H, hd = self.num_heads, self.head_dim

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(N, N, H)
        bias = bias.permute(2, 0, 1).contiguous()  # heads,N,N

        p = self.attn_drop if self.training else 0.0
        if mask is None:
            qkv = self.qkv(x).reshape(Bn, N, 3, H, hd).permute(2, 0, 3, 1, 4)
            # bias broadcasts over the (B*nW) batch dim
            out = F.scaled_dot_product_attention(
                qkv[0], qkv[1], qkv[2], attn_mask=bias.unsqueeze(0), dropout_p=p)
            out = out.transpose(1, 2).reshape(Bn, N, C)
        else:
            B = Bn // nW
            qkv = self.qkv(x).reshape(B, nW, N, 3, H, hd).permute(3, 0, 1, 4, 2, 5)
            # (1, nW, heads, N, N) broadcasts over B -- no repeat over batch
            attn_bias = (bias.unsqueeze(0) + mask.unsqueeze(1)).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                qkv[0], qkv[1], qkv[2], attn_mask=attn_bias, dropout_p=p)
            out = out.permute(0, 1, 3, 2, 4).reshape(Bn, N, C)
        return self.proj_drop(self.proj(out))


class SwinBlock3D(nn.Module):
    """Swin transformer block: (shifted) window attention + MLP, pre-norm."""

    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        # Standard Swin guard: if a stage has collapsed to at most one window along
        # an axis, that axis becomes a single full-extent window with no shift.
        window_size, shift_size = _adjust_window(window_size, shift_size, input_resolution)
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        for r, w in zip(input_resolution, window_size):
            assert r % w == 0, \
                f"stage res {input_resolution} not divisible by window {window_size}"

        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = WindowAttention3D(dim, window_size, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop))

        if any(s > 0 for s in shift_size):
            self.register_buffer("attn_mask",
                                 self._build_mask(input_resolution, window_size, shift_size),
                                 persistent=False)
        else:
            self.attn_mask = None

    @staticmethod
    def _build_mask(res, ws, ss):
        D, H, W = res
        img_mask = torch.zeros((1, D, H, W, 1))
        cnt = 0
        for d in (slice(0, -ws[0]), slice(-ws[0], -ss[0]), slice(-ss[0], None)):
            for h in (slice(0, -ws[1]), slice(-ws[1], -ss[1]), slice(-ss[1], None)):
                for w in (slice(0, -ws[2]), slice(-ws[2], -ss[2]), slice(-ss[2], None)):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mw = window_partition_3d(img_mask, ws).squeeze(-1)  # nW, N
        attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)  # nW, N, N
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x):
        """x: B,D,H,W,C"""
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=[-s for s in self.shift_size], dims=(1, 2, 3))

        wd, wh, ww = self.window_size
        nW = (D // wd) * (H // wh) * (W // ww)
        xw = window_partition_3d(x, self.window_size)
        xw = self.attn(xw, self.attn_mask, nW)
        x = window_reverse_3d(xw, self.window_size, B, D, H, W)

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=list(self.shift_size), dims=(1, 2, 3))

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchMerging3D(nn.Module):
    """2x spatial downsample, channel doubling (concat 8 neighbours -> Linear)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim, eps=1e-5)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x):
        parts = [x[:, i::2, j::2, k::2, :]
                 for i in range(2) for j in range(2) for k in range(2)]
        x = torch.cat(parts, dim=-1)
        return self.reduction(self.norm(x))


class SwinStage(nn.Module):
    """`depth` Swin blocks, alternating non-shifted / shifted windows."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, downsample=True):
        super().__init__()
        shift = tuple(w // 2 for w in window_size)
        self.blocks = nn.ModuleList([
            SwinBlock3D(dim, input_resolution, num_heads, window_size,
                        (0, 0, 0) if (i % 2 == 0) else shift,
                        mlp_ratio, drop, attn_drop)
            for i in range(depth)])
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class BeamletSwinUNETR3D(nn.Module):
    """3D Swin Transformer encoder + UNet3D-identical convolutional decoder."""

    def __init__(self, in_channels=4, out_channels=1, img_size=(96, 192, 224),
                 patch_size=2, window_size=(6, 6, 7), feature_size=128,
                 depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0, base_features=32, num_groups=8):
        super().__init__()
        assert len(depths) == 4 and len(num_heads) == 4, "expected 4 encoder stages"
        C = feature_size
        res0 = tuple(s // patch_size for s in img_size)
        for r, w in zip(res0, window_size):
            assert r % w == 0, f"stage-0 res {res0} not divisible by window {window_size}"

        # every stage that is followed by PatchMerging must have even extents
        _r = res0
        for i in range(len(depths) - 1):
            assert all(v % 2 == 0 for v in _r), (
                f"stage-{i} resolution {_r} must be even for PatchMerging; "
                f"img_size={img_size}, patch_size={patch_size}")
            _r = tuple(v // 2 for v in _r)

        self.patch_embed = nn.Conv3d(in_channels, C, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(C, eps=1e-5)
        self.pos_drop = nn.Dropout(drop_rate)

        self.stages = nn.ModuleList()
        res = res0
        dim = C
        for i, (d, h) in enumerate(zip(depths, num_heads)):
            last = (i == len(depths) - 1)
            self.stages.append(SwinStage(dim, res, d, h, window_size, mlp_ratio,
                                         drop_rate, attn_drop_rate, downsample=not last))
            if not last:
                res = tuple(r // 2 for r in res)
                dim *= 2
        self.enc_dims = [C * (2 ** i) for i in range(len(depths))]

        bf = base_features
        # project Swin stage outputs onto the widths UNet3D's decoder expects
        self.proj_bottleneck = nn.Sequential(
            nn.Conv3d(self.enc_dims[3], bf * 16, kernel_size=1),
            nn.GroupNorm(num_groups, bf * 16), nn.LeakyReLU(0.01, inplace=True))
        self.proj_s3 = nn.Sequential(
            nn.Conv3d(self.enc_dims[2], bf * 8, kernel_size=1),
            nn.GroupNorm(num_groups, bf * 8), nn.LeakyReLU(0.01, inplace=True))
        self.proj_s2 = nn.Sequential(
            nn.Conv3d(self.enc_dims[1], bf * 4, kernel_size=1),
            nn.GroupNorm(num_groups, bf * 4), nn.LeakyReLU(0.01, inplace=True))
        self.proj_s1 = nn.Sequential(
            nn.Conv3d(self.enc_dims[0], bf * 2, kernel_size=1),
            nn.GroupNorm(num_groups, bf * 2), nn.LeakyReLU(0.01, inplace=True))
        self.enc1 = DoubleConv(in_channels, bf, num_groups=num_groups)  # full res

        # ---- decoder: identical to UNet3D ----
        self.upconv4 = nn.ConvTranspose3d(bf * 16, bf * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(bf * 16, bf * 8, num_groups=num_groups)
        self.upconv3 = nn.ConvTranspose3d(bf * 8, bf * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(bf * 8, bf * 4, num_groups=num_groups)
        self.upconv2 = nn.ConvTranspose3d(bf * 4, bf * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(bf * 4, bf * 2, num_groups=num_groups)
        self.upconv1 = nn.ConvTranspose3d(bf * 2, bf, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(bf * 2, bf, num_groups=num_groups)
        self.conv_out = nn.Conv3d(bf, out_channels, kernel_size=1)

        self.patch_size = patch_size
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

    @staticmethod
    def _to_map(x):
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x):
        inp = x
        z = self.patch_embed(x).permute(0, 2, 3, 4, 1).contiguous()  # B,D,H,W,C
        z = self.pos_drop(self.patch_norm(z))

        skips = []
        for st in self.stages:
            z, s = st(z)
            skips.append(s)

        bottleneck = self.proj_bottleneck(self._to_map(skips[3]))
        s3 = self.proj_s3(self._to_map(skips[2]))
        s2 = self.proj_s2(self._to_map(skips[1]))
        s1 = self.proj_s1(self._to_map(skips[0]))
        e1 = self.enc1(inp)

        dec4 = self.dec4(torch.cat((self.upconv4(bottleneck), s3), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), s2), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), s1), dim=1))
        up1 = self.upconv1(dec2)
        # exact at patch_size=2; guard keeps other patch sizes usable
        if up1.shape[2:] != e1.shape[2:]:
            up1 = F.interpolate(up1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        dec1 = self.dec1(torch.cat((up1, e1), dim=1))
        return self.conv_out(dec1)
