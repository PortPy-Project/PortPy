# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# ----------------------------------------------------------------------
"""
3D Attention Residual U-Net for per-beamlet dose prediction.

Vendored verbatim from the research training script
(``ANet_First_Train_GJ_d1_d2_save_light.py``) so the production inference
workflow is self-contained. The default configuration matches the trained
checkpoint: ``in_channels=3`` (CT, d1, d1_out), ``num_levels=3``,
``base_features=32``, ``num_groups=8``, ``use_cbam=False``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_ch)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_ch)
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act2(out + identity)


class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool3d(x, 1)
        max_pool = F.adaptive_max_pool3d(x, 1)
        return x * self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_map, max_map], dim=1)
        return x * self.sigmoid(self.conv(attn))


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        self.channel = ChannelAttention3D(channels, reduction)
        self.spatial = SpatialAttention3D(spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


class ResCBAMBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8, reduction=8):
        super().__init__()
        self.res = ResBlock3D(in_ch, out_ch, num_groups=num_groups)
        self.cbam = CBAM3D(out_ch, reduction=reduction)

    def forward(self, x):
        return self.cbam(self.res(x))


class AttentionGate3D(nn.Module):
    """Attention U-Net gate: x = encoder skip, g = decoder gating feature."""
    def __init__(self, x_ch, g_ch, inter_ch):
        super().__init__()
        self.theta_x = nn.Conv3d(x_ch, inter_ch, 1, bias=False)
        self.phi_g = nn.Conv3d(g_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        attn = self.psi(self.theta_x(x) + self.phi_g(g))
        return x * attn


class AttentionResUNet3D(nn.Module):
    """Attention Residual U-Net 3D with selectable depth (num_levels in {2,3,4})."""
    def __init__(self, in_channels=3, out_channels=1, base_features=32,
                 num_groups=8, use_cbam=False, num_levels=3):
        super().__init__()
        assert num_levels in (2, 3, 4), "num_levels must be 2, 3, or 4"
        self.num_levels = num_levels
        Block = ResCBAMBlock3D if use_cbam else ResBlock3D
        bf = base_features

        self.enc1 = Block(in_channels, bf, num_groups=num_groups)
        self.enc2 = Block(bf, bf * 2, num_groups=num_groups)
        if num_levels >= 3:
            self.enc3 = Block(bf * 2, bf * 4, num_groups=num_groups)
        if num_levels >= 4:
            self.enc4 = Block(bf * 4, bf * 8, num_groups=num_groups)
        self.pool = nn.MaxPool3d(2)

        if num_levels == 2:
            self.bottleneck = Block(bf * 2, bf * 4, num_groups=num_groups)
            self.upconv2 = nn.ConvTranspose3d(bf * 4, bf * 2, 2, stride=2)
            self.att2 = AttentionGate3D(bf * 2, bf * 2, bf)
            self.dec2 = Block(bf * 4, bf * 2, num_groups=num_groups)
        elif num_levels == 3:
            self.bottleneck = Block(bf * 4, bf * 8, num_groups=num_groups)
            self.upconv3 = nn.ConvTranspose3d(bf * 8, bf * 4, 2, stride=2)
            self.att3 = AttentionGate3D(bf * 4, bf * 4, bf * 2)
            self.dec3 = Block(bf * 8, bf * 4, num_groups=num_groups)
            self.upconv2 = nn.ConvTranspose3d(bf * 4, bf * 2, 2, stride=2)
            self.att2 = AttentionGate3D(bf * 2, bf * 2, bf)
            self.dec2 = Block(bf * 4, bf * 2, num_groups=num_groups)
        else:  # num_levels == 4
            self.bottleneck = Block(bf * 8, bf * 16, num_groups=num_groups)
            self.upconv4 = nn.ConvTranspose3d(bf * 16, bf * 8, 2, stride=2)
            self.att4 = AttentionGate3D(bf * 8, bf * 8, bf * 4)
            self.dec4 = Block(bf * 16, bf * 8, num_groups=num_groups)
            self.upconv3 = nn.ConvTranspose3d(bf * 8, bf * 4, 2, stride=2)
            self.att3 = AttentionGate3D(bf * 4, bf * 4, bf * 2)
            self.dec3 = Block(bf * 8, bf * 4, num_groups=num_groups)
            self.upconv2 = nn.ConvTranspose3d(bf * 4, bf * 2, 2, stride=2)
            self.att2 = AttentionGate3D(bf * 2, bf * 2, bf)
            self.dec2 = Block(bf * 4, bf * 2, num_groups=num_groups)

        self.upconv1 = nn.ConvTranspose3d(bf * 2, bf, 2, stride=2)
        self.att1 = AttentionGate3D(bf, bf, max(bf // 2, 1))
        self.dec1 = Block(bf * 2, bf, num_groups=num_groups)
        self.conv_out = nn.Conv3d(bf, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        if self.num_levels == 2:
            bottleneck = self.bottleneck(self.pool(enc2))
            dec2 = self.upconv2(bottleneck)
            enc2 = self.att2(enc2, dec2)
            dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        elif self.num_levels == 3:
            enc3 = self.enc3(self.pool(enc2))
            bottleneck = self.bottleneck(self.pool(enc3))
            dec3 = self.upconv3(bottleneck)
            enc3 = self.att3(enc3, dec3)
            dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
            dec2 = self.upconv2(dec3)
            enc2 = self.att2(enc2, dec2)
            dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        else:  # num_levels == 4
            enc3 = self.enc3(self.pool(enc2))
            enc4 = self.enc4(self.pool(enc3))
            bottleneck = self.bottleneck(self.pool(enc4))
            dec4 = self.upconv4(bottleneck)
            enc4 = self.att4(enc4, dec4)
            dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
            dec3 = self.upconv3(dec4)
            enc3 = self.att3(enc3, dec3)
            dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
            dec2 = self.upconv2(dec3)
            enc2 = self.att2(enc2, dec2)
            dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.upconv1(dec2)
        enc1 = self.att1(enc1, dec1)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        return self.conv_out(dec1)


def load_checkpoint(model, ckpt_path, device):
    """Load a Lightning/plain checkpoint, stripping a leading 'model.' if present."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[attention_resunet3d] missing keys:", missing)
    if unexpected:
        print("[attention_resunet3d] unexpected keys:", unexpected)
    return model