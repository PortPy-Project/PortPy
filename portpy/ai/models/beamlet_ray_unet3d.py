# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# ----------------------------------------------------------------------
"""
Plain 3D U-Net for per-beamlet dose prediction.

Vendored verbatim from ANet_First_Train_GJ_d1_d2_save_light.py so the
inference workflow is self-contained. Architecture matches the 'unet3d'
backbone in LitDoseModel.build_model:
    UNet3D(in_channels=3, out_channels=1, base_features=32, num_groups=8)
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3d -> GroupNorm -> LeakyReLU) x 2"""
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        assert out_ch % num_groups == 0, \
            f"out_ch={out_ch} must be divisible by num_groups={num_groups}"
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.relu1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.relu1(self.gn1(self.conv1(x)))
        x = self.relu2(self.gn2(self.conv2(x)))
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_features=32, num_groups=8):
        super().__init__()
        bf = base_features
        self.enc1      = DoubleConv(in_channels, bf,      num_groups=num_groups)
        self.enc2      = DoubleConv(bf,           bf * 2, num_groups=num_groups)
        self.enc3      = DoubleConv(bf * 2,       bf * 4, num_groups=num_groups)
        self.enc4      = DoubleConv(bf * 4,       bf * 8, num_groups=num_groups)
        self.pool      = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(bf * 8, bf * 16, num_groups=num_groups)
        self.upconv4   = nn.ConvTranspose3d(bf * 16, bf * 8, kernel_size=2, stride=2)
        self.dec4      = DoubleConv(bf * 16, bf * 8, num_groups=num_groups)
        self.upconv3   = nn.ConvTranspose3d(bf * 8,  bf * 4, kernel_size=2, stride=2)
        self.dec3      = DoubleConv(bf * 8,  bf * 4, num_groups=num_groups)
        self.upconv2   = nn.ConvTranspose3d(bf * 4,  bf * 2, kernel_size=2, stride=2)
        self.dec2      = DoubleConv(bf * 4,  bf * 2, num_groups=num_groups)
        self.upconv1   = nn.ConvTranspose3d(bf * 2,  bf,     kernel_size=2, stride=2)
        self.dec1      = DoubleConv(bf * 2,  bf,     num_groups=num_groups)
        self.conv_out  = nn.Conv3d(bf, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.upconv4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.conv_out(d1)