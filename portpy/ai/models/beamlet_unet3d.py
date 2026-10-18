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

"""Beamlet-dose U-Net used by the ``beamlet_unet`` generator, plus its
``DoubleConv`` building block. Moved out of ``networks3d.py`` verbatim;
behaviour is unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv3d -> GroupNorm -> ReLU) x 2
    """
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

class BeamletUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_features=32, num_groups=8):
        super(BeamletUNet3D, self).__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_features, num_groups=num_groups)
        self.enc2 = DoubleConv(base_features, base_features * 2, num_groups=num_groups)
        self.enc3 = DoubleConv(base_features * 2, base_features * 4, num_groups=num_groups)
        self.enc4 = DoubleConv(base_features * 4, base_features * 8, num_groups=num_groups)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_features * 8, base_features * 16, num_groups=num_groups)

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_features * 16, base_features * 8, num_groups=num_groups)

        self.upconv3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_features * 8, base_features * 4, num_groups=num_groups)

        self.upconv2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features * 4, base_features * 2, num_groups=num_groups)

        self.upconv1 = nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features * 2, base_features, num_groups=num_groups)

        # Output layer
        self.conv_out = nn.Conv3d(base_features, out_channels, kernel_size=1)
        self.activation = nn.ReLU()  # optional clamp to non-negative

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))
        # you can optionally keep a tiny dropout here:
        # bottleneck = F.dropout3d(bottleneck, p=0.1, training=self.training)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        out = self.conv_out(dec1)
        # out = self.activation(out)  # enable if you want to force non-negative
        return out


# Defines DVH loss class
