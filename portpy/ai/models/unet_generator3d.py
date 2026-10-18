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

"""3D U-Net generator (pix2pix style) used by the ``unet_128`` / ``unet_256``
generators. Moved out of ``networks3d.py`` verbatim; behaviour is unchanged.
"""
import functools

import torch
import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class UnetGenerator3d(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a 3D Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator3d, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                               innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock3d(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                               norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3d(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                               norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3d(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock3d(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                               norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input, site=None):
        """Standard forward. ``site`` is accepted and ignored."""
        return self.model(input)


class UnetSkipConnectionBlock3d(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock3d, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        interp = 'trilinear'  # 'nearest'
        transp_conv = False  # Use transposed convolution or resize convolution?
        # transp_conv = True

        if outermost:
            if transp_conv is True:
                upconv = [nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                             kernel_size=4, stride=2,
                                             padding=1)]
            else:
                upsamp = nn.Upsample(scale_factor=2, mode=interp)
                conv = nn.Conv3d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
                upconv = [upsamp, conv]

            down = [downconv]
            up = [uprelu, *upconv, nn.ReLU()]

            model = down + [submodule] + up
        elif innermost:
            if transp_conv is True:
                upconv = [nn.ConvTranspose3d(inner_nc, outer_nc,
                                             kernel_size=4, stride=2,
                                             padding=1, bias=use_bias)]
            else:
                upsamp = nn.Upsample(scale_factor=2, mode=interp)
                conv = nn.Conv3d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
                upconv = [upsamp, conv]

            down = [downrelu, downconv]
            up = [uprelu, *upconv, upnorm]
            model = down + up
        else:
            if transp_conv is True:
                upconv = [nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                             kernel_size=4, stride=2,
                                             padding=1, bias=use_bias)]
            else:
                upsamp = nn.Upsample(scale_factor=2, mode=interp)
                conv = nn.Conv3d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
                upconv = [upsamp, conv]
            down = [downrelu, downconv, downnorm]
            up = [uprelu, *upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


# Standard 3D Unet with padding; No dropout
