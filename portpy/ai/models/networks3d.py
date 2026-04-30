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

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import warnings

###############################################################################
# Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


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


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, threshold=1e-4, patience=3)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=None, mednext_model_id='B', mednext_kernel_size=3):
    if gpu_ids is None:
        gpu_ids = []
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'unet_128':
        net = UnetGenerator3d(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator3d(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'stand_unet':
        net = UNet3D(in_ch=input_nc, out_ch=output_nc)
    elif netG == 'mednext':
        net = MedNeXtDose(
            in_channels=input_nc,
            out_channels=output_nc,
            model_id=mednext_model_id,
            kernel_size=mednext_kernel_size,
        )
    elif netG == "beamlet_unet":
        net = BeamletUNet3D(input_nc, output_nc)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

# 3D version of UnetGenerator
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

    def forward(self, input):
        """Standard forward"""
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
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.BN = nn.BatchNorm3d(out_ch)
        # block = [conv1, relu, BN, conv2, relu, BN]
        # self.model = nn.Sequential(*block)

    def forward(self, x):
        return self.BN(self.relu(self.conv2(self.BN(self.relu(self.conv1(x))))))
        # return self.model(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, chs=(32, 64, 128, 256, 512)):
        super().__init__()
        chs = (in_ch,) + chs
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class ResizeConv(nn.Module):
    # Resize (upsample) with nearest neighbor or trilinear interpolation and then do convolution
    def __init__(self, in_ch, out_ch, interp='nearest'):
        super().__init__()
        self.upsamp = nn.Upsample(scale_factor=2, mode=interp)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsamp(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        # self.upconvs = nn.ModuleList([nn.ConvTranspose3d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.upconvs = nn.ModuleList([ResizeConv(chs[i], chs[i + 1], interp='trilinear') for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            # enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x

    # def crop(self, enc_ftrs, x):
    #     _, _, H, W = x.shape
    #     enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
    #     return enc_ftrs


class UNet3D(nn.Module):
    def __init__(self, in_ch=1, enc_chs=(32, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64, 32), out_ch=1):
        super().__init__()
        self.encoder = Encoder(in_ch, enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv3d(dec_chs[-1], out_ch, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        # activation = nn.Sigmoid()
        activation = nn.ReLU()
        out = activation(out)
        return out

class MedNeXtDose(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        model_id: str = 'B',
        kernel_size: int = 3,
        deep_supervision: bool = False,

    ):
        super().__init__()
        create_mednext_v1 = None
        try:
            from nnunet_mednext import create_mednext_v1
        except ImportError:
            warnings.warn(
                "MedNeXt is not installed. Install it with "
                "`pip install git+https://github.com/MIC-DKFZ/MedNeXt.git`."
            )

        if create_mednext_v1 is None:
            raise ImportError(
                "Requested MedNeXtDose but MedNeXt is not installed. "
                "Install it with "
                "`pip install git+https://github.com/MIC-DKFZ/MedNeXt.git`."
            )
        from typing import Any, cast
        create_mednext_v1 = cast(Any, create_mednext_v1)
        # Keep open kbp style loading path
        self.model = create_mednext_v1(
            num_input_channels=in_channels,
            num_classes=out_channels,
            model_id=model_id,
            kernel_size=kernel_size,
            deep_supervision=deep_supervision,
        )

        # self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y

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
class DVHLoss(nn.Module):
    def __init__(self):
        super(DVHLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, predicted_dose, target_hist, target_bins, oar):
        """
        Calculate DVH loss: averaged over all OARs. Target hist is already computed
            predicted dose (tensor) -- [N, C, D, H, W] C = 1
            target hist (tensor)    -- [N, n_bins, n_oars]
            target bins (tensor)    -- [N, n_bins]
            oar (tensor)            -- [N, C, D, H, W] C == n_oars one hot encoded OAR including PTV
        """

        # Calculate predicted hist
        vols = torch.sum(oar, axis=(2, 3, 4))
        n_bins = target_bins.shape[1]
        hist = torch.zeros_like(target_hist)
        bin_w = target_bins[0, 1] - target_bins[0, 0]

        # print(vols.shape, hist.shape)

        for i in range(n_bins):
            diff = torch.sigmoid((predicted_dose - target_bins[:, i]) / bin_w)
            # print(diff.shape)
            # diff = torch.cat(oar.shape[1] * [diff.unsqueeze(axis=1)]) * oar
            diff = diff.repeat(1, oar.shape[1], 1, 1, 1) * oar
            num = torch.sum(diff, axis=(2, 3, 4))
            # print(diff.shape, num.shape)
            hist[:, i] = (num / vols)

        # print(hist.detach().cpu().numpy())
        # print(vols.detach().cpu().numpy())

        return self.loss(hist, target_hist)


# Defines Bhattacharya loss class
class BhattLoss(nn.Module):
    def __init__(self):
        super(BhattLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, predicted_dose, target_hist, target_bins, oar):
        """
        Calculate DVH loss: averaged over all OARs. Target hist is already computed
            predicted dose (tensor) -- [N, C, D, H, W] C = 1
            target hist (tensor)    -- [N, n_bins, n_oars]
            target bins (tensor)    -- [N, n_bins]
            oar (tensor)            -- [N, C, D, H, W] C == n_oars one hot encoded OAR including PTV
        """

        # Calculate predicted hist
        vols = torch.sum(oar, axis=(2, 3, 4))
        n_bins = target_bins.shape[1]
        hist = torch.zeros_like(target_hist)
        bin_w = target_bins[0, 1] - target_bins[0, 0]

        # print(vols.shape, hist.shape)

        for i in range(n_bins):
            diff = torch.sigmoid((predicted_dose - target_bins[:, i]) / bin_w)
            # print(diff.shape)
            # diff = torch.cat(oar.shape[1] * [diff.unsqueeze(axis=1)]) * oar
            diff = diff.repeat(1, oar.shape[1], 1, 1, 1) * oar
            num = torch.sum(diff, axis=(2, 3, 4))
            # print(diff.shape, num.shape)
            hist[:, i] = (num / vols)

        print(hist.detach().cpu().numpy())
        print(vols.detach().cpu().numpy())
        histprod = torch.sqrt(hist * target_hist)
        # histprod = torch.clamp(histprod, 1e-8)
        print(histprod.detach().cpu().numpy())
        bhattdist = torch.sum(histprod, axis=(1, 2))  # Sum of bhattacharya distances for each OAR
        bhattdist = torch.clamp(bhattdist, 1e-3)
        print(bhattdist.detach().cpu().numpy())
        bhattloss = torch.mean(-torch.log(bhattdist))
        return bhattloss


import torch.nn.functional as F

class WeightedL1Loss(nn.Module):
    def __init__(self, alpha=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        denom = torch.clamp(target.amax(dim=(2, 3, 4), keepdim=True), min=self.eps)
        rel = target / denom
        w = 1.0 + self.alpha * rel
        loss = torch.abs(pred - target) * w
        return loss.sum() / torch.clamp(w.sum(), min=self.eps)


class MaskedWeightedL1Loss(nn.Module):
    def __init__(self, alpha=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target, mask):
        denom = torch.clamp(target.amax(dim=(2, 3, 4), keepdim=True), min=self.eps)
        rel = target / denom
        w = (1.0 + self.alpha * rel) * mask
        loss = torch.abs(pred - target) * w
        return loss.sum() / torch.clamp(w.sum(), min=self.eps)

class MomentLoss(nn.Module):
    """
    Matches specified dose moments inside each ROI channel.

    Expected:
      y_pred:    [B, 1, D, H, W]
      y_true:    [B, 1, D, H, W]
      mask_dict: dict[str, Tensor], each tensor shaped [B, C, D, H, W]
                 Example:
                   {
                     "OARPTV": structures_tensor,   # channels = OARs + PTV
                   }

    Computes moments per patient, per channel, then averages.
    """
    def __init__(self, moments=(1, 2, 10), reduction="mean", eps=1e-6):
        super().__init__()
        self.moments = moments
        self.reduction = reduction
        self.eps = eps
        self.loss = nn.MSELoss()

    def _masked_moment(self, dose: torch.Tensor, mask: torch.Tensor, n: int) -> torch.Tensor:
        """
        dose: [B,1,D,H,W]
        mask: [B,1,D,H,W]
        returns: [B] nth raw moment inside mask for each sample
        """
        mask = (mask > 0.5).to(dose.dtype)
        vol = mask.sum(dim=(1, 2, 3, 4)).clamp(min=self.eps)   # [B]
        moment_n = torch.pow(
            ((dose.pow(n) * mask).sum(dim=(1, 2, 3, 4)) / vol).clamp(min=self.eps),
            1.0 / n
        )
        return moment_n

    def forward(self,
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                mask_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []

        for n in self.moments:
            for m in mask_dict.values():
                # m: [B,C,D,H,W] or [B,1,D,H,W]
                for c in range(m.size(1)):
                    mask_c = m[:, c:c+1, ...]  # [B,1,D,H,W]

                    # part for skipping empty masks; if vol=0, both moments become 0 and loss becomes 0, so they are automatically skipped in the final averaging
                    mask_bin = (mask_c > 0.5).to(y_true.dtype)
                    vol = mask_bin.sum(dim=(1, 2, 3, 4))  # [B]

                    valid = vol > 0
                    if not valid.any():
                        continue

                    pred_m = self._masked_moment(y_pred, mask_c, n)  # [B]
                    true_m = self._masked_moment(y_true, mask_c, n)  # [B]

                    # skip empty-mask samples automatically because both moments become 0
                    losses.append(torch.abs(pred_m - true_m).mean())
                    # losses.append(self.loss(pred_m[valid], true_m[valid])) # use valid to skip empty-mask samples in average

        if len(losses) == 0:
            return y_true.new_tensor(0.0)

        loss = torch.stack(losses).mean()
        if self.reduction == "mean":
            return loss
        return loss * y_true.numel()


def _central_diff(x: torch.Tensor, dim: int):
    """
    Central finite difference along spatial dimension dim:
      dim = 0 -> depth
      dim = 1 -> height
      dim = 2 -> width

    x shape: [B, 1, D, H, W]
    output shape: same as x
    """
    # F.pad for 5D tensors uses:
    # (W_left, W_right, H_left, H_right, D_left, D_right)
    pad = [0, 0, 0, 0, 0, 0]

    if dim == 0:      # depth
        pad[4] = 1
        pad[5] = 1
    elif dim == 1:    # height
        pad[2] = 1
        pad[3] = 1
    elif dim == 2:    # width
        pad[0] = 1
        pad[1] = 1
    else:
        raise ValueError("dim must be 0, 1, or 2")

    x_pad = F.pad(x, pad, mode="replicate")

    # spatial dims of x are (2, 3, 4)
    spatial_axis = dim + 2

    forward = x_pad.narrow(spatial_axis, 2, x.size(spatial_axis))
    backward = x_pad.narrow(spatial_axis, 0, x.size(spatial_axis))
    return (forward - backward) / 2.0


def _gradient_3d(x: torch.Tensor):
    """
    Returns stacked spatial gradients:
    shape = [3, B, 1, D, H, W]
    """
    return torch.stack([_central_diff(x, d) for d in range(3)], dim=0)


class DualGradientL2Loss(nn.Module):
    """
    Dual gradient loss:
      - edge term emphasizes regions with large GT gradients
      - flat term matches gradients everywhere more uniformly

    Total = lambda_edge * L_edge + lambda_flat * L_flat
    """
    def __init__(self,
                 gamma_edge: int = 25,
                 gamma_flat: int = 0,
                 lambda_edge: float = 0.1,
                 lambda_flat: float = 0.05,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma_edge = gamma_edge
        self.gamma_flat = gamma_flat
        self.lambda_edge = lambda_edge
        self.lambda_flat = lambda_flat
        self.reduction = reduction

    @staticmethod
    def _sharp_term(gp: torch.Tensor, gg: torch.Tensor, gamma: int):
        """
        gp, gg shape: [3, B, 1, D, H, W]
        """
        diff2 = (gp - gg).pow(2).sum(0)   # [B,1,D,H,W]
        weight = (gg.pow(2).sum(0).sqrt() + 1e-6).pow(gamma)
        weight = weight.clamp(max=6)
        return diff2 * weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        gp = _gradient_3d(y_pred)
        gg = _gradient_3d(y_true)

        l_edge = self._sharp_term(gp, gg, self.gamma_edge)
        l_flat = self._sharp_term(gp, gg, self.gamma_flat)

        loss = self.lambda_edge * l_edge + self.lambda_flat * l_flat

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

class BerHuLoss(nn.Module):
    def __init__(self, delta=1.0, eps=1e-8):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, pred, true, mask):
        e = (pred - true) * mask
        ae = e.abs()
        loss = torch.where(
            ae <= self.delta,
            ae,
            (e * e + self.delta * self.delta) / (2 * self.delta)
        )
        return (loss * mask).sum() / (mask.sum() + self.eps)