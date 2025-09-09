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
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'unet_128':
        net = UnetGenerator3d(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator3d(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'stand_unet':
        net = UNet3D(in_ch=input_nc, out_ch=output_nc)
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


##Moment loss
class MomentLoss(nn.Module):
    def __init__(self):
        super(MomentLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, predicted_dose, oar, dose):
        """
        Calculate DVH loss: averaged over all OARs. Target hist is already computed
            predicted dose (tensor) -- [N, C, D, H, W] C = 1
            target hist (tensor)    -- [N, n_bins, n_oars]
            target bins (tensor)    -- [N, n_bins]
            oar (tensor)            -- [N, C, D, H, W] C == n_oars one hot encoded OAR including PTV
        """

        # Calculate predicted hist
        vols = torch.sum(oar, axis=(2, 3, 4))
        # n_bins = target_bins.shape[1]
        # hist = torch.zeros_like(target_hist)
        # bin_w = target_bins[0,1] - target_bins[0,0]
        keys = ['Eso', 'Cord', 'Heart', 'Lung_L', 'Lung_R', 'PTV']
        # moment = [[1, 2, 10], [1, 2, 10], [1, 2, 10], [1, 2, 10], [1, 2, 10], [2, 4, 6]]
        # momentOfStructure = dict(zip(keys, moment))
        momentOfStructure = {'Eso': {'moments': [1, 2, 10], 'weights': [1, 1, 1]},
                             'Cord': {'moments': [1, 2, 10], 'weights': [1, 1, 1]},
                             'Heart': {'moments': [1, 2, 10], 'weights': [1, 1, 1]},
                             'Lung_L': {'moments': [1, 2, 10], 'weights': [1, 1, 1]},
                             'Lung_R': {'moments': [1, 2, 10], 'weights': [1, 1, 1]},
                             'PTV': {'moments': [2, 4, 6], 'weights': [1, 1, 1]}}
        # momentOfStructure = dict([(k, v) for k in keys for v in values])
        oarPredMoment = []
        oarRealMoment = []
        pres = 60
        epsilon = 0.00001  # Added epsilon as the loss function can become sqrt(0)
        for i in range(oar.shape[1]):
            moments = momentOfStructure[keys[i]]['moments']
            weights = momentOfStructure[keys[i]]['weights']
            for j in range(len(moments)):
                gEUDa = moments[j]
                weight = weights[j]
                oarpreddose = predicted_dose * oar[:, i, :, :, :]
                oarRealDose = dose * oar[:, i, :, :, :]
                if i < (oar.shape[1] - 1):
                    # oarPredMomenta = torch.pow((1 / vols[0, i]) * (torch.sum(torch.pow(oarpreddose-pres, gEUDa), axis=(2, 3, 4))) + epsilon, 1 / gEUDa)
                    # oarRealMomenta = torch.pow((1 / vols[0, i]) * (torch.sum(torch.pow(oarRealDose - pres, gEUDa), axis=(2, 3, 4))) + epsilon, 1 / gEUDa)
                    # print(oarPredMomenta)
                    # Use (1 / (oar[:, i, ...] == 1).sum()) to count the number of voxels
                    # else:
                    oarPredMomenta = weight * torch.pow((1 / vols[0, i]) * (torch.sum(torch.pow(oarpreddose, gEUDa), axis=(2, 3, 4))) + epsilon, 1 / gEUDa)
                    oarRealMomenta = weight * torch.pow((1 / vols[0, i]) * (torch.sum(torch.pow(oarRealDose, gEUDa), axis=(2, 3, 4))) + epsilon, 1 / gEUDa)
                    oarPredMoment.append(oarPredMomenta)
                    oarRealMoment.append(oarRealMomenta)
        oarPredMoment = torch.stack(oarPredMoment)
        oarRealMoment = torch.stack(oarRealMoment)
        # print(oarPredMoment)
        # oarPredMoment = oarPredMoment + oarPredMomenta

        return self.loss(oarPredMoment, oarRealMoment)  # self.loss(oarPredMoment, oarRealMoment)
        # #self.loss(oarPredMoment, torch.zeros_like(oarPredMoment))
        # #torch.sum(oarPredMoment)#, torch.zeros_like(oarPredMoment))