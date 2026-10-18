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

import json

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

###############################################################################
# Functions
###############################################################################

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
             gpu_ids=None, mednext_model_id='B', mednext_kernel_size=3,
             mednext_n_channels=32, mednext_sites=('lung', 'prostate'), netG_cfg=None):
    """Build a generator. ``netG`` is the only thing most users need to change.

    To try a different model, set ``netG`` in your training/testing options; no
    other file needs editing. Available generators:

    ===================== ============================================= ===========================
    netG                  model                                         defined in
    ===================== ============================================= ===========================
    ``unet_128``          3D U-Net, 7 downs (the historical default)     ``unet_generator3d.py``
    ``unet_256``          3D U-Net, 8 downs                              ``unet_generator3d.py``
    ``mednext``           MedNeXt dose model (GDP-HMM Top-1 style)       ``mednext_dose.py``
    ``mednext_attn``      MedNeXt + attention-gated skips (Top-2)        ``mednext_attention.py``
    ``mednext_duo``       MedNeXt, shared encoder + per-site decoders    ``mednext_duo.py``
    ``beamlet_unet3d``    beamlet-dose U-Net (patient grid)              ``beamlet_unet3d.py``
    ``beamlet_vit3d``     beamlet-dose 3D ViT encoder (patient grid)     ``beamlet_vit3d.py``
    ``beamlet_swin_unetr3d`` beamlet-dose 3D Swin encoder (patient grid) ``beamlet_swin_unetr3d.py``
    ``beamlet_ray_unet3d`` beamlet-dose U-Net (BEV / ray patch)          ``beamlet_ray_unet3d.py``
    ``beamlet_ray_attention_resunet3d`` attention ResU-Net on a BEV patch ``beamlet_ray_attention_resunet3d.py``
    ===================== ============================================= ===========================

    Every beamlet-dose generator is prefixed ``beamlet_``; the ``_ray_`` ones work on a
    BEV / ray-aligned patch, the others on the patient grid. The former names
    ``beamlet_unet``, ``ray_unet3d`` and ``ray_attention_resunet3d`` are still accepted
    as deprecated aliases.

    The two transformer generators are PATIENT-GRID ONLY: their defaults are tied to
    the (96, 192, 224) grid, and ``beamlet_swin_unetr3d`` asserts that the grid divides
    exactly by its patch and window sizes. Override any constructor argument through
    ``netG_cfg``, e.g. a parameter-matched ViT against ``beamlet_unet3d``'s 22.58 M::

        define_G(4, 1, 32, 'beamlet_vit3d',
                 netG_cfg={'embed_dim': 288, 'depth': 8, 'num_heads': 8})

    :param input_nc: number of input channels (8 for PortPy dose prediction)
    :param output_nc: number of output channels (1 for a dose map)
    :param ngf: base filter count, used by the U-Net generators
    :param netG: generator name from the table above
    :param norm: normalization for ``unet_128`` / ``unet_256`` ONLY; every other
        model here uses its own GroupNorm and ignores this
    :param mednext_model_id: MedNeXt size preset for ``mednext`` ('S'/'B'/'M'/'L')
    :param mednext_kernel_size: MedNeXt conv kernel; the UpKern schedule walks 3 -> 5 -> 7
    :param mednext_n_channels: base filters for ``mednext_attn`` / ``mednext_duo``
        (32 = MedNeXt-B at 10.5 M params, 16 = the lightweight Top-3 setting)
    :param mednext_sites: decoder names for ``mednext_duo``, one per treatment site
    :param netG_cfg: dict of constructor overrides for the generator, used by
        ``beamlet_vit3d`` and ``beamlet_swin_unetr3d``. None means the published
        defaults (ViT-B 110 M / Swin f96 59.8 M).
    :return: the initialised network

    :Example:

    >>> net = define_G(8, 1, 32, 'mednext_attn')                     # doctest: +SKIP
    >>> net = define_G(8, 1, 32, 'mednext_duo', mednext_n_channels=16)  # doctest: +SKIP
    """
    if gpu_ids is None:
        gpu_ids = []
    if isinstance(netG_cfg, str):      # from the command line it arrives as JSON text
        netG_cfg = json.loads(netG_cfg) if netG_cfg.strip() else {}
    net = None
    if netG in ('unet_128', 'unet_256'):
        # `norm` applies to these two generators only; every other model below
        # brings its own normalization (MedNeXt and the beamlet U-Nets use GroupNorm).
        num_downs = 7 if netG == 'unet_128' else 8
        net = UnetGenerator3d(input_nc, output_nc, num_downs, ngf,
                              norm_layer=get_norm_layer(norm_type=norm),
                              use_dropout=use_dropout)
    elif netG == 'mednext':
        net = MedNeXtDose(
            in_channels=input_nc,
            out_channels=output_nc,
            model_id=mednext_model_id,
            kernel_size=mednext_kernel_size,
        )
    elif netG == 'mednext_attn':
        # GDP-HMM Top-2 (Xiong & Cai): MedNeXt with attention-gated skips.
        from portpy.ai.models.mednext_attention import MedNeXtDoseAttention
        net = MedNeXtDoseAttention(
            in_channels=input_nc,
            out_channels=output_nc,
            n_channels=mednext_n_channels,
            kernel_size=mednext_kernel_size,
        )
    elif netG == 'mednext_duo':
        # GDP-HMM Top-3 (Gao et al.): shared encoder, one decoder per site.
        from portpy.ai.models.mednext_duo import MedNeXtDuo
        net = MedNeXtDuo(
            in_channels=input_nc,
            out_channels=output_nc,
            n_channels=mednext_n_channels,
            kernel_size=mednext_kernel_size,
            sites=mednext_sites,
        )
    elif netG in ("beamlet_unet3d", "beamlet_unet"):   # 'beamlet_unet' = deprecated alias
        net = BeamletUNet3D(input_nc, output_nc)
    elif netG == "beamlet_vit3d":
        # 3D ViT encoder + the same decoder beamlet_unet3d uses, so a comparison against
        # that baseline isolates the encoder. Default is ViT-B scale (110 M).
        from portpy.ai.models.beamlet_vit3d import BeamletViT3D
        cfg = dict(in_channels=input_nc, out_channels=output_nc,
                   img_size=(96, 192, 224), patch_size=16, embed_dim=768, depth=12,
                   num_heads=12, base_features=ngf, num_groups=8)
        cfg.update(netG_cfg or {})
        net = BeamletViT3D(**cfg)
    elif netG == "beamlet_swin_unetr3d":
        # 3D Swin encoder + the same decoder. Default f96 / depths (2,2,18,2) = 59.8 M,
        # which needs ~48 GB at batch 2; see the module docstring for smaller presets.
        from portpy.ai.models.beamlet_swin_unetr3d import BeamletSwinUNETR3D
        cfg = dict(in_channels=input_nc, out_channels=output_nc,
                   img_size=(96, 192, 224), patch_size=2, window_size=(6, 6, 7),
                   feature_size=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
                   base_features=ngf, num_groups=8)
        cfg.update(netG_cfg or {})
        net = BeamletSwinUNETR3D(**cfg)
    elif netG in ("beamlet_ray_unet3d", "ray_unet3d"):   # 'ray_unet3d' = deprecated alias
        # BEV / ray-aligned patch model. Same class RayUNetPredictor loads, so a
        # checkpoint trained here is directly usable by build_ray_unet_predictor
        # (model_type='unet3d', in_channels=3, base_features=ngf).
        from portpy.ai.models.beamlet_ray_unet3d import UNet3D as RayUNet3D
        net = RayUNet3D(in_channels=input_nc, out_channels=output_nc,
                        base_features=ngf, num_groups=8)
    elif netG in ("beamlet_ray_attention_resunet3d", "ray_attention_resunet3d"):
        # As above for build_ray_unet_predictor(model_type='attention_resunet3d').
        from portpy.ai.models.beamlet_ray_attention_resunet3d import AttentionResUNet3D
        net = AttentionResUNet3D(in_channels=input_nc, out_channels=output_nc,
                                 base_features=ngf, num_groups=8,
                                 use_cbam=False, num_levels=3)
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
# Architectures and losses live in their own modules. They are re-exported here
# so existing callers (``networks.MomentLoss``, ``from ... networks3d import
# BeamletUNet3D``) keep working unchanged.
##############################################################################
from portpy.ai.models.unet_generator3d import (  # noqa: E402,F401
    Identity, get_norm_layer, UnetGenerator3d, UnetSkipConnectionBlock3d)
from portpy.ai.models.mednext_dose import MedNeXtDose  # noqa: E402,F401
from portpy.ai.models.beamlet_unet3d import DoubleConv, BeamletUNet3D  # noqa: E402,F401
from portpy.ai.models.losses3d import (  # noqa: E402,F401
    DVHLoss, BhattLoss, WeightedL1Loss, MaskedWeightedL1Loss, MomentLoss,
    DualGradientL2Loss, BerHuLoss)
