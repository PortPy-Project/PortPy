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
from portpy.ai.models.base_model import BaseModel
from portpy.ai.models import networks3d as networks


class BeamletDose3DModel(BaseModel):
    """ This class implements the generic model (using class structure of pix2pix), for learning a mapping from
    ct images to a dose map.

    By default, it uses a '--netG unet128' U-Net model generator (no discriminator)
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(
            norm="batch",
            netG="beamlet_unet",
            dataset_mode="beamletdose3d",
        )

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            # parser.add_argument('--lambda_L1', type=float, default=1, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_hot", "G_scatter", "G_bg"]
        self.visual_names = ["real_CT", "fake_Dose", "real_Dose"]

        self.epoch_num = 0
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain: # Only G during both test and train times
            self.model_names = ['G']
        else:
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, mednext_model_id=opt.mednext_model_id, mednext_kernel_size=opt.mednext_kernel_size)


        self.criterionMAE = torch.nn.L1Loss()
        self.criterionWeightedL1 = networks.WeightedL1Loss().to(self.device)
        self.criterionMaskedWeightedL1 = networks.MaskedWeightedL1Loss().to(self.device)
        self.criterionMoment = networks.MomentLoss(moments=(1, 2, 10)).to(self.device)
        self.criterionSharp = networks.DualGradientL2Loss(
            gamma_edge=25,
            gamma_flat=0,
            lambda_edge=0.1,
            lambda_flat=0.05
        ).to(self.device)
        self.criterionBerhu = networks.BerHuLoss().to(self.device)

        if self.isTrain and opt.netG == 'mednext' and opt.load_upkern:
            if not opt.upkern_pretrained_path:
                raise ValueError("load_upkern=True but no --upkern_pretrained_path was provided.")
            print(f"Initializing kernel-{opt.mednext_kernel_size} MedNeXt from kernel-3 checkpoint using UpKern.")
            self.load_upkern_weights(opt.upkern_pretrained_path)
        if self.isTrain:
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_epoch(self, epoch):
        self.epoch_num = epoch
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_CT = input["A"].to(self.device)
        self.real_Dose = input["B"].to(self.device)
        self.hot_mask = input["HOT_MASK"].to(self.device)
        self.scatter_mask = input["SCATTER_MASK"].to(self.device)
        self.image_paths = input["A_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_Dose = self.netG(self.real_CT)  # pred_dose = netG(CT)

    def _beamlet_loss(self):
        hot_mask = self.hot_mask
        scatter_mask = self.scatter_mask
        combined = torch.clamp(hot_mask + scatter_mask, 0, 1)
        other_mask = 1.0 - combined

        self.loss_G_hot = self.criterionBerhu(self.fake_Dose, self.real_Dose, hot_mask)
        self.loss_G_scatter = 1.5 * self.criterionBerhu(self.fake_Dose, self.real_Dose, scatter_mask)
        self.loss_G_bg = self.opt.lambda_bg * self.criterionBerhu(self.fake_Dose, self.real_Dose, other_mask)

        self.loss_G = self.loss_G_hot + self.loss_G_scatter + self.loss_G_bg
        return self.loss_G

    def backward_G(self):
        self._beamlet_loss()
        self.loss_G.backward()

    def calculate_validation_loss(self):
        self._beamlet_loss()

    def optimize_parameters(self):
        self.forward()                   # compute output predicted dose: G(A)

        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        # if self.epoch_num < 10:
        #     nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=0.05, norm_type=2)
        #nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=0.05, norm_type=2)
        self.optimizer_G.step()

    def calculate_validation_metrics(self):
        mask = torch.clamp(self.hot_mask + self.scatter_mask, 0, 1)

        p = self.fake_Dose * mask
        t = self.real_Dose * mask
        n = mask.sum().clamp_min(1.0)

        mae = torch.abs(p - t).sum() / n

        t_mean = t.sum() / n
        sst = ((t - t_mean) ** 2).sum().clamp_min(1e-8)
        sse = ((t - p) ** 2).sum()
        r2 = 1.0 - sse / sst

        pm = p.sum() / n
        tm = t_mean

        corr = ((p - pm) * (t - tm)).sum()
        corr = corr / (
                ((p - pm) ** 2).sum().sqrt().clamp_min(1e-8)
                * ((t - tm) ** 2).sum().sqrt().clamp_min(1e-8)
        )

        rmse = (sse / n).sqrt()
        nrmse = rmse / (t.max() - t.min()).clamp_min(1e-8)

        return {
            "mae": mae.detach(),
            "r2": r2.detach(),
            "corr": corr.detach(),
            "nrmse": nrmse.detach(),
        }

    def calculate_validation_mae(self):
        return self.calculate_validation_metrics()["mae"]