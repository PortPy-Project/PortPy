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
import torch.nn as nn
import random



class DosePrediction3DModel(BaseModel):
    """ This class implements the generic model (using class structure of pix2pix), for learning a mapping from
    ct images to a dose map.

    By default, it uses a '--netG unet256' U-Net model generator (no discriminator)
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
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='dosepred3d')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_MAE']
        #self.loss_names = ['G_MSE']
        #self.loss_names = ['G_DVH']
        # self.loss_names = ['G_MOMENT']
        #self.loss_names = ['G_Bhatt']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_CT', 'fake_Dose', 'real_Dose']
        self.epoch_num = 0
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain: # Only G during both test and train times
            self.model_names = ['G']
        else:
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionMAE = torch.nn.L1Loss()
            #self.criterionMSE = torch.nn.MSELoss()
            #self.criterionDVH = networks.DVHLoss().to(self.device)
            # self.criterionMoment = networks.MomentLoss().to(self.device)
            #self.criterionBHATT = networks.BhattLoss().to(self.device)

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
        self.real_CT = input['A' if AtoB else 'B'].to(self.device)
        self.real_Dose = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # self.target_hist = input['HIST'].to(self.device)
        # self.target_bins = input['BINS'].to(self.device)

        #print(self.real_CT.dtype, self.real_Dose.dtype, self.real_CT.shape)
        #print(self.real_CT.shape, self.target_hist.shape, self.target_bins.shape)
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_Dose = self.netG(self.real_CT)  # pred_dose = netG(CT)

    def backward_G(self):
        """Calculate L1 loss for the generator"""
        self.loss_G_MAE = self.criterionMAE(self.fake_Dose, self.real_Dose) * self.opt.lambda_L1
        # self.loss_G_MSE = self.criterionMSE(self.fake_Dose, self.real_Dose) * self.opt.lambda_L1
        #self.loss_G_DVH = 10*self.criterionDVH(self.fake_Dose, self.target_hist, self.target_bins, self.real_CT[:, 1:,...])#Changed(Gourav)
        # self.loss_G_MOMENT = 0.1 * self.criterionMoment(self.fake_Dose, self.real_CT[:, 2:, ...], self.real_Dose)  # Changed for no beam(Gourav)
        #self.loss_G_DVH = 10 * self.criterionDVH(self.fake_Dose, self.target_hist, self.target_bins, self.real_CT[:, 2:, ...])
        #self.loss_G_Bhatt = 10 * self.criterionBHATT(self.fake_Dose, self.target_hist, self.target_bins, self.real_CT[:, 2:, ...])


        # if self.epoch_num > 10:
        #     if self.loss_G_MAE > 6:
        #         with open('High_MAE_loss.txt', 'a') as f:
        #             print(self.image_paths, file=f)
        
        self.loss_G = self.loss_G_MAE #+ self.loss_G_MOMENT#  + self.loss_G_DVH + self.loss_G_MOMENT Could add other losses here like DVH, RMSE or SSIM
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute output predicted dose: G(A)

        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        # if self.epoch_num < 10:
        #     nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=0.05, norm_type=2)
        #nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=0.05, norm_type=2)
        self.optimizer_G.step()

    def calculate_validation_loss(self):
        self.loss_G_MAE = self.criterionMAE(self.fake_Dose, self.real_Dose) * self.opt.lambda_L1
        # self.loss_G_MSE = self.criterionMSE(self.fake_Dose, self.real_Dose) * self.opt.lambda_L1
        # self.loss_G_DVH = 10*self.criterionDVH(self.fake_Dose, self.target_hist, self.target_bins, self.real_CT[:, 1:,...])#Changed(Gourav)
        # self.loss_G_MOMENT = 0.1 * self.criterionMoment(self.fake_Dose, self.real_CT[:, 2:, ...], self.real_Dose)  # Change
        self.loss_G = self.loss_G_MAE #+ self.loss_G_MOMENT