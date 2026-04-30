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
            # parser.add_argument('--lambda_L1', type=float, default=1, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_MAE', 'G_wL1', 'G_masked_wL1', 'G_MOMENT', 'G_SHARP']
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
        self.real_CT = input['A' if AtoB else 'B'].to(self.device)
        self.real_Dose = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # self.target_hist = input['HIST'].to(self.device)
        # self.target_bins = input['BINS'].to(self.device)

        #print(self.real_CT.dtype, self.real_Dose.dtype, self.real_CT.shape)
        #print(self.real_CT.shape, self.target_hist.shape, self.target_bins.shape)
        self.body_mask = input.get('BODY', None)
        if self.body_mask is not None:
            self.body_mask = self.body_mask.to(self.device)

        self.prescription_gy = input.get('PRESCRIPTION_GY', None)
        if self.prescription_gy is not None:
            self.prescription_gy = self.prescription_gy.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_Dose = self.netG(self.real_CT)  # pred_dose = netG(CT)

    def backward_G(self):
        """Calculate generator loss"""

        if self.opt.dose_loss == 'mae':
            # Original PortPy behavior: plain MAE only
            self.loss_G_MAE = self.criterionMAE(self.fake_Dose, self.real_Dose) * self.opt.lambda_L1
            self.loss_G_wL1 = self.real_Dose.new_tensor(0.0)
            self.loss_G_masked_wL1 = self.real_Dose.new_tensor(0.0)
            self.loss_G_MOMENT = self.real_Dose.new_tensor(0.0)
            self.loss_G_SHARP = self.real_Dose.new_tensor(0.0)

            self.loss_G = self.loss_G_MAE

        elif self.opt.dose_loss == 'mednext_hybrid':
            # Current PortPy input layout:
            # [CT, BEAM, OAR channels..., PTV]
            structures = self.real_CT[:, 2:, ...]  # OAR + PTV channels

            # Minimal body-mask surrogate for now
            body_mask = (self.body_mask > 0.5).float()

            self.loss_G_MAE = self.real_Dose.new_tensor(0.0)
            #     # self.loss_G_MSE = self.criterionMSE(self.fake_Dose, self.real_Dose) * self.opt.lambda_L1
            #     #self.loss_G_DVH = 10*self.criterionDVH(self.fake_Dose, self.target_hist, self.target_bins, self.real_CT[:, 1:,...])#Changed(Gourav)
            #     # self.loss_G_MOMENT = 0.1 * self.criterionMoment(self.fake_Dose, self.real_CT[:, 2:, ...], self.real_Dose)  # Changed for no beam(Gourav)

            self.loss_G_wL1 = self.criterionWeightedL1(
                self.fake_Dose, self.real_Dose
            ) * self.opt.lambda_wL1

            self.loss_G_masked_wL1 = self.criterionMaskedWeightedL1(
                self.fake_Dose, self.real_Dose, body_mask
            ) * self.opt.lambda_masked_wL1

            self.loss_G_MOMENT = self.criterionMoment(
                self.fake_Dose, self.real_Dose, {"OARPTV": structures}
            ) * self.opt.lambda_moment

            self.loss_G_SHARP = self.criterionSharp(
                self.fake_Dose, self.real_Dose
            )

            self.loss_G = (
                    # self.loss_G_MAE +
                    self.loss_G_wL1 +
                    self.loss_G_masked_wL1 +
                    self.loss_G_MOMENT +
                    self.loss_G_SHARP
            )
        elif self.opt.dose_loss == 'mae_moment':
            structures = self.real_CT[:, 2:, ...]  # OAR + PTV channels
            # Original PortPy behavior: plain MAE only
            self.loss_G_MAE = self.criterionMAE(self.fake_Dose, self.real_Dose) * self.opt.lambda_L1
            self.loss_G_wL1 = self.real_Dose.new_tensor(0.0)
            self.loss_G_masked_wL1 = self.real_Dose.new_tensor(0.0)
            self.loss_G_MOMENT = self.criterionMoment(
                self.fake_Dose, self.real_Dose, {"OARPTV": structures}
            ) * self.opt.lambda_moment
            self.loss_G_SHARP = self.real_Dose.new_tensor(0.0)

            self.loss_G = self.loss_G_MAE + self.loss_G_MOMENT
        else:
            raise ValueError(f"Unsupported dose_loss: {self.opt.dose_loss}")

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
        if self.opt.dose_loss == 'mae':
            # Original PortPy behavior: plain MAE only
            self.loss_G_MAE = self.criterionMAE(self.fake_Dose, self.real_Dose) * self.opt.lambda_L1
            self.loss_G_wL1 = self.real_Dose.new_tensor(0.0)
            self.loss_G_masked_wL1 = self.real_Dose.new_tensor(0.0)
            self.loss_G_MOMENT = self.real_Dose.new_tensor(0.0)
            self.loss_G_SHARP = self.real_Dose.new_tensor(0.0)

            self.loss_G = self.loss_G_MAE

        elif self.opt.dose_loss == 'mednext_hybrid':
            structures = self.real_CT[:, 2:, ...]
            body_mask = (self.body_mask > 0.5).float()

            self.loss_G_MAE = self.real_Dose.new_tensor(0.0)

            self.loss_G_wL1 = self.criterionWeightedL1(
                self.fake_Dose, self.real_Dose
            ) * self.opt.lambda_wL1

            self.loss_G_masked_wL1 = self.criterionMaskedWeightedL1(
                self.fake_Dose, self.real_Dose, body_mask
            ) * self.opt.lambda_masked_wL1

            self.loss_G_MOMENT = self.criterionMoment(
                self.fake_Dose, self.real_Dose, {"OARPTV": structures}
            ) * self.opt.lambda_moment

            self.loss_G_SHARP = self.criterionSharp(
                self.fake_Dose, self.real_Dose
            )

            self.loss_G = (
                    self.loss_G_wL1 +
                    self.loss_G_masked_wL1 +
                    self.loss_G_MOMENT +
                    self.loss_G_SHARP
            )
        elif self.opt.dose_loss == 'mae_moment':
            structures = self.real_CT[:, 2:, ...]
            # body_mask = (self.body_mask > 0.5).float()
            # Original PortPy behavior: plain MAE only
            self.loss_G_MAE = self.criterionMAE(self.fake_Dose, self.real_Dose) * self.opt.lambda_L1
            self.loss_G_wL1 = self.real_Dose.new_tensor(0.0)
            self.loss_G_masked_wL1 = self.real_Dose.new_tensor(0.0)
            self.loss_G_MOMENT = self.criterionMoment(
                self.fake_Dose, self.real_Dose, {"OARPTV": structures}
            ) * self.opt.lambda_moment
            self.loss_G_SHARP = self.real_Dose.new_tensor(0.0)

            self.loss_G = self.loss_G_MAE + self.loss_G_MOMENT
        else:
            raise ValueError(f"Unsupported dose_loss: {self.opt.dose_loss}")

    # def calculate_validation_mae(self):
    #     """
    #     Challenge-style MAE using a simple body surrogate.
    #     Returns scalar tensor.
    #     """
    #     body_mask = (self.real_Dose > 0).float()
    #     abs_err = torch.abs(self.fake_Dose - self.real_Dose) * body_mask
    #     denom = body_mask.sum().clamp(min=1.0)
    #     return abs_err.sum() / denom

    def calculate_validation_mae(self):
        """
        Challenge-style MAE:
          1) use BODY mask if available, otherwise CT-based surrogate
          2) compute GT D97 inside PTV
          3) scale pred and gt by prescription / D97_GT
          4) average absolute error only on BODY ∩ (pred>5 Gy or gt>5 Gy)
        """
        pred = self.fake_Dose
        gt = self.real_Dose

        # BODY mask preferred; fallback to CT-based surrogate
        if hasattr(self, 'body_mask') and self.body_mask is not None:
            body_mask = (self.body_mask > 0.5)
        else:
            body_mask = (self.real_CT[:, 0:1, ...] > 0)

        # assume last structure channel corresponds to PTV in your current layout
        ptv_mask = (self.real_CT[:, -1:, ...] > 0.5)

        gt_ptv = gt[ptv_mask]
        if gt_ptv.numel() == 0:
            return pred.new_tensor(0.0)

        if hasattr(self, 'prescription_gy') and self.prescription_gy is not None:
            prescribed_dose = self.prescription_gy.view(-1, 1, 1, 1, 1)
        else:
            # fallback only if metadata missing
            prescribed_dose = gt.new_tensor(1.0)

        d97_gt = torch.quantile(gt_ptv, 0.03)
        scale = prescribed_dose / (d97_gt + 1e-5)

        pred_scaled = pred * scale
        gt_scaled = gt * scale

        isodose_0Gy_mask = ((gt_scaled > 0) | (pred_scaled > 0)) & body_mask

        masked_abs = torch.abs(gt_scaled - pred_scaled)[isodose_0Gy_mask]
        if masked_abs.numel() == 0:
            return pred.new_tensor(0.0)

        return masked_abs.mean()

    def load_upkern_weights(self, pretrained_path):
        """
        Initialize current kernel-5 MedNeXt model from a pretrained kernel-3 checkpoint using UpKern.
        Assumes self.netG is already the target kernel-5 model.
        """
        if self.opt.netG != 'mednext':
            raise ValueError("UpKern is only supported for netG='mednext'.")

        try:
            from nnunet_mednext import create_mednext_v1
            from nnunet_mednext.run.load_weights import upkern_load_weights
        except ImportError as e:
            raise ImportError(
                "Could not import MedNeXt UpKern utilities. "
                "Install MedNeXt with `pip install -e .` from the MedNeXt repo "
                "or `pip install git+https://github.com/MIC-DKFZ/MedNeXt.git`."
            ) from e

        # build matching kernel-3 source architecture
        src_model = create_mednext_v1(
            num_input_channels=self.opt.input_nc,
            num_classes=self.opt.output_nc,
            model_id=self.opt.mednext_model_id,
            kernel_size=3,
            deep_supervision=False
        )

        checkpoint = torch.load(pretrained_path, map_location=self.device)

        # Handle both raw state_dict checkpoints and PortPy/BaseModel-style checkpoints
        if isinstance(checkpoint, dict):
            if 'G' in checkpoint:
                state_dict = checkpoint['G']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            nk = k

            if nk.startswith('module.'):
                nk = nk[7:]

            if nk.startswith('model.'):
                nk = nk[6:]

            cleaned_state_dict[nk] = v

        src_model.load_state_dict(cleaned_state_dict, strict=True)

        target_model = self.netG.module if hasattr(self.netG, 'module') else self.netG
        upkern_load_weights(target_model, src_model)