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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:16:11 2020

@author: ndahiya
"""

import os.path
from portpy.ai.data.base_dataset import BaseDataset, get_params, get_transform, transform_3d_data
from portpy.ai.data.image_folder import make_dataset
import numpy as np
import torch


class DosePred3DDataset(BaseDataset):
    """
  A dataset class for 2D CT to Dose prediction dataset.
  
  It assumes that the directory '/path/to/data/train' contains *.npz images which
  have at least two arrays named 'CT', 'DOSE'. Assuming data is already the desired size.
  
  During test time, you need to prepare a directory '/path/to/data/test'.
  """

    def __init__(self, opt):
        """Initialize this dataset class.

    Parameters:
    opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
    """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.phase = self.opt.phase
        if hasattr(self.opt, 'augment'):
            self.augment = self.opt.augment
        else:
            self.augment = False  # default to False even transform is not present


    def __getitem__(self, index):
        """Return a data point and its metadata information.

    Parameters:
      index - - a random integer for data indexing

    Returns a dictionary that contains A, B, A_paths and B_paths
      A (tensor) - - an image in the input domain
      B (tensor) - - its corresponding image in the target domain
      A_paths (str) - - image paths
      B_paths (str) - - image paths (same as A_paths)
      """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = np.load(AB_path, allow_pickle=True)  ##Added (Gourav)

        # Get AB image into A and B
        A = AB['CT']
        B = AB['DOSE']
        OAR = AB['OAR']
        PTV = AB['PTV']
        # hist = AB['HIST']
        # bins = AB['BINS']
        beam = AB['BEAM']

        # apply the same transform to both A and B
        if self.phase == 'train':
            # A, B, OAR, beam, hist, bins = transform_3d_data(A, B, OAR, PTV, beam, hist, bins, transform=False)

            A, B, OAR, beam = transform_3d_data(A, B, OAR, PTV, beam, augment=self.transform)
            A = torch.unsqueeze(A, dim=0)  # Add channel dimensions as data is 3D
            B = torch.unsqueeze(B, dim=0)
            beam = torch.unsqueeze(beam, dim=0)

            A = torch.cat((A, beam, OAR), axis=0)
            # A = torch.cat((A, OAR), axis=0)  # No Beam
            # print(OAR.shape, hist.shape, bins.shape)
        else:
            # A, B, OAR, beam, hist, bins = transform_3d_data(A, B, OAR, PTV, beam, hist, bins, transform=False)
            A, B, OAR, beam = transform_3d_data(A, B, OAR, PTV, beam, augment=False)
            A = torch.unsqueeze(A, dim=0)  # Add channel dimensions as data is 3D
            B = torch.unsqueeze(B, dim=0)
            beam = torch.unsqueeze(beam, dim=0)
            A = torch.cat((A, beam, OAR), axis=0)
            # A = torch.cat((A, OAR), axis=0)  # No beam
            # print(OAR.max())
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path} #'HIST': hist, 'BINS': bins}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
