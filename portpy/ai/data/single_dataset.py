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

from portpy.ai.data.base_dataset import BaseDataset, get_transform
from portpy.ai.data.image_folder import make_dataset
import torch
import numpy as np
from portpy.ai.data.base_dataset import transform_3d_data_test
import os

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot)  # get the directory where images are located
        # self.dir_A = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        # self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        AB = np.load(A_path)  ##Added (Gourav)

        # Get AB image into A and B
        A = AB['CT']
        OAR = AB['OAR']
        PTV = AB['PTV']
        # hist = AB['HIST']
        # bins = AB['BINS']
        beam = AB['BEAM']

        A, OAR, beam = transform_3d_data_test(A, OAR, PTV, beam, transform=False)
        A = torch.unsqueeze(A, dim=0)  # Add channel dimensions as data is 3D
        # B = torch.unsqueeze(B, dim=0)
        beam = torch.unsqueeze(beam, dim=0)

        A = torch.cat((A, beam, OAR), axis=0)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
