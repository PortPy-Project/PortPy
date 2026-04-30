# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from portpy.ai.train import main
from portpy.ai.options.train_options import TrainOptions


def train_upkern(args=None):
    """
    Continue training a kernel-3 MedNeXt model as a kernel-5 model using UpKern.
    Works like train(args), so you can pass only overridden options.
    """
    if args is None:
        opt = TrainOptions().parse()
    else:
        default_opt = TrainOptions().parse()
        vars(default_opt).update(args)
        opt = default_opt

        for k, v in vars(opt).items():
            print(f"{k}: {v}")

    # enforce UpKern continuation settings unless user explicitly overrides
    opt.netG = 'mednext'
    opt.load_upkern = True

    if not hasattr(opt, 'mednext_kernel_size') or opt.mednext_kernel_size is None:
        opt.mednext_kernel_size = 5

    if not hasattr(opt, 'mednext_model_id') or opt.mednext_model_id is None:
        opt.mednext_model_id = 'S'

    if not getattr(opt, 'upkern_pretrained_path', None):
        raise ValueError("Please provide 'upkern_pretrained_path' for UpKern continuation training.")

    main(opt)


if __name__ == '__main__':
    train_upkern()