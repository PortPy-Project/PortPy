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

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        parser.add_argument('--dose_loss', type=str, default='mae', choices=['mae', 'mednext_hybrid','mae_moment'])
        parser.add_argument('--lambda_L1', type=float, default=1.0)
        parser.add_argument('--lambda_wL1', type=float, default=1.0)
        parser.add_argument('--lambda_masked_wL1', type=float, default=1.0)
        parser.add_argument('--lambda_moment', type=float, default=0.1)
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser.add_argument('--mednext_model_id', type=str, default='B',
                            choices=['S', 'B', 'M', 'L'],
                            help='MedNeXt model size')
        parser.add_argument('--mednext_kernel_size', type=int, default=3,
                            choices=[3, 5],
                            help='MedNeXt depthwise kernel size')
        parser.add_argument('--upkern_pretrained_path', type=str, default='',
                            help='Path to pretrained kernel-3 checkpoint to initialize kernel-5 model')
        parser.add_argument('--load_upkern', action='store_true',
                            help='Use UpKern to initialize a kernel-5 MedNeXt from a kernel-3 checkpoint')
        parser.add_argument("--lambda_bg", type=float, default=0.0005)
        parser.add_argument("--beamlet_y_max", type=float, default=0.7)
        parser.add_argument("--d1_norm", type=float, default=1600.0)
        parser.add_argument("--d2_norm", type=float, default=600.0)
        parser.add_argument("--d1_out_norm", type=float, default=800.0)
        parser.add_argument(
            "--test_task",
            type=str,
            default="dose_prediction",
            choices=["dose_prediction", "beamlet_dose_prediction"],
            help="Task-specific testing behavior."
        )
        self.isTrain = False
        return parser
