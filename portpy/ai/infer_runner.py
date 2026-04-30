import os

from portpy.ai.options.test_options import TestOptions
from portpy.ai.data import create_dataset
from portpy.ai.models import create_model
from portpy.ai.util.visualizer import save_images
from portpy.ai.util import html

# Match the output naming used elsewhere
npy_out_fnames = ['CT2DOSE']


def build_infer_opt(args: dict):
    """
    Build a test/inference options object from a dict.
    """
    opt = TestOptions().parse()
    vars(opt).update(args)

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    return opt


class InferenceRunner:
    """
    Load model once, run inference many times.
    """

    def __init__(self, args: dict):
        self.opt = build_infer_opt(args)
        self.model = create_model(self.opt)
        self.model.setup(self.opt)

        if self.opt.eval:
            self.model.eval()

    def run(self, dataroot: str):
        """
        Run inference on the given dataroot and save outputs
        exactly like test.py would.
        """
        self.opt.dataroot = dataroot
        dataset = create_dataset(self.opt)

        web_dir = os.path.join(
            self.opt.results_dir,
            self.opt.name,
            '{}_{}'.format(self.opt.phase, self.opt.epoch)
        )
        if self.opt.load_iter > 0:
            web_dir = '{:s}_iter{:d}'.format(web_dir, self.opt.load_iter)

        print('creating web directory', web_dir)
        webpage = html.HTML(
            web_dir,
            'Experiment = %s, Phase = %s, Epoch = %s'
            % (self.opt.name, self.opt.phase, self.opt.epoch)
        )

        for i, data in enumerate(dataset):
            self.model.set_input(data)
            self.model.test()

            visuals = self.model.get_current_visuals()
            img_path = self.model.get_image_paths()

            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))

            save_images(
                webpage,
                visuals,
                img_path,
                aspect_ratio=self.opt.aspect_ratio,
                width=self.opt.display_winsize,
                npy_out_fnames=npy_out_fnames
            )

        webpage.save()