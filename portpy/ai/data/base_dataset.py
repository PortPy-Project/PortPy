"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from abc import ABC, abstractmethod
import torch
import warnings

warnings.filterwarnings("ignore")


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def transform_ct_dose_pair2D(ct_arr, dose_arr, oar_arr, ptv_arr, apply_transform=False):
    """Apply several (same) transforms to ct/dose pair which are numpy arrays.
  """

    oar_arr[np.where(ptv_arr > 0)] = 0
    ptv_arr = ptv_arr * (62 / 74)

    ct_pil = TF.to_pil_image(ct_arr, mode='F')
    dose_pil = TF.to_pil_image(dose_arr, mode='F')
    oar_pil = TF.to_pil_image(oar_arr, mode='F')
    ptv_pil = TF.to_pil_image(ptv_arr, mode='F')

    if apply_transform is True:
        if random.random() < 0.5:  # 50 % chance of applying random transformation

            scale, angle, h, w, x, y = 1, 0, 0, 0, 0, 0

            # Rotation +/- 25 degrees
            if random.random() < 0.5:
                angle = random.uniform(-25.0, 25.0)
            # Scale 0.8/1.2
            if random.random() < 0.5:
                scale = random.uniform(0.8, 1.2)
            # translate -10 to 10 pixels
            if random.random() < 0.5:
                h = random.uniform(-10, 10)
                w = random.uniform(-10, 10)
            # Shear +/- 8 degrees
            if random.random() < 0.5:
                x = random.uniform(-8, 8)
                y = random.uniform(-8, 8)

            # Apply transforms

            ct_pil = TF.affine(ct_pil, angle=angle, translate=(w, h), scale=scale, shear=(x, y),
                               resample=Image.BILINEAR)
            dose_pil = TF.affine(dose_pil, angle=angle, translate=(w, h), scale=scale, shear=(x, y),
                                 resample=Image.BILINEAR)
            oar_pil = TF.affine(oar_pil, angle=angle, translate=(w, h), scale=scale, shear=(x, y),
                                resample=Image.NEAREST)
            ptv_pil = TF.affine(ptv_pil, angle=angle, translate=(w, h), scale=scale, shear=(x, y),
                                resample=Image.NEAREST)

    ct_arr = TF.to_tensor(ct_pil)
    dose_arr = TF.to_tensor(dose_pil)
    oar_arr = TF.to_tensor(oar_pil).to(dtype=torch.int64)
    ptv_arr = TF.to_tensor(ptv_pil)

    oar_arr = torch.nn.functional.one_hot(oar_arr, 6)  # 6 classes possible, BG, Eso, Cord, Heart, Lung_L, Lung_R
    oar_arr = torch.squeeze(oar_arr, axis=0).permute(2, 0, 1).to(torch.float) * (1 / 74)
    # print(oar_arr.min(), oar_arr.max())
    return ct_arr, dose_arr, oar_arr, ptv_arr


def transform_3d_data(ct_arr, dose_arr, oar_arr, ptv_arr, beam_arr, transform=False):
    ct = torch.from_numpy(ct_arr)
    dose = torch.from_numpy(dose_arr)
    beam = torch.from_numpy(beam_arr)
    # hist = torch.from_numpy(hist_arr)
    # bins = torch.from_numpy(bins_arr)
    oar = torch.from_numpy(oar_arr).long()
    ptv = torch.from_numpy(ptv_arr).unsqueeze(dim=0)
    # ptv *= 60  # For no beam case

    oar = torch.nn.functional.one_hot(oar, 6)[..., 1:]  # Remove BG
    oar = oar.permute(3, 0, 1, 2).to(torch.float)  # Channels first
    ptv = ptv.to(torch.float)  # added as we need both in float to cat(Gourav)
    oar = torch.cat((oar, ptv), axis=0)
    return ct, dose, oar, beam  # , hist, bins

def transform_3d_data_test(ct_arr, oar_arr, ptv_arr, beam_arr, transform=False):
    ct = torch.from_numpy(ct_arr)
    # dose = torch.from_numpy(dose_arr)
    beam = torch.from_numpy(beam_arr)
    # hist = torch.from_numpy(hist_arr)
    # bins = torch.from_numpy(bins_arr)
    oar = torch.from_numpy(oar_arr).long()
    ptv = torch.from_numpy(ptv_arr).unsqueeze(dim=0)
    # ptv *= 60  # For no beam case

    oar = torch.nn.functional.one_hot(oar, 6)[..., 1:]  # Remove BG
    oar = oar.permute(3, 0, 1, 2).to(torch.float)  # Channels first
    ptv = ptv.to(torch.float)  # added as we need both in float to cat(Gourav)
    oar = torch.cat((oar, ptv), axis=0)
    return ct, oar, beam  # , hist, bins