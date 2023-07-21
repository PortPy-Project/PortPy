"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def tensor2im3d(image_tensor):
    image_numpy = image_tensor[0].cpu().float().numpy()
    return image_numpy
  
def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_image_numpy(image_numpy, image_path):
    """Save a numpy image to the disk as float npz (label: 'CBC2CT')

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    #print_numpy(image_numpy, shp=True)
    np.savez(image_path, CBCT2CT=image_numpy)

def save_image_numpy_multitask(npy_images, save_paths):
    """Save a numpy image to the disk as float npz (label: 'CBC2CT', 'ESO', 'GTV', 'DOSE' etc.)

    Parameters:
        npy_images (numpy array) -- list of input numpy arrays
        image_path (str)         -- list of paths the path of the image
        npy_labels (str)         -- labels for the npz arrays to be written to disk
    """
    #for image_numpy in npy_images:
    #  print_numpy(image_numpy, shp=True)
    image_path = save_paths[0]
    
    # if len(npy_labels) == 1:
    #   np.savez(image_path, CBCT2CT=npy_images[0])
    # elif len(npy_labels) == 2:
    #   np.savez(image_path, CBCT2CT=npy_images[0], ESO=npy_images[1])
    # elif len(npy_labels) == 3:
    #   np.savez(image_path, CBCT2CT=npy_images[0], ESO=npy_images[1], GTV=npy_images[2])
    # else:
    #   print("Can't save numpy images.")
    np.savez(image_path, **npy_images)
    
def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
