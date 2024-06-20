import torchvision, glob
import torch.utils.data as data
from PIL import ImageFilter, ImageOps
import numpy as np
import random, pdb
import torch,os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

class SimpleDataLoader(Dataset):
    def __init__(self, data_path, img_transform):
        self.img_path = glob.glob(os.path.join(data_path + '/*.jpg')) + glob.glob(os.path.join(data_path + '/*.png'))
        self.img_path.sort()
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.img_transform(img)
        return index, img

def random_crop_resize_affine(x1,x2,y1,y2,width,height):
    affine = np.eye(3)
    affine[0,0] = (x2 - x1) / (width - 1)
    affine[1,1] = (y2 - y1) / (height - 1)
    affine[0,2] = (x1 + x2 - width + 1) / (width - 1)
    affine[1,2] = (y1 + y2 - height + 1) / (height - 1)
    return affine

def get_params(height, width, scale, ratio, num_of_sample):
    area = height * width
    log_ratio = torch.log(torch.tensor(ratio))
    target_area = area * torch.empty(num_of_sample).uniform_(scale[0], scale[1])
    aspect_ratio = torch.exp(torch.empty(num_of_sample).uniform_(log_ratio[0], log_ratio[1]))

    w = torch.sqrt(target_area * aspect_ratio).round().int()
    h = torch.sqrt(target_area / aspect_ratio).round().int()

    valid = (w > 0).float() * (w <= width).float() * (h > 0).float() * (h <= height).float()
    valid_h = h[valid > 0]
    valid_w = w[valid > 0]
    i = (torch.rand(valid_h.shape[0]) * (height - valid_h + 1)).int().clamp(0, height)
    j = (torch.rand(valid_w.shape[0]) * (width - valid_w + 1)).int().clamp(0, width)
    return torch.stack([j,i, valid_w + j,valid_h + i], dim = -1)

def crop_resize_affine(x1,x2,y1,y2,width,height):
    affine = np.eye(3)
    affine[0,0] = (x2 - x1) / (width - 1)
    affine[1,1] = (y2 - y1) / (height - 1)
    affine[0,2] = (x1 + x2 - width + 1) / (width - 1)
    affine[1,2] = (y1 + y2 - height + 1) / (height - 1)
    return torch.from_numpy(affine)

def compute_flip_affine(hori = True):
    affine = np.eye(3)
    if hori:
        affine[0,0] = -1
        affine[0,2] = 0
    else:
        affine[1,1] = -1
        affine[1,2] = 0
    return affine

def apply_affine(img, affine, out_size = None, mode = 'bilinear', padding_mode = "zeros"):
    if out_size is None:
        out_size = img.shape
    elif isinstance(out_size, int):
        out_size = torch.Size([img.shape[0], img.shape[1], out_size, out_size])
    elif isinstance(out_size, tuple):
        out_size = torch.Size([img.shape[0], img.shape[1], out_size[0], out_size[1]])
    grid = F.affine_grid(affine.float(), out_size, align_corners=True)
    out = F.grid_sample(img, grid, mode, padding_mode = padding_mode, align_corners=True)
    return out

class CIFAR10(data.Dataset):
    def __init__(self, root, train = True, aug_transform = None, norm_transform = None):
        self.aug_transform = aug_transform
        self.norm_transform = norm_transform
        self.cifar10 = torchvision.datasets.CIFAR10(root, train = train, download = False)

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        img, target = self.cifar10[index]
        ori_img = self.norm_transform(img)
        img = self.norm_transform(self.aug_transform(img))
        return img, ori_img, target

class TwoCropDataLoader(data.Dataset):
    def __init__(self, root, img_transform, img_color_transform1,  img_color_transform2):
        self.img_transform = img_transform
        self.img_color_transform1 = img_color_transform1
        self.img_color_transform2 = img_color_transform2
        # self.img_list = glob.glob(root + '*.jpg') + glob.glob(root + '*.png')
        self.img_list = glob.glob(f'{root}/*.jpg') + glob.glob(f'{root}/*.png')
        self.img_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        img = self.img_transform(img)
        img1 = self.img_color_transform1(img)
        img2 = self.img_color_transform2(img)
        return index, img1, img2, index


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
