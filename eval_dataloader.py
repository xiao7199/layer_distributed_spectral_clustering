import os, pdb
import random
from os.path import join

import numpy as np
import torch.multiprocessing
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from torchvision import transforms

def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def create_cityscapes_colormap():
    colors = [(128, 64, 128),
              (244, 35, 232),
              (250, 170, 160),
              (230, 150, 140),
              (70, 70, 70),
              (102, 102, 156),
              (190, 153, 153),
              (180, 165, 180),
              (150, 100, 100),
              (150, 120, 90),
              (153, 153, 153),
              (153, 153, 153),
              (250, 170, 30),
              (220, 220, 0),
              (107, 142, 35),
              (152, 251, 152),
              (70, 130, 180),
              (220, 20, 60),
              (255, 0, 0),
              (0, 0, 142),
              (0, 0, 70),
              (0, 60, 100),
              (0, 0, 90),
              (0, 0, 110),
              (0, 80, 100),
              (0, 0, 230),
              (119, 11, 32),
              (0, 0, 0)]
    return np.array(colors)


class DirectoryDataset(Dataset):
    def __init__(self, root, path, image_set, transform, target_transform):
        super(DirectoryDataset, self).__init__()
        self.split = image_set
        self.dir = join(root, path)
        self.img_dir = join(self.dir, "imgs", self.split)
        self.label_dir = join(self.dir, "labels", self.split)

        self.transform = transform
        self.target_transform = target_transform

        self.img_files = np.array(sorted(os.listdir(self.img_dir)))
        assert len(self.img_files) > 0
        if os.path.exists(join(self.dir, "labels")):
            self.label_files = np.array(sorted(os.listdir(self.label_dir)))
            assert len(self.img_files) == len(self.label_files)
        else:
            self.label_files = None

    def __getitem__(self, index):
        image_fn = self.img_files[index]
        img = Image.open(join(self.img_dir, image_fn))

        if self.label_files is not None:
            label_fn = self.label_files[index]
            label = Image.open(join(self.label_dir, label_fn))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        if self.label_files is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.target_transform(label)
        else:
            label = torch.zeros(img.shape[1], img.shape[2], dtype=torch.int64) - 1

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.img_files)


class Potsdam(Dataset):
    def __init__(self, root, image_set, transform, target_transform, coarse_labels):
        super(Potsdam, self).__init__()
        self.split = image_set
        self.root = os.path.join(root, "potsdam")
        self.transform = transform
        self.target_transform = target_transform
        split_files = {
            "train": ["labelled_train.txt"],
            "unlabelled_train": ["unlabelled_train.txt"],
            # "train": ["unlabelled_train.txt"],
            "val": ["labelled_test.txt"],
            "train+val": ["labelled_train.txt", "labelled_test.txt"],
            "all": ["all.txt"]
        }
        assert self.split in split_files.keys()

        self.files = []
        for split_file in split_files[self.split]:
            with open(join(self.root, split_file), "r") as f:
                self.files.extend(fn.rstrip() for fn in f.readlines())

        self.coarse_labels = coarse_labels
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               255: -1
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id + ".mat"))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id + ".mat"))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.target_transform(label).squeeze(0)
        if self.coarse_labels:
            new_label_map = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.files)


class PotsdamRaw(Dataset):
    def __init__(self, root, image_set, transform, target_transform, coarse_labels):
        super(PotsdamRaw, self).__init__()
        self.split = image_set
        self.root = os.path.join(root, "potsdamraw", "processed")
        self.transform = transform
        self.target_transform = target_transform
        self.files = []
        for im_num in range(38):
            for i_h in range(15):
                for i_w in range(15):
                    self.files.append("{}_{}_{}.mat".format(im_num, i_h, i_w))

        self.coarse_labels = coarse_labels
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               255: -1
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.target_transform(label).squeeze(0)
        if self.coarse_labels:
            new_label_map = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.files)


class Coco(Dataset):
    def __init__(self, root, image_set, transform, target_transform,
                 coarse_labels, exclude_things, subset=None):
        super(Coco, self).__init__()
        self.split = image_set
        self.root = root
        self.coarse_labels = coarse_labels
        self.transform = transform
        self.target_transform = target_transform
        self.subset = subset
        self.exclude_things = exclude_things

        if self.subset is None:
            self.image_list = "Coco164kFull_Stuff_Coarse.txt"
        elif self.subset == 6:  # IIC Coarse
            self.image_list = "Coco164kFew_Stuff_6.txt"
        elif self.subset == 7:  # IIC Fine
            self.image_list = "Coco164kFull_Stuff_Coarse_7.txt"

        assert self.split in ["train", "val"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
        }
        self.image_path_prefix = self.root + '/'  + split_dirs[self.split][0]
        self.label_path_prefix = self.root + '/cocostuff/'  + split_dirs[self.split][0]
        self.image_files = []
        self.label_files = []
        for split_dir in split_dirs[self.split]:
            with open(join(self.root, 'CocoStuff164k', "curated", split_dir, self.image_list), "r") as f:
                img_ids = [fn.rstrip() for fn in f.readlines()]
                for img_id in img_ids:
                    self.image_files.append(img_id)
                    self.label_files.append(img_id)
        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

        self._label_names = [
            "ground-stuff",
            "plant-stuff",
            "sky-stuff",
        ]
        self.cocostuff3_coarse_classes = [23, 22, 21]
        self.first_stuff_index = 12

    def get_label(self, index):
        label_path = self.label_files[index]
        label = torch.from_numpy(np.array(Image.open(self.label_path_prefix + '/'+ label_path + '.png'))).long()
        label[label == 255] = -1  # to be consistent with 10k
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1
        coarse_label = coarse_label[None]
        if self.coarse_labels:
            coarser_labels = -torch.ones_like(label)
            for i, c in enumerate(self.cocostuff3_coarse_classes):
                coarser_labels[coarse_label == c] = i
            return coarser_labels, coarser_labels >= 0
        else:
            if self.exclude_things:
                return index, coarse_label - self.first_stuff_index, (coarse_label >= self.first_stuff_index)
            else:
                return index, coarse_label, coarse_label >= 0

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(self.image_path_prefix + '/' + image_path + '.jpg').convert("RGB"))

        random.seed(seed)
        torch.manual_seed(seed)
        label = torch.from_numpy(np.array(Image.open(self.label_path_prefix + '/'+ label_path + '.png'))).long()
        label[label == 255] = -1  # to be consistent with 10k
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1
        coarse_label = self.target_transform(coarse_label[None])
        if self.coarse_labels:
            coarser_labels = -torch.ones_like(label)
            for i, c in enumerate(self.cocostuff3_coarse_classes):
                coarser_labels[coarse_label == c] = i
            return img, coarser_labels, coarser_labels >= 0
        else:
            if self.exclude_things:
                return index, img, coarse_label - self.first_stuff_index, (coarse_label >= self.first_stuff_index)
            else:
                return index, img, coarse_label, coarse_label >= 0

    def __len__(self):
        #return len(self.image_files)
        return 100

class CityscapesSeg(Dataset):
    def __init__(self, root, image_set, transform, target_transform):
        super(CityscapesSeg, self).__init__()
        self.split = image_set
        self.root = join(root, "cityscapes")
        if image_set == "train":
            # our_image_set = "train_extra"
            # mode = "coarse"
            our_image_set = "train"
            mode = "fine"
        else:
            our_image_set = image_set
            mode = "fine"
        self.inner_loader = Cityscapes(self.root, our_image_set,
                                       mode=mode,
                                       target_type="semantic",
                                       transform=None,
                                       target_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        self.first_nonvoid = 7

    def __getitem__(self, index):
        if self.transform is not None:
            image, target = self.inner_loader[index]

            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            target = torch.from_numpy(np.array(self.target_transform(target))) * 1.0
            target = target - self.first_nonvoid
            target[target < 0] = -1
            mask = target == -1
            return index, image, target[None,...], mask
        else:
            return self.inner_loader[index]

    def __len__(self):
        return len(self.inner_loader)

def get_dataset(dataset_name, data_root, image_transform, target_transform, image_set = 'val'):
    crop_type = None
    if dataset_name == "cityscapes" and crop_type is None:
        n_classes = 27
        dataset_class = CityscapesSeg
        extra_args = dict()
    elif dataset_name == "cityscapes" and crop_type is not None:
        n_classes = 27
        dataset_class = CroppedDataset
        extra_args = dict(dataset_name="cityscapes", crop_type=crop_type, crop_ratio=cfg.crop_ratio)
    elif dataset_name == "cocostuff3":
        n_classes = 3
        dataset_class = Coco
        extra_args = dict(coarse_labels=True, subset=6, exclude_things=True)
    elif dataset_name == "cocostuff15":
        n_classes = 15
        dataset_class = Coco
        extra_args = dict(coarse_labels=False, subset=7, exclude_things=True)
    elif dataset_name == "cocostuff27":
        n_classes = 27
        dataset_class = Coco
        extra_args = dict(coarse_labels=False, subset=None, exclude_things=False)
        if image_set == "val":
            extra_args["subset"] = 7
    dataset = dataset_class(
        root=data_root,
        image_set=image_set,
        transform=image_transform,
        target_transform=target_transform, **extra_args)
    dataset.n_classes = n_classes
    return dataset
#if __name__ == '__main__':
#data_root = '/net/projects/willettlab/roxie62/dataset/coco2017'
#image_transform = transforms.Resize((256, 256))
#target_transform = transforms.Resize((256, 256))
#dataset = get_dataset('cocostuff27', data_root, image_transform, target_transform)
#pdb.set_trace()
