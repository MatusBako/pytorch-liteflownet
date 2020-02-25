import re
import torch.utils.data as data

from copy import deepcopy
from io import BytesIO
from functools import partial
import numpy as np
from operator import add
from os import listdir
from os.path import join, dirname, basename, isdir
import pickle
from PIL import Image
from torch import tensor
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine
from torchvision.transforms.functional import crop
from typing import List, Tuple

from .transforms import RandomCrop

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    list.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def open_image(path):
    img = Image.open(path)
    ret = img.copy()
    img.close()
    return ret


def get_img_size(path):
    img = Image.open(path)
    return img.size


def build_transform(rng, h, w):
    return Compose([
        # RandomCrop((h, w), rng=rng),
        Resize((h, w), interpolation=Image.LINEAR),
        ToTensor(),
    ])


class Dataset(data.Dataset):
    def __init__(self, dataset_dir, input_transform=None, length=None):

        super().__init__()
        subdirectories = [join(dataset_dir, subdir) for subdir in sorted(listdir(dataset_dir)) if
                          isdir(join(dataset_dir, subdir))]

        self.images: List[str] = []
        self.flows: List[str] = []

        for subdir in subdirectories:
            self.images.extend([join(subdir, f) for f in sorted(listdir(subdir), key=natural_keys)
                                if f.endswith((".jpg", ".bmp", ".png"))])

            self.flows.extend([join(subdir, f) for f in sorted(listdir(subdir), key=natural_keys)
                               if f.endswith(".flowc")])

        self.length = length if length and length <= len(self.flows) else len(self.flows)
        self.w, self.h = get_img_size(self.images[0])

        seed = np.random.randint(0, 2 ** 32 - 1)

        if input_transform:
            self.input_transform = input_transform
        else:
            self.input_transform = build_transform(np.random.RandomState(seed), self.w, self.h)

    def random_crop(self, i1, i2, flows, crop_size):
        shape = i1.shape
        crop = [np.random.randint(0, shape[0] - crop_size), np.random.randint(0, shape[1] - crop_size)]

        i1 = i1[crop[0]:crop[0] + crop_size, crop[1]:crop[1] + crop_size]
        i2 = i2[crop[0]:crop[0] + crop_size, crop[1]:crop[1] + crop_size]

        for i in range(len(flows) - 1, -1, -1):
            flows[i] = flows[i][0, :, crop[0]:crop[0] + crop_size, crop[1]:crop[1] + crop_size]
            crop = [coord // 2 for coord in crop]
            crop_size //= 2

        return i1, i2, flows

    @staticmethod
    def unit_vector(vector):
        return vector / np.linalg.norm(vector, axis=0)

    @staticmethod
    def rotate_via_numpy(xy, radians):
        # TODO: might need to be optimized?
        x, y = xy
        c, s = np.cos(radians), np.sin(radians)
        j = np.array([[c, s], [-s, c]])
        m = np.dot(j, [x, y])

        return float(m.T[0]), float(m.T[1])

    def random_rotate(self, i1, i2, flows):
        from scipy.ndimage import rotate

        angle = np.pi #np.random.rand() * 2 * np.pi

        i1 = rotate(i1, np.degrees(angle), axes=(0, 1))
        i2 = rotate(i2, np.degrees(angle), axes=(0, 1))

        for idx, flow in enumerate(flows):
            # rotate each point in optical flow (vector for each point)
            flow = np.apply_along_axis(Dataset.rotate_via_numpy, 1, flow, angle)
            # rotate optical flow as a whole
            flows[idx] = rotate(flow, np.degrees(angle), axes=(0, 1))

        return i1, i2, flows

    def center_crop(self, img, bounding):
        start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
        end = tuple(map(add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]

    def __getitem__(self, index):
        flow_path = self.flows[index]

        with open(flow_path, "rb") as f:
            flow = pickle.load(f)

        base1, base2 = basename(flow_path).split(".")[0].split("-")

        # TODO: find suffix dynamically
        path1 = join(dirname(flow_path), base1 + ".png")
        path2 = join(dirname(flow_path), base2 + ".png")

        """
        RandomCrop parametrizovany -> pre oba obrazky
        """

        i1 = np.array(open_image(path1))
        i2 = np.array(open_image(path2))

        center_shape = [s - s % 32 for s in i1.shape]
        i1 = self.center_crop(i1, center_shape)
        i2 = self.center_crop(i2, center_shape)

        crop_size = 256

        # TODO: skip rotation for now
        # i1, i2, flow = self.random_rotate(i1, i2, flow)
        i1, i2, flow = self.random_crop(i1, i2, flow, crop_size)

        return tensor(i1).unsqueeze(0).float(), tensor(i2).unsqueeze(0).float(), [tensor(f) for f in flow]

    def __len__(self):
        return self.length
