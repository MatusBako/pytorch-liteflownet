import cv2
import re
from torch import full, stack
import torch.utils.data as data

from copy import deepcopy
from io import BytesIO
from flowiz import convert_from_flow
from functools import partial
import numpy as np
from operator import add
from os import listdir
from os.path import join, dirname, basename, isdir
import pickle
from PIL import Image
from scipy.ndimage import rotate
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


def build_transform(rng):
    return Compose([
        # RandomCrop((h, w), rng=rng),
        ToTensor(),
    ])

from functools import lru_cache

@lru_cache(maxsize=4096)
def load_image(file_name):
    img = np.array(open_image(file_name))
    return img


class DatasetDtd(data.Dataset):
    def __init__(self, dataset_dir, levels=6, input_transform=None, length=None, input_size=64, shift_range=6):

        super().__init__()
        subdirectories = [join(dataset_dir, subdir) for subdir in sorted(listdir(dataset_dir)) if
                          isdir(join(dataset_dir, subdir))]

        self.input_size = input_size
        self.images: List[str] = []

        for subdir in subdirectories:
            self.images.extend([join(subdir, f) for f in sorted(listdir(subdir), key=natural_keys)
                                if f.endswith((".jpg", ".bmp", ".png"))])

        self.shift_range = shift_range
        self.images = list(filter(lambda x: (np.array(Image.open(x).size[:2]) >= input_size + shift_range * 2).all(), self.images))
        # filtered = []
        #
        # for i in range(len(self.images)):
        #     size = Image.open(self.images[i]).size[:2]
        #
        #     if i == 2470:
        #         a = 0
        #         pass
        #
        #     if size[0] >= input_size + shift_range * 2 and size[1] >= input_size + shift_range * 2:
        #         filtered.append(self.images[i])
        #     else:
        #         pass
        #
        # self.images = filtered

        self.length = length if length and length <= len(self.images) else len(self.images)

        seed = np.random.randint(0, 2 ** 32 - 1)
        self.levels = levels

        if input_transform:
            self.input_transform = input_transform
        else:
            self.input_transform = build_transform(np.random.RandomState(seed))

    def random_crop(self, img):
        x = np.random.randint(0, img.shape[1] - self.input_size)
        y = np.random.randint(0, img.shape[0] - self.input_size)
        img = img[y:y + self.input_size, x:x + self.input_size]

        return img

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
        angle = np.random.rand() * 2 * np.pi

        i1 = rotate(i1, np.degrees(angle), axes=(0, 1))
        i2 = rotate(i2, np.degrees(angle), axes=(0, 1))

        # TODO: possibility for random rotation that we crop area without original picture inside
        #  possible workaround -> generate only small rotations, but how small?
        for idx, flow in enumerate(flows):
            # rotate each point in optical flow (vector for each point)
            flow = np.apply_along_axis(DatasetDtd.rotate_via_numpy, 0, flow, angle)
            # rotate optical flow as a whole
            flows[idx] = rotate(flow, np.degrees(angle), axes=(1, 2))

        return i1, i2, flows

    @staticmethod
    def perlin_noise(max_distance=6, size=(64, 64), levels=4, smoothness=0.1, tiles=(1, 1)):
        noise = np.random.normal(size=(tiles[0], tiles[1], 2)).astype(np.float32)
        for l in range(levels):
            noise = cv2.resize(noise * smoothness, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            noise += np.random.normal(size=noise.shape).astype(np.float32)

        max_magnitude = (noise * noise).sum(axis=2).max() ** 0.5
        noise = noise / max_magnitude * max_distance

        noise = cv2.resize(noise, (size[0] * tiles[0], size[1] * tiles[1]), interpolation=cv2.INTER_LINEAR)
        return noise

    def _debug_display(self, i1, i2, flows, winpref="", wait=True):
        cv2.imshow(winpref + "1", cv2.cvtColor(i1, cv2.COLOR_GRAY2BGR))
        cv2.imshow(winpref + "2", cv2.cvtColor(i2, cv2.COLOR_GRAY2BGR))
        cv2.imshow(winpref + "f", cv2.cvtColor(convert_from_flow(flows[-1].transpose((1, 2, 0))), cv2.COLOR_RGB2BGR))
        if wait:
            cv2.waitKey()

    def __getitem__(self, index):
        img = load_image(self.images[index])
        # self._debug_display(i1, i2, flow, winpref="full")

        # TODO: skip rotation for now
        # i1, i2, flow = self.random_rotate(i1, i2, flow)
        w, h = self.input_size, self.input_size

        # img = self.random_crop(img)

        x1 = np.random.randint(self.shift_range + 1, img.shape[1] - w - 2 - self.shift_range)
        y1 = np.random.randint(self.shift_range + 1, img.shape[0] - h - 2 - self.shift_range)
        # x2 = x1 + np.random.randint(-shift_range, +shift_range)
        # y2 = y1 + np.random.randint(-shift_range, +shift_range)

        i1 = img[y1:y1+h, x1:x1+w]
        # i2 = img[y2:y2+h, x2:x2+w]

        flows = []

        # flow_y = y1 - y2
        # flow_x = x1 - x2
        #
        #
        # # from smallest to largest => flip list after making pyramid
        # for i in range(self.levels):
        #     # TODO: unsqueeze??
        #     flows.append(
        #         stack([full((h, w), flow_x),
        #                full((h, w), flow_y)])
        #     )
        #     h //= 2
        #     w //= 2

        # TODO: consider cv2.convertMaps()
        flow = DatasetDtd.perlin_noise(np.random.uniform(7), (h, w), levels=4, smoothness=3)
        tmp = - flow.copy()
        tmp[:, :, 0] += np.arange(w)
        tmp[:, :, 1] += np.arange(h)[:, np.newaxis]
        i2 = cv2.remap(i1, tmp, None, cv2.INTER_LINEAR)

        for i in range(self.levels):
            # TODO: unsqueeze??
            flows.append(flow.transpose((2, 0, 1)))
            flow = cv2.resize(flow, None, fx=0.5, fy=0.5)

        # # from smallest to largest => flip list after making pyramid
        # for i in range(self.levels):
        #     # TODO: unsqueeze??
        #     flows.append(
        #         stack([full((h, w), flow_x),
        #                full((h, w), flow_y)])
        #     )
        #     h //= 2
        #     w //= 2

        mm = lambda x: (x - x.min()) / (x.max() - x.min())
        # cv2.imshow("",
        #            np.hstack(
        #                (i1,
        #                 i2,
        #                 (mm(np.stack((flow[:, :, 0], flow[:, :, 0], flow[:, :, 0]), 2)) * 256).astype(np.uint8),
        #                 (mm(np.stack((flow[:, :, 1], flow[:, :, 1], flow[:, :, 1]), 2)) * 256).astype(np.uint8)
        #                 )
        #            )
        # )
        # cv2.waitKey()
        # exit()
        # self._debug_display(i1, i2, flow)

        return tensor(i1.transpose((2, 0, 1))).float() / 255, \
            tensor(i2.transpose((2, 0, 1))).float() / 255, \
            flows[::-1]

    def __len__(self):
        return self.length
