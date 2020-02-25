#!/usr/bin/env python3

"""
Compute flow for directory with images.
"""

import cv2
import numpy as np
import pickle
import re

from argparse import ArgumentParser
from flowiz import convert_from_flow
from itertools import tee
from os import mkdir, listdir
from os.path import realpath, dirname, join, exists, basename
from torch import tensor
from typing import Optional

from run import estimate, Network


def center_crop(img):
    h, w = img.shape[:2]

    nh = h - h % 32
    nw = w - w % 32

    mar_h = (h - nh) // 2
    mar_w = (w - nw) // 2

    return img[mar_h: mar_h + nh, mar_w: mar_w + nw]


def get_args():
    parser = ArgumentParser()
    parser.add_argument("frames", help="Path to directory containing images.")
    parser.add_argument("-s", "--save-path", help="Directory where flow files will be saved.")
    parser.add_argument("-d", "--dataset", help="Whole cascade of flows is saved in file instead of last flow.",
                        action='store_true')
    return parser.parse_args()


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    list.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def load_directory(dir_path: str):
    """
    Load images from given directory.
    :param dir_path: path to directory containing images
    """

    paths = [join(dir_path, file) for file in sorted(listdir(dir_path), key=natural_keys) if file.endswith(("jpg", "png", "bmp"))]
    frames = [cv2.imread(path, cv2.IMREAD_COLOR) for path in paths]
    return paths, np.stack(frames)


def main():
    args = get_args()
    if args.save_path is not None:
        if not exists(args.save_path):
            mkdir(args.save_path)

    # frames = get_every_nth_frame(args.path, args.dist)[:args.count]
    paths, frames = load_directory(args.frames)
    args.count = len(frames)

    models = ["default"]
    # models = ["default", "kitti", "sintel"]

    model_name = "default"
    arguments_strModel = f"/home/xbakom01/src/pytorch-liteflownet/models/network-{model_name}.pytorch"

    model = Network(arguments_strModel, 5).cuda().eval()
    # from custom.model.liteflownet import LiteFlowNet
    # model = LiteFlowNet().cuda().eval()

    # h, w, c
    for n, ((path1, path2), (frame1, frame2)) in enumerate(zip(pairwise(paths), pairwise(frames))):
        # cv2.imwrite(join(realpath(dirname(__file__) + "/../"), "data", "frames", f"f{n}.png"), frame1)

        # security through obscurity
        frame1_t = tensor(cv2.cvtColor(center_crop(frame1), cv2.COLOR_BGR2RGB).transpose((2, 0, 1)).astype(np.float32) / 255).cuda()
        frame2_t = tensor(cv2.cvtColor(center_crop(frame2), cv2.COLOR_BGR2RGB).transpose((2, 0, 1)).astype(np.float32) / 255).cuda()

        flows = estimate(model, frame1_t, frame2_t)

        # flow = model(frame1_t.unsqueeze(0), frame2_t.unsqueeze(0))[-1][0].cpu()

        print(f"{n + 1}/{len(frames) - 1}", flush=True, end="\r")

        if not args.dataset:
            flow = flows[-1].cpu()

            if args.save_path is None:
                # display flow
                img = convert_from_flow(flow.numpy().transpose((1, 2, 0)).astype(np.float16))
                cv2.imshow("", img[:, :, ::-1])
                cv2.waitKey()
            else:
                # save in .flo format
                f1, f2 = list(map(lambda x: basename(x).split(".")[0], [path1, path2]))
                with open(join(args.save_path, f"{f1}-{f2}.flowc"), "wb") as f:
                    np.array([80, 73, 69, 72], np.uint8).tofile(f)
                    np.array([flow.size(2), flow.size(1)], np.int32).tofile(f)
                    np.array(flow.numpy().transpose(1, 2, 0), np.float32).tofile(f)

                # if n == len(frames) - 2:
                #     cv2.imwrite(join(realpath(dirname(__file__)), "data", "frames", f"f{args.count - 1}.png"), frame2)
        else:
            # save cascade of flows
            flow = [flow.cpu().numpy().astype(np.float16) for flow in flows]

            save_dir = args.save_path if args.save_path is not None else args.frames

            f1, f2 = list(map(lambda x: basename(x).split(".")[0], [path1, path2]))
            with open(join(args.save_path, f"{f1}-{f2}.flowc"), "wb") as f:
                pickle.dump(flow, f)


if __name__ == '__main__':
    main()
