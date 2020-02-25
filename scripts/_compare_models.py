#!/usr/bin/env python3

"""
DEPRECATED

Compare performance of given models performing optical flow.
"""

import cv2
import numpy as np
import pickle

from itertools import tee
from os import mkdir
from os.path import realpath, dirname, join, exists
from torch import tensor
from typing import Optional

from run import estimate, Network


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_every_nth_frame(video_path, n) -> np.ndarray:
    cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Error opening video stream or file")

    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        frame_idx = 0

        if ret:
            if frame_idx % n == 0:
                frames.append(frame)

        # Break the loop
        else:
            break

    cap.release()
    return np.stack(frames)


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def main():
    count = 20
    frames = get_every_nth_frame("data/video.3gp", 3)[:count]

    models = ["default", "kitti", "sintel"]

    for model_name in models:
        arguments_strModel = join(realpath("../"), f"models/network-{model_name}.pytorch")

        # TODO: launch pycharm from terminal to load env. variables
        model = Network(arguments_strModel).cuda().eval()

        if not exists(join(realpath(dirname(__file__)), f"data/flows_{model_name}")):
            mkdir(join(realpath(dirname(__file__)), f"data/flows_{model_name}"))

        mse_arr = []

        # h, w, c
        for n, (frame1, frame2) in enumerate(pairwise(frames)):
            # security through obscurity
            input_frame1 = tensor(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)).astype(np.float32) / 255)
            input_frame2 = tensor(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)).astype(np.float32) / 255)

            flow = estimate(model, input_frame1, input_frame2)

            warped_image = warp_flow(frame1, flow.cpu().numpy().transpose((1, 2, 0)))

            # compute mse
            mse_arr.append(((warped_image - frame2) ** 2).mean())

        print(f"{model_name}: {np.array(mse_arr).mean():.3f}")


if __name__ == '__main__':
    main()
