#!/usr/bin/env python3

import cv2
import numpy as np
import pickle

from argparse import ArgumentParser
from itertools import tee
from os import mkdir
from os.path import realpath, dirname, join, exists
from torch import tensor
from typing import Optional

from run import estimate, Network


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to video.", default=realpath(dirname(__file__)+"../data/video.gp3"))
    parser.add_argument("-c", "--count", type=int, help="Number of frames to be extracted.", default=20)
    parser.add_argument("-d", "--dist", type=int, help="Number of frames between every two extracted frames <0, inf).",
                        default=3)
    return parser.parse_args()


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


def main():
    args = get_args()
    frames = get_every_nth_frame(args.path, args.dist)[:args.count]

    models = ["default", "kitti", "sintel"]

    for model_name in models:
        arguments_strModel = f"/home/xbakom01/src/pytorch-liteflownet/models/network-{model_name}.pytorch"

        model = Network(arguments_strModel).cuda().eval()

        if not exists(join(realpath(dirname(__file__) + "/../"), f"data", "flows_{model_name}")):
            mkdir(join(realpath(dirname(__file__) + "/../"), f"data", "flows_{model_name}"))

        # h, w, c
        for n, (frame1, frame2) in enumerate(pairwise(frames)):
            cv2.imwrite(join(realpath(dirname(__file__) + "/../"), "data", "frames", f"f{n}.png"), frame1)

            # security through obscurity
            frame1_t = tensor(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)).astype(np.float32) / 255)
            frame2_t = tensor(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)).astype(np.float32) / 255)

            flow = estimate(model, frame1_t, frame2_t)

            # save in .flo format
            with open(join(realpath(dirname(__file__) + "/../"), f"data", f"flows_{model_name}", f"f{n}.flo"), "wb") as f:
                np.array([80, 73, 69, 72], np.uint8).tofile(f)
                np.array([flow.size(2), flow.size(1)], np.int32).tofile(f)
                np.array(flow.numpy().transpose(1, 2, 0), np.float32).tofile(f)

            if n == len(frames) - 2:
                cv2.imwrite(join(realpath(dirname(__file__)), "data", "frames", f"f{args.count - 1}.png"), frame2)


if __name__ == '__main__':
    main()
