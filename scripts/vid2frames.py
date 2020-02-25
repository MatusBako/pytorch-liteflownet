#!/usr/bin/env python3

import cv2
import numpy as np
import pickle

from argparse import ArgumentParser
from flowiz import convert_from_flow
from itertools import tee
from os import mkdir
from os.path import realpath, dirname, join, exists
from torch import tensor
from typing import Optional

from run import estimate, Network


def get_args():
    parser = ArgumentParser()
    parser.add_argument("path", help="Path to video.", default="/home/xbakom01/src/pytorch-liteflownet/data/video.3gp")#realpath(dirname(__file__)+"../data/video.gp3"))
    parser.add_argument("-c", "--count", type=int, help="Number of frames to be extracted.", default=None)
    parser.add_argument("-d", "--dist", type=int, help="Number of frames between every two extracted frames <0, inf).",
                        default=1)
    parser.add_argument("-o", "--output", help="Path to output directory.")
    return parser.parse_args()


def get_every_nth_frame(video_path, stride) -> np.ndarray:
    cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Error opening video stream or file")

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if frame_idx % stride == 0:
                h, w = frame.shape[:2]
                center = (w/2, h/2)
                mat = cv2.getRotationMatrix2D(center, 180, 1)
                frame = cv2.warpAffine(frame, mat, (w, h))
                frames.append(frame)

        else:
            break
        frame_idx += 1

    cap.release()
    return np.stack(frames)


def main():
    args = get_args()
    frames = get_every_nth_frame(args.path, args.dist)[:args.count]

    if not exists(args.output):
        mkdir(args.output)

    # h, w, c
    for n, frame1 in enumerate(frames):
        cv2.imwrite(join(args.output, f"frame_{n}.png"), frame1)


if __name__ == '__main__':
    main()
