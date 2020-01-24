#!/usr/bin/env python3

"""
Extract optical flow using Farneback algorithm from cv2 package.
"""

import cv2
import numpy as np
import pickle

from itertools import tee
from os.path import realpath, dirname, join
from typing import Optional


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
    count = 20
    frames = get_every_nth_frame("data/video.3gp", 3)[:count]

    # h, w, c
    for n, (frame1, frame2) in enumerate(pairwise(frames)):
        #cv2.imwrite(join(realpath(dirname(__file__)), "frames", f"frame_{n}.png"), frame1)

        # security through obscurity
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        with open(join(realpath(dirname(__file__)), "cv_flows", f"f{n}.flo"), "wb") as f:
            pickle.dump(flow.astype(np.float32), f)

        # if n == len(frames) - 1:
        #     cv2.imwrite(join(realpath(dirname(__file__)), "frames", f"frame_{count}.png"), frame2)


if __name__ == '__main__':
    main()
