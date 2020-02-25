#!/usr/bin/env python3

"""
Compare original image with image warped using optical flow. (cycle using arrows)
"""

import cv2
import numpy as np
import re

from argparse import ArgumentParser
from flowiz import read_flow
from os import listdir
from os.path import dirname, realpath, join, isfile, isdir


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    list.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input", help="Path to image or directory with images.")
    parser.add_argument("flow", help="Path to image or directory with images.")
    args = parser.parse_args()

    assert isfile(args.input) and isfile(args.flow) or isdir(args.input) and isdir(args.flow), "Given path does not point to file or directory!"
    return args


def main():
    # TODO: args
    flows_dir_name = "data/flows_default"

    idx = 0
    flows = sorted(listdir(join(realpath(dirname(__file__) + "/../"), flows_dir_name)), key=natural_keys)
    frames = sorted(listdir(join(realpath(dirname(__file__) + "/../"), "data", "frames")), key=natural_keys)

    while True:
        flow = read_flow(join(realpath(dirname(__file__) + "/../"), flows_dir_name, flows[idx]))

        i1 = cv2.imread(join(realpath(dirname(__file__) + "/../"), "data", "frames", frames[idx]))
        i2 = cv2.imread(join(realpath(dirname(__file__) + "/../"), "data", "frames", frames[idx + 1]))

        i_res = warp_flow(i1, flow)

        cv2.imshow("Reconstructed", i_res)
        cv2.imshow("Original", i2)

        wk = cv2.waitKey()
        if wk == ord('q'):
            break
        elif wk == 81 and idx > 0:  # left arrow
            idx -= 1
        elif wk == 83 and idx < len(flows) - 2:  # right arrow
            idx += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
