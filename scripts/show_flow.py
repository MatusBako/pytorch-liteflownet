#!/usr/bin/env python3

"""
Display computed optical flows from given directory and go over them using arrows.
"""

import cv2
import re

from argparse import ArgumentParser
from os import listdir
from os.path import dirname, realpath, join, isfile, isdir

from flowiz import convert_from_file


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input", help="Path to image or directory with flow.", default="cv_flows")
    args = parser.parse_args()

    assert isfile(args.input) or isdir(args.input), "Given path does not point to file or directory!"
    return args


def show_flow(path: str):
    rgb = convert_from_file(path)

    filename = path.split('/')[-1]

    rgb = cv2.putText(rgb,
                      filename,
                      org=(50, 50),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=1,
                      color=(255, 255, 255))

    cv2.imshow("", rgb)
    return cv2.waitKey()


def main():
    args = parse_args()

    if isfile(args.input):
        show_flow(args.input)
    else:
        idx = 0
        files = [join(args.input, file) for file in sorted(listdir(args.input), key=natural_keys)]

        while True:
            file = files[idx]

            wk = show_flow(file)
            if wk == ord('q'):
                break
            elif wk == 81 and idx > 0:  # left arrow
                idx -= 1
            elif wk == 83 and idx < len(files) - 1:  # right arrow
                idx += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
