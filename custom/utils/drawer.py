import cv2
from flowiz import convert_from_flow
import numpy as np


class Drawer:
    def __init__(self, output_folder, img_count=4):
        self.train_cumloss = []
        self.test_cumloss = []
        self.psnr_list = []

        self.img_count = img_count

        self.output_folder = output_folder

    def save_images(self, input1: np.array, input2: np.array, result: np.array, target: np.array, label: str):
        img_count = result.data.shape[0]

        img_count = self.img_count if img_count >= self.img_count else img_count

        # TODO: this worked for grayscale images, rewrite to work for both formats
        # in_stack1 = np.repeat(np.hstack(input1[:img_count, :, :, :]), 3, 2)
        # in_stack2 = np.repeat(np.hstack(input2[:img_count, :, :, :]), 3, 2)

        in_stack1 = np.hstack(input1[:img_count, :, :, :])
        in_stack2 = np.hstack(input2[:img_count, :, :, :])
        re_stack = []
        tar_stack = []

        for i in range(img_count):
            re_stack.append(convert_from_flow(result[i]))
            tar_stack.append(convert_from_flow(target[i]))

        # stacks = (in_stack1, in_stack2, np.hstack(re_stack), np.hstack(tar_stack))
        stacks = (cv2.addWeighted(in_stack1, 0.9, in_stack2, 0.3, 0),
                  np.hstack(re_stack), np.hstack(tar_stack))

        collage = np.vstack(stacks)

        # convert to BGR because imwritew
        cv2.imwrite(f"{self.output_folder}/{label}.png", collage[:, :, ::-1])

