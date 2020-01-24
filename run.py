#!/usr/bin/env python

import torch

import math
import numpy as np
import PIL.Image
import sys

from argparse import ArgumentParser
from os.path import isabs

from src.warp import Warp
from src.features import Features
from src.matching import Matching
from src.regularization import Regularization
from src.subpixel import Subpixel

assert(int(str('').join(torch.__version__.split('.')[0:3]).split('+')[0]) >= 41) # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance


class Network(torch.nn.Module):
	def __init__(self, model_path):
		super(Network, self).__init__()

		warp = Warp()

		self.moduleFeatures = Features()
		self.moduleMatching = torch.nn.ModuleList([Matching(intLevel, warp) for intLevel in [2, 3, 4, 5, 6]])
		self.moduleSubpixel = torch.nn.ModuleList([Subpixel(intLevel, warp) for intLevel in [2, 3, 4, 5, 6]])
		self.moduleRegularization = torch.nn.ModuleList([Regularization(intLevel, warp) for intLevel in [2, 3, 4, 5, 6]])

		if isabs(model_path):
			self.load_state_dict(torch.load(model_path))
		else:
			self.load_state_dict(torch.load('./network-' + model_path + '.pytorch'))

		self.register_buffer("t1_norm", torch.tensor([0.411618, 0.434631, 0.454253]).view((1, 3, 1, 1)))
		self.register_buffer("t2_norm", torch.tensor([0.410782, 0.433645, 0.452793]).view((1, 3, 1, 1)))

	def forward(self, tensorFirst, tensorSecond):
		tensorFirst -= self.t1_norm
		tensorSecond -= self.t2_norm
		features1 = self.moduleFeatures(tensorFirst)
		features2 = self.moduleFeatures(tensorSecond)

		tensorFirst = [tensorFirst]
		tensorSecond = [tensorSecond]

		# progressively downsample image up-down
		for level in [1, 2, 3, 4, 5]:
			tensorFirst.append(torch.nn.functional.interpolate(input=tensorFirst[-1], size=features1[level].size()[2:], mode='bilinear', align_corners=False))
			tensorSecond.append(torch.nn.functional.interpolate(input=tensorSecond[-1], size=features2[level].size()[2:], mode='bilinear', align_corners=False))

		flow_tensor: torch.tensor = None

		# extract and enhance flow from features down-up
		for level in [-1, -2, -3, -4, -5]:
			flow_tensor = self.moduleMatching[level](tensorFirst[level], tensorSecond[level], features1[level], features2[level], flow_tensor)
			flow_tensor = self.moduleSubpixel[level](tensorFirst[level], tensorSecond[level], features1[level], features2[level], flow_tensor)
			flow_tensor = self.moduleRegularization[level](tensorFirst[level], tensorSecond[level], features1[level], features2[level], flow_tensor)

		return flow_tensor * 20.0


def estimate(net, tensorFirst, tensorSecond):
	assert(tensorFirst.size(1) == tensorSecond.size(1))
	assert(tensorFirst.size(2) == tensorSecond.size(2))

	width = tensorFirst.size(2)
	height = tensorFirst.size(1)

	# assert(width == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	# assert(height == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, height, width)
	tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, height, width)

	rounded_width = int(math.floor(math.ceil(width / 32.0) * 32.0))
	rounded_height = int(math.floor(math.ceil(height / 32.0) * 32.0))

	tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(rounded_height, rounded_width), mode='bilinear', align_corners=False)
	tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(rounded_height, rounded_width), mode='bilinear', align_corners=False)

	flow_tensor = torch.nn.functional.interpolate(input=net(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(height, width), mode='bilinear', align_corners=False)

	flow_tensor[:, 0, :, :] *= float(width) / float(rounded_width)
	flow_tensor[:, 1, :, :] *= float(height) / float(rounded_height)

	return flow_tensor[0, :, :, :].cpu()


def get_args():
	parser = ArgumentParser()

	parser.add_argument("-m", "--model", required=True)
	parser.add_argument("-f", "--first", required=True)
	parser.add_argument("-s", "--second", required=True)
	parser.add_argument("-o", "--out", required=True)

	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()

	tensorFirst = torch.FloatTensor(np.array(PIL.Image.open(args.first))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
	tensorSecond = torch.FloatTensor(np.array(PIL.Image.open(args.second))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)

	model = Network(args.model).cuda().eval()
	tensorOutput = estimate(model, tensorFirst, tensorSecond)

	with open(args.out, 'wb') as f:
		# save in .flo format
		np.array([80, 73, 69, 72], np.uint8).tofile(f)
		np.array([tensorOutput.size(2), tensorOutput.size(1)], np.int32).tofile(f)
		np.array(tensorOutput.numpy().transpose(1, 2, 0), np.float32).tofile(f)

		import pickle
		# pickle.dump(tensorOutput.numpy().transpose(1, 2, 0).astype(np.float32), f)
