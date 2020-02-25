from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, LeakyReLU
from torch.nn.functional import interpolate

from .correlation import FunctionCorrelation
from .warp import Warp


class MatchingModule(Module):
    def __init__(self, level, warp):
        super().__init__()

        self.warp: Warp = warp
        self.warp_weight = [10.0, 5.0, 2.5, 1.25, 0.625][level]

        self.processing = Sequential(
            Conv2d(in_channels=49, out_channels=64, kernel_size=3, stride=1, padding=1),
            LeakyReLU(inplace=False, negative_slope=0.1),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            LeakyReLU(inplace=False, negative_slope=0.1),
            Conv2d(in_channels=32, out_channels=2, kernel_size=[7, 5, 5, 3, 3][level], stride=1,
                   padding=[3, 2, 2, 1, 1][level]))

    def forward(self, features1: Tensor, features2: Tensor, flow: Tensor):
        if flow is not None:
            # TODO: verify if size is correct
            flow = interpolate(flow, features1.shape[-2:], mode='bilinear')
            features2 = self.warp(image=features2, flow=flow * self.warp_weight)

        # TODO: OPT raise stride and upscale correlation tensor
        correlation_tensor = FunctionCorrelation(tensorFirst=features1, tensorSecond=features2, stride=1)

        return (flow if flow is not None else 0.) + self.processing(correlation_tensor)
