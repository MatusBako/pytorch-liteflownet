from torch import Tensor, cat
from torch.nn import Module, Conv2d, Sequential, LeakyReLU


class SubpixelModule(Module):
    def __init__(self, level, warp):
        super().__init__()

        self.warp = warp
        self.warp_weight: float = [10.0, 5.0, 2.5, 1.25, 0.625][level]

        self.processing = Sequential(
            Conv2d(in_channels=[66, 130, 194, 258, 386][level], out_channels=64,
                   kernel_size=3, stride=1, padding=1),
            LeakyReLU(inplace=False, negative_slope=0.1),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            LeakyReLU(inplace=False, negative_slope=0.1),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            LeakyReLU(inplace=False, negative_slope=0.1),
            Conv2d(in_channels=32, out_channels=2, kernel_size=[7, 5, 5, 3, 3][level], stride=1,
                   padding=[3, 2, 2, 1, 1][level])
        )

    def forward(self, features1: Tensor, features2: Tensor, flow: Tensor):
        features2 = self.warp(image=features2, flow=flow * self.warp_weight)

        module_input = cat([features1, features2, flow], 1)
        return (flow if flow is not None else 0.) + self.processing(module_input)
