from torch import Tensor
from torch.nn import Module, Conv2d, LeakyReLU, Sequential, ModuleList


class FeaturesModule(Module):
    def __init__(self, level_cnt):
        super().__init__()

        self.level_cnt = level_cnt
        self.module_list = ModuleList()

        self.module_list.append(
            Sequential(
                Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3),
                LeakyReLU(inplace=False, negative_slope=0.1)
            )
        )

        self.module_list.append(
            Sequential(
                Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1)
            )
        )

        self.module_list.append(
            Sequential(
                Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1)
            )
        )

        self.module_list.append(
            Sequential(
                Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1)
            )
        )

        self.module_list.append(
            Sequential(
                Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1)
            )
        )

        self.module_list.append(
            Sequential(
                Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1)
            )
        )

        self.module_list = self.module_list[:level_cnt]

    def forward(self, image: Tensor):
        result = []

        for i in range(self.level_cnt):
            # input is used only in first iteration
            result.append(self.module_list[i](result[-1] if result else image))

        return result
