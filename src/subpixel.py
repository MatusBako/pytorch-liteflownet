import torch

from src.backward import backward


class Subpixel(torch.nn.Module):
    def __init__(self, level: int):
        super(Subpixel, self).__init__()

        self.backward_weight: float = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][level]

        if level != 2:
            self.moduleFeat = torch.nn.Sequential()
        elif level == 2:
            self.moduleFeat = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.moduleMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=[0, 0, 130, 130, 194, 258, 386][level], out_channels=128,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][level],
                            stride=1, padding=[0, 0, 3, 2, 2, 1, 1][level]))

    def forward(self, tensor1, tensor2, features_tensor1, features_tensor2, flow_tensor):
        features_tensor1 = self.moduleFeat(features_tensor1)
        features_tensor2 = self.moduleFeat(features_tensor2)

        if flow_tensor is not None:
            features_tensor2 = backward(input_tensor=features_tensor2,
                                        flow_tensor=flow_tensor * self.backward_weight)

        return (flow_tensor if flow_tensor is not None else 0.) + \
            self.moduleMain(torch.cat([features_tensor1, features_tensor2, flow_tensor], 1))
