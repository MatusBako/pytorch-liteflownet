import torch

from src import correlation


class Matching(torch.nn.Module):
    def __init__(self, level: int, warp):
        super(Matching, self).__init__()

        self.warp_weight = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][level]
        self.warp = warp

        if level != 2:
            self.moduleFeat = torch.nn.Sequential()
        else:
            self.moduleFeat = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        if level == 6:
            self.moduleUpflow = None
        else:
            self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2,
                                                         padding=1, bias=False, groups=2)

        if level >= 4:
            self.moduleUpcorr = None
        else:
            self.moduleUpcorr = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2,
                                                         padding=1, bias=False, groups=49)

        self.moduleMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][level], stride=1,
                            padding=[0, 0, 3, 2, 2, 1, 1][level]))

    def forward(self, tensor1, tensor2, features_tensor1, features_tensor2, flow_tensor):
        features_tensor1 = self.moduleFeat(features_tensor1)
        features_tensor2 = self.moduleFeat(features_tensor2)

        if flow_tensor is not None:
            flow_tensor = self.moduleUpflow(flow_tensor)
            features_tensor2 = self.warp(input_tensor=features_tensor2, flow_tensor=flow_tensor * self.warp_weight)

        if self.moduleUpcorr is None:
            correlation_tensor = torch.nn.functional.leaky_relu(
                input=correlation.FunctionCorrelation(tensorFirst=features_tensor1,
                                                      tensorSecond=features_tensor2, intStride=1),
                negative_slope=0.1, inplace=False)
        else:
            correlation_tensor = self.moduleUpcorr(torch.nn.functional.leaky_relu(
                input=correlation.FunctionCorrelation(tensorFirst=features_tensor1,
                                                      tensorSecond=features_tensor2, intStride=2),
                negative_slope=0.1, inplace=False))

        return (flow_tensor if flow_tensor is not None else 0.0) + self.moduleMain(correlation_tensor)
