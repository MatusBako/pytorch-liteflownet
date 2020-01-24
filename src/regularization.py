import torch

from src.backward import backward


class Regularization(torch.nn.Module):
    def __init__(self, level: int):
        super(Regularization, self).__init__()

        self.backward_weight: float = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][level]
        self.unfold: int = [0, 0, 7, 5, 5, 3, 3][level]

        if level >= 5:
            self.moduleFeat = torch.nn.Sequential()
        else:
            self.moduleFeat = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=[0, 0, 32, 64, 96, 128, 192][level], out_channels=128, kernel_size=1, stride=1, padding=0),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.moduleMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=[ 0, 0, 131, 131, 131, 131, 195 ][level], out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        if level >= 5:
            self.moduleDist = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][level],
                                kernel_size=[0, 0, 7, 5, 5, 3, 3][level], stride=1,
                                padding=[0, 0, 3, 2, 2, 1, 1][level]))
        else:
            self.moduleDist = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][level],
                                kernel_size=([0, 0, 7, 5, 5, 3, 3][level], 1), stride=1,
                                padding=([0, 0, 3, 2, 2, 1, 1][level], 0)),
                torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][level],
                                out_channels=[0, 0, 49, 25, 25, 9, 9][level],
                                kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][level]), stride=1,
                                padding=(0, [0, 0, 3, 2, 2, 1, 1][level])))

        self.moduleScaleX = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][level], out_channels=1,
                                            kernel_size=1, stride=1, padding=0)
        self.moduleScaleY = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][level], out_channels=1,
                                            kernel_size=1, stride=1, padding=0)

    def forward(self, tensor1, tensor2, features_tensor1, features_tensor2, flow_tensor):
        # occusion probability map = brightness error => L2_dist(i1, i2)
        warp = backward(input_tensor=tensor2, flow_tensor=flow_tensor * self.backward_weight)
        difference_tensor = (tensor1 - warp).pow(2.0).sum(1, True).sqrt().detach()

        dist_in = torch.cat([
            difference_tensor,
            # preco robi mean cez kazdy kanal?
            flow_tensor - flow_tensor.view(flow_tensor.size(0), 2, -1).mean(2, True).view(flow_tensor.size(0), 2, 1, 1),
            # lepsi zapis?
            # flow_tensor - flow_tensor.mean((2, 3), keepdim=True),
            self.moduleFeat(features_tensor1)]
        , 1)

        dist_tensor = self.moduleDist(self.moduleMain(dist_in))
        dist_tensor = dist_tensor.pow(2.0).neg()
        dist_tensor = (dist_tensor - dist_tensor.max(1, True)[0]).exp()

        # menovatel
        # reciprocal(x): return 1 / x
        divisor_tensor = dist_tensor.sum(1, True).reciprocal()

        # ?? boh vi
        unfolded1 = torch.nn.functional.unfold(input=flow_tensor[:, 0:1, :, :], kernel_size=self.unfold, stride=1,
                                               padding=int((self.unfold - 1) / 2)).view_as(dist_tensor)
        unfolded2 = torch.nn.functional.unfold(input=flow_tensor[:, 1:2, :, :], kernel_size=self.unfold, stride=1,
                                               padding=int((self.unfold - 1) / 2)).view_as(dist_tensor)

        tensorScaleX = self.moduleScaleX(dist_tensor * unfolded1) * divisor_tensor
        tensorScaleY = self.moduleScaleY(dist_tensor * unfolded2) * divisor_tensor

        return torch.cat([tensorScaleX, tensorScaleY], 1)