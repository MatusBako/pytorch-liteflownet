from torch import cat
from torch.nn import Module, Sequential, LeakyReLU, Conv2d
from torch.nn.functional import unfold

from .warp import Warp


class RegularizationModule(Module):
    def __init__(self, level, warp):
        super().__init__()

        self.warp: Warp = warp
        self.warp_weight: float = [10.0, 5.0, 2.5, 1.25, 0.625][level]
        self.unfold: int = [7, 5, 5, 3, 3][level]

        self.processing = Sequential(
            Conv2d(in_channels=[131, 131, 131, 131, 195][level], out_channels=64, kernel_size=3, stride=1, padding=1),
            LeakyReLU(inplace=False, negative_slope=0.1),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            LeakyReLU(inplace=False, negative_slope=0.1),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            LeakyReLU(inplace=False, negative_slope=0.1),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleScaleX = Conv2d(in_channels=[49, 25, 25, 9, 9][level], out_channels=1,
                                   kernel_size=1, stride=1, padding=0)
        self.moduleScaleY = Conv2d(in_channels=[49, 25, 25, 9, 9][level], out_channels=1,
                                   kernel_size=1, stride=1, padding=0)

        self.moduleDist = Conv2d(in_channels=32, out_channels=[49, 25, 25, 9, 9][level],
                                 kernel_size=[7, 5, 5, 3, 3][level], stride=1,
                                 padding=[3, 2, 2, 1, 1][level])

        """
        TODO: vykasli sa na to co tu je
            sprav konvoluciu ktora ziska vahy zo vstupneho obrazku
        
        init:     
            w = torch.empty((c, h, w)).view((1, c, h, w))
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
            
        usage:
            output = F.conv2d(x, weights)
            
        """

    def forward(self, tensor1, tensor2, features_tensor1, features_tensor2, flow_tensor):
        warp = self.warp(image=tensor2, flow=flow_tensor * self.warp_weight)

        # second image - warped first image
        difference_tensor = (tensor1 - warp).pow(2.0).sum(1, True).sqrt().detach()

        dist_in = cat(
            [difference_tensor,
             # preco robi mean cez kazdy kanal?
             flow_tensor - flow_tensor.mean((2, 3), keepdim=True),
             features_tensor1
             ]
            , 1
        )

        # TODO: co to je ??? send help, pls

        dist_tensor = self.moduleDist(self.processing(dist_in))
        dist_tensor = dist_tensor.pow(2.0).neg()
        dist_tensor = (dist_tensor - dist_tensor.max(1, True)[0]).exp()

        # reciprocal(x): return 1 / x
        divisor_tensor = dist_tensor.sum(1, True).reciprocal()

        unfolded1 = unfold(input=flow_tensor[:, 0:1, :, :], kernel_size=self.unfold, stride=1,
                           padding=int((self.unfold - 1) / 2)).view_as(dist_tensor)
        unfolded2 = unfold(input=flow_tensor[:, 1:2, :, :], kernel_size=self.unfold, stride=1,
                           padding=int((self.unfold - 1) / 2)).view_as(dist_tensor)

        tensorScaleX = self.moduleScaleX(dist_tensor * unfolded1) * divisor_tensor
        tensorScaleY = self.moduleScaleY(dist_tensor * unfolded2) * divisor_tensor

        return cat([tensorScaleX, tensorScaleY], 1)
