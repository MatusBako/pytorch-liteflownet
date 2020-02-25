from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn.functional import interpolate
from typing import Optional

from .features import FeaturesModule
from .warp import Warp
from .matching import MatchingModule
from .subpixel import SubpixelModule
from .regularization import RegularizationModule


class LiteFlowNet(Module):
    def __init__(self, level_cnt: int = 6):
        super().__init__()

        assert 1 <= level_cnt <= 6
        self.level_cnt = level_cnt

        self.warp = Warp()

        self.feature_extr = FeaturesModule(level_cnt)

        self.moduleMatching = ModuleList([MatchingModule(level, self.warp) for level in range(5)])
        self.moduleSubpixel = ModuleList([SubpixelModule(level, self.warp) for level in range(5)])
        # self.moduleRegularization = ModuleList(
        #     [RegularizationModule(level, self.warp) for level in range(5)])

        # TODO: create rest of the modules

    def forward(self, image1: Tensor, image2: Tensor, levels: Optional[int] = None):
        levels = self.level_cnt if levels is None else levels

        features1 = self.feature_extr(image1)
        features2 = self.feature_extr(image2)

        tensor1 = [image1]
        tensor2 = [image2]

        # progressive downsampling
        for level in range(1, levels):
            tensor1.append(interpolate(input=tensor1[-1],
                                       size=features1[level].size()[2:],
                                       mode='bilinear', align_corners=True))
            tensor2.append(interpolate(input=tensor2[-1],
                                       size=features2[level].size()[2:],
                                       mode='bilinear', align_corners=True))

        flow: Optional[Tensor] = None
        flows = []

        for level in range(levels - 1, 0, -1):
            flow = self.moduleMatching[level - 1](features1[level], features2[level], flow)
            flow = self.moduleSubpixel[level - 1](features1[level], features2[level], flow)
            # flow = self.moduleRegularization[level - 1](tensor1[level], tensor2[level],
            #                                             features1[level], features2[level], flow)

            flows.append(flow.clone())

        # TODO: substitute with another level of upsampling
        flows.append(interpolate(input=flows[-1],
                                 size=tensor1[0].shape[-2:], mode='bilinear',
                                 align_corners=True)) # or False?

        return flows
