#!/usr/bin/env python3

from torchsummary import summary
from custom.models.LiteFlowNet.liteflownet import LiteFlowNet


def main():
    model = LiteFlowNet(32, 3)


if __name__ == "__main__":
    main()
