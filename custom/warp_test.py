import torch


def main():
    flow = torch.ones((1, 2, 5, 5))
    img = torch.Tensor(range(5)).view((1, 1, 1, 5)).expand((-1, -1, 5, -1))

    lin_x = torch.linspace(-1.0, 1.0, flow.size(3))
    lin_y = torch.linspace(-1.0, 1.0, flow.size(2))

    meshx, meshy = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((meshx, meshy), 2).unsqueeze(0)

    # flow ukazuje displacement v pixeloch, musis to premapovat na interval [-1, 1]
    flow /= torch.tensor([(img.size(3) - 1.0) / 2.0, (img.size(2) - 1.0) / 2.0]) \
        .view(1, 2, 1, 1)

    print(torch.nn.functional.grid_sample(input=img, mode='bilinear', padding_mode='zeros',
                                          grid=grid + flow.permute(0, 2, 3, 1), align_corners=True))


if __name__ == "__main__":
    main()
