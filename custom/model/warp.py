import torch

from typing import Dict


class Warp:
    def __init__(self):
        self.backward_tensor_grid: Dict[str, torch.Tensor] = {}

    def __call__(self, image: torch.Tensor, flow: torch.Tensor):
        batch_size = flow.size(0)

        if str(flow.size()) not in self.backward_tensor_grid:
            """
            Possible space optimization: don't store whole 4d tensors, instead store 3d
                and stack them on first dimension, is it worth it though?    
                
                flow: (n, 2, h, w)         
            """

            lin_x = torch.linspace(-1.0, 1.0, flow.size(3))
            lin_y = torch.linspace(-1.0, 1.0, flow.size(2))

            meshx, meshy = torch.meshgrid((lin_y, lin_x))
            grid = torch.stack((meshx, meshy), 2).expand(batch_size, -1, -1, -1)
            self.backward_tensor_grid[str(flow.shape)] = grid.cuda()

        grid = self.backward_tensor_grid[str(flow.shape)] + flow.permute(0, 2, 3, 1)

        # flow contains displacement in pixels, values need to be mapped to interval [-1, 1]
        flow /= torch.tensor([(image.size(3) - 1.0) / 2.0, (image.size(2) - 1.0) / 2.0]) \
            .view(1, 2, 1, 1).cuda()

        # grid ma hodnoty [-1, 1] normalizovane voci rozmeru obrazku
        return torch.nn.functional.grid_sample(input=image, mode='bilinear', padding_mode='zeros',
                                               grid=grid + flow.permute(0, 2, 3, 1), align_corners=True)
