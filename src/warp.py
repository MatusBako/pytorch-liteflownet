import torch


class Warp:
    def __init__(self):
        self.backward_tensor_grid = {}

    def __call__(self, input_tensor, flow_tensor):
        if str(flow_tensor.size()) not in self.backward_tensor_grid:
            horizontal = torch.linspace(-1.0, 1.0, flow_tensor.size(3)) \
                .view(1, 1, 1, flow_tensor.size(3)) \
                .expand(flow_tensor.size(0), -1, flow_tensor.size(2), -1)

            vertical = torch.linspace(-1.0, 1.0, flow_tensor.size(2)) \
                .view(1, 1, flow_tensor.size(2), 1) \
                .expand(flow_tensor.size(0), -1, -1, flow_tensor.size(3))

            self.backward_tensor_grid[str(flow_tensor.size())] = torch.cat([horizontal, vertical], 1).cuda()

        flow_tensor = torch.cat(
            [flow_tensor[:, 0:1, :, :] / ((input_tensor.size(3) - 1.0) / 2.0),
             flow_tensor[:, 1:2, :, :] / ((input_tensor.size(2) - 1.0) / 2.0)],
            1)

        return torch.nn.functional.grid_sample(input=input_tensor, mode='bilinear', padding_mode='zeros',
                                               grid=(self.backward_tensor_grid[
                                                         str(flow_tensor.size())] + flow_tensor).permute(0, 2, 3, 1),
                                               align_corners=True)
