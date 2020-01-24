import torch

Backward_tensorGrid = {}


def backward(input_tensor, flow_tensor):
    if str(flow_tensor.size()) not in Backward_tensorGrid:
        """
        nejakym sposobom vygenerujes mesh a ulozis si ho??
        """
        tensorHorizontal = torch.linspace(-1.0, 1.0, flow_tensor.size(3))\
            .view(1, 1, 1, flow_tensor.size(3))\
            .expand(flow_tensor.size(0), -1, flow_tensor.size(2), -1)

        tensorVertical = torch.linspace(-1.0, 1.0, flow_tensor.size(2))\
            .view(1, 1, flow_tensor.size(2), 1)\
            .expand(flow_tensor.size(0), -1, -1, flow_tensor.size(3))

        Backward_tensorGrid[str(flow_tensor.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()

    flow_tensor = torch.cat(
        [flow_tensor[:, 0:1, :, :] / ((input_tensor.size(3) - 1.0) / 2.0),
         flow_tensor[:, 1:2, :, :] / ((input_tensor.size(2) - 1.0) / 2.0)],
        1)

    return torch.nn.functional.grid_sample(
        input=input_tensor,
        grid=(Backward_tensorGrid[str(flow_tensor.size())] + flow_tensor).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros')
