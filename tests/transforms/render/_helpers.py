import torch


def _occupied_size(bitmap: torch.Tensor) -> tuple[int, int]:
    ys, xs = torch.nonzero(bitmap > 0, as_tuple=True)
    width = int(xs.max().item()) - int(xs.min().item()) + 1
    height = int(ys.max().item()) - int(ys.min().item()) + 1
    return width, height
