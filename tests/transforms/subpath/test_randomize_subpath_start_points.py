import pytest
import torch

from torchfont.structures import Outline
from torchfont.transforms import RandomizeSubpathStartPoints


def test_randomize_subpath_start_points_is_reproducible(
    square: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = square
    torch.manual_seed(7)
    out1 = RandomizeSubpathStartPoints()(Outline(types, coords))
    torch.manual_seed(7)
    out2 = RandomizeSubpathStartPoints()(Outline(types, coords))

    assert torch.equal(out1.types, out2.types)
    assert torch.equal(out1.coords, out2.coords)


def test_randomize_subpath_start_points_changes_start_endpoint(
    square: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = square

    torch.manual_seed(0)
    out_coords = RandomizeSubpathStartPoints()(Outline(types, coords)).coords

    assert out_coords[0, 4:6].tolist() == [2.0, 1.0]


def test_randomize_subpath_start_points_leaves_open_subpaths_unchanged(
    open_subpath: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = open_subpath

    torch.manual_seed(0)
    output = RandomizeSubpathStartPoints()(Outline(types, coords))
    out_types, out_coords = output.types, output.coords

    assert torch.equal(out_types, types)
    assert torch.equal(out_coords, coords)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_randomize_subpath_start_points_accepts_cpu_generator_for_cuda_input(
    square: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = (tensor.cuda() for tensor in square)
    output = RandomizeSubpathStartPoints()(Outline(types, coords))
    out_types, out_coords = output.types, output.coords

    assert out_types.device.type == "cuda"
    assert out_coords.device.type == "cuda"
