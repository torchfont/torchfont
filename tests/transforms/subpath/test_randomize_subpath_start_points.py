import torch

from torchfont.transforms import randomize_subpath_start_points


def test_randomize_subpath_start_points_is_reproducible(
    square: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = square
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)

    out1 = randomize_subpath_start_points(types, coords, generator=g1)
    out2 = randomize_subpath_start_points(types, coords, generator=g2)

    assert torch.equal(out1[0], out2[0])
    assert torch.equal(out1[1], out2[1])


def test_randomize_subpath_start_points_changes_start_endpoint(
    square: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = square

    _out_types, out_coords = randomize_subpath_start_points(
        types,
        coords,
        generator=torch.Generator().manual_seed(0),
    )

    assert out_coords[0, 4:6].tolist() == [2.0, 1.0]


def test_randomize_subpath_start_points_leaves_open_subpaths_unchanged(
    open_subpath: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = open_subpath

    out_types, out_coords = randomize_subpath_start_points(
        types,
        coords,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(out_types, types)
    assert torch.equal(out_coords, coords)
