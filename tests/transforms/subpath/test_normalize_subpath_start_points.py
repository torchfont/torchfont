import torch

from torchfont.io import ElementType
from torchfont.transforms import normalize_subpath_start_points


def test_normalize_subpath_start_points_uses_smallest_endpoint(
    square: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = square
    coords[0, 4:6] = torch.tensor([3.0, 3.0])

    _out_types, out_coords = normalize_subpath_start_points(types, coords)

    assert out_coords[0, 4:6].tolist() == [1.0, 2.0]


def test_normalize_subpath_start_points_materializes_old_close_edge(
    square: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = square
    coords[0, 4:6] = torch.tensor([3.0, 3.0])

    out_types, out_coords = normalize_subpath_start_points(types, coords)

    assert out_coords[0, 4:6].tolist() == [1.0, 2.0]
    assert out_types[1].item() == ElementType.LINE_TO.value
    assert out_coords[1, 4:6].tolist() == [3.0, 3.0]
    assert len(out_types) == len(types) + 1


def test_normalize_subpath_start_points_preserves_incoming_curve() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.CURVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0, 0, 0, 0, 1.0, 0.0],
            [0.7, 0.0, 0.3, 0.0, 0.0, 0.0],
            [0, 0, 0, 0, 1.0, 1.0],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = normalize_subpath_start_points(types, coords)

    assert out_types[-3].item() == ElementType.CURVE_TO.value
    assert torch.equal(out_coords[-3], coords[1])


def test_normalize_subpath_start_points_leaves_open_subpaths_unchanged(
    open_subpath: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = open_subpath

    out_types, out_coords = normalize_subpath_start_points(types, coords)

    assert torch.equal(out_types, types)
    assert torch.equal(out_coords, coords)


def test_normalize_subpath_start_points_skips_degenerate_close_edge(
    square: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = square
    coords[-3, 4:6] = coords[0, 4:6]

    out_types, _out_coords = normalize_subpath_start_points(types, coords)

    assert len(out_types) == len(types)
