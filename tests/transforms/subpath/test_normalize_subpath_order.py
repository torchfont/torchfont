import torch

from torchfont import ElementType, Outline
from torchfont.transforms import NormalizeSubpathOrder
from torchfont.transforms import functional as F  # noqa: N812


def _two_squares() -> Outline:
    types = torch.tensor(
        [
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.LINE_TO,
            ElementType.CLOSE,
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.LINE_TO,
            ElementType.CLOSE,
            ElementType.END,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0, 0, 0, 0, 2.0, 2.0],
            [0, 0, 0, 0, 3.0, 2.0],
            [0, 0, 0, 0, 2.0, 3.0],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 1.0, 1.0],
            [0, 0, 0, 0, 2.0, 1.0],
            [0, 0, 0, 0, 1.0, 2.0],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return Outline(types, coords)


def test_normalize_subpath_order_sorts_by_tight_bounds() -> None:
    output = NormalizeSubpathOrder()(_two_squares())

    assert output.coords[0, 4:6].tolist() == [1.0, 1.0]
    assert output.coords[4, 4:6].tolist() == [2.0, 2.0]


def test_normalize_subpath_order_does_not_depend_on_start_point() -> None:
    outline = _two_squares()
    rotated_coords = outline.coords.clone()
    rotated_coords[:3, 4:6] = rotated_coords[[1, 2, 0], 4:6]

    output = F.normalize_subpath_order(Outline(outline.types, rotated_coords))

    assert output.coords[0, 4:6].tolist() == [1.0, 1.0]
    assert output.coords[4, 4:6].tolist() == [3.0, 2.0]


def test_normalize_subpath_order_is_idempotent() -> None:
    once = F.normalize_subpath_order(_two_squares())
    twice = F.normalize_subpath_order(once)

    assert torch.equal(twice.types, once.types)
    assert torch.equal(twice.coords, once.coords)


def test_normalize_subpath_order_preserves_rendering() -> None:
    outline = _two_squares()
    output = F.normalize_subpath_order(outline)

    assert torch.equal(F.render_bitmap(output, 64), F.render_bitmap(outline, 64))
