"""Which functional kernels carry gradients, and how the rest fail."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Callable


def _curved() -> Outline:
    types = torch.tensor(
        [
            ElementType.MOVE_TO,
            ElementType.CURVE_TO,
            ElementType.LINE_TO,
            ElementType.CLOSE,
            ElementType.END,
        ],
        dtype=torch.long,
    )
    coords = torch.zeros(5, 6)
    coords[1] = torch.tensor([0.1, 0.9, 0.6, 0.9, 1.0, 0.2])
    coords[2, 4:] = torch.tensor([0.4, 0.7])
    return Outline(types, coords)


def _grad_outline() -> tuple[Outline, torch.Tensor]:
    outline = _curved()
    coords = outline.coords.clone()
    coords.requires_grad_()
    return Outline(outline.types, coords), coords


def test_affine_is_differentiable() -> None:
    outline, coords = _grad_outline()

    F.affine(
        outline, angle=10.0, scale=1.2, translate=(0.1, 0.0)
    ).coords.sum().backward()

    assert coords.grad is not None
    assert coords.grad.abs().sum() > 0


def test_affine_does_not_backpropagate_through_the_bbox_centre() -> None:
    """The pivot is a reference frame, so a pure translation has unit gradient."""
    outline, coords = _grad_outline()

    F.affine(outline, translate=(0.25, 0.0)).coords.sum().backward()

    active = F.affine(outline, translate=(0.25, 0.0)).coords != 0
    assert coords.grad is not None
    assert torch.equal(coords.grad[active.detach()], torch.ones(int(active.sum())))


def test_add_coordinate_noise_is_differentiable_in_both_arguments() -> None:
    outline, coords = _grad_outline()
    noise = torch.zeros(5, 3, 2, requires_grad=True)

    F.add_coordinate_noise(outline, noise).coords.sum().backward()

    assert coords.grad is not None
    assert noise.grad is not None


def test_scale_is_differentiable() -> None:
    outline, coords = _grad_outline()

    F.scale(outline, (0.9, 1.1)).coords.sum().backward()

    assert coords.grad is not None


def test_rotate_is_differentiable() -> None:
    outline, coords = _grad_outline()

    F.rotate(outline, 10.0).coords.sum().backward()

    assert coords.grad is not None


def test_flip_without_winding_preservation_is_differentiable() -> None:
    outline, coords = _grad_outline()

    F.horizontal_flip(outline, preserve_winding=False).coords.sum().backward()

    assert coords.grad is not None


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("cubic_to_quad", F.cubic_to_quad),
        ("quad_to_cubic", F.quad_to_cubic),
        ("merge_curves", F.merge_curves),
        ("remove_overlaps", F.remove_overlaps),
        ("render_bitmap", F.render_bitmap),
        ("normalize_subpath_start_points", F.normalize_subpath_start_points),
        ("horizontal_flip", F.horizontal_flip),
        ("vertical_flip", F.vertical_flip),
    ],
)
def test_rust_kernels_name_themselves_when_grad_is_required(
    name: str, call: Callable[[Outline], object]
) -> None:
    outline, _ = _grad_outline()

    with pytest.raises(RuntimeError, match=f"{name}.*is not differentiable"):
        call(outline)
