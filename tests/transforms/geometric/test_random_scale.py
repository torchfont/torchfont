from collections.abc import Callable

import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import RandomScale
from torchfont.transforms.functional import scale


def test_scale_applies_independent_factors(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    output = scale(Outline(types, coords), (2.0, 0.5))

    line_idx = types.tolist().index(ElementType.LINE_TO.value)
    assert output.coords[line_idx, 4].item() == pytest.approx(
        coords[line_idx, 4].item() * 2.0 - 0.5
    )
    assert output.coords[line_idx, 5].item() == pytest.approx(
        coords[line_idx, 5].item() * 0.5 + 0.25
    )


def test_random_scale_is_reproducible_with_torch_seed(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    transform = RandomScale(scale_x=(0.8, 1.2), scale_y=(0.9, 1.1))
    torch.manual_seed(7)
    first = transform(outline)
    torch.manual_seed(7)
    second = transform(outline)
    assert torch.equal(first.coords, second.coords)


def test_random_scale_shares_factors_between_outlines(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    outline = Outline(*simple_outline)
    first, second = RandomScale(scale_x=(0.8, 1.2), scale_y=(0.9, 1.1))(
        [outline, outline]
    )
    assert torch.equal(first.coords, second.coords)
    assert close_end_zeros(first.types, first.coords)


def test_random_scale_defaults_to_identity(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    output = RandomScale()(outline)
    assert torch.equal(output.coords, outline.coords)


def test_random_scale_samples_requested_ranges() -> None:
    transform = RandomScale(scale_x=(0.8, 0.9), scale_y=(1.1, 1.2))

    scale_x, scale_y = transform.make_params([])["factors"]

    assert 0.8 <= scale_x <= 0.9
    assert 1.1 <= scale_y <= 1.2


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("scale_x", (0.0, 1.0), "positive and finite"),
        ("scale_x", (1.0, float("nan")), "positive and finite"),
        ("scale_y", (1.0, float("inf")), "positive and finite"),
        ("scale_y", (1.2, 0.8), "ordered"),
    ],
)
def test_random_scale_rejects_invalid_ranges(
    name: str, value: tuple[float, float], message: str
) -> None:
    with pytest.raises(ValueError, match=f"{name} values must be {message}"):
        RandomScale(**{name: value})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_scale_rejects_cuda_input(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*(tensor.cuda() for tensor in simple_outline))
    with pytest.raises(NotImplementedError, match="CUDA"):
        RandomScale(scale_x=(0.8, 1.2))(outline)
