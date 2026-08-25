from collections.abc import Callable

import pytest
import torch

from torchfont import Outline
from torchfont.transforms import ElasticTransform
from torchfont.transforms.functional import elastic


def test_elastic_is_reproducible_with_torch_seed(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    torch.manual_seed(7)
    first = ElasticTransform()(outline)
    torch.manual_seed(7)
    second = ElasticTransform()(outline)
    assert torch.equal(first.coords, second.coords)


def test_elastic_shares_displacement_between_outlines(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    first, second = ElasticTransform()([outline, outline])
    assert torch.equal(first.coords, second.coords)


def test_elastic_zero_alpha_is_identity(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    output = ElasticTransform(alpha=0.0)(outline)
    assert torch.equal(output.coords, outline.coords)


def test_elastic_constant_displacement_changes_only_active_coordinates(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    types, coords = simple_outline
    displacement = torch.empty((1, 2, 2, 2))
    displacement[..., 0] = 0.1
    displacement[..., 1] = -0.2

    output = elastic(Outline(types, coords), displacement)

    assert torch.allclose(
        output.coords[0, 4:6], coords[0, 4:6] + torch.tensor([0.1, -0.2])
    )
    assert close_end_zeros(types, output.coords)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("alpha", -0.1),
        ("alpha", (0.1, float("nan"))),
        ("sigma", -0.1),
        ("sigma", (float("inf"), 0.1)),
    ],
)
def test_elastic_rejects_invalid_parameters(
    name: str, value: float | tuple[float, float]
) -> None:
    with pytest.raises(ValueError, match=f"{name} values"):
        ElasticTransform(**{name: value})


@pytest.mark.parametrize("shape", [(2, 4, 4, 2), (1, 4, 4, 3), (1, 1, 4, 2)])
def test_elastic_rejects_incompatible_displacement_shape(
    simple_outline: tuple[torch.Tensor, torch.Tensor], shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="displacement must have shape"):
        elastic(Outline(*simple_outline), torch.zeros(shape))


def test_elastic_is_differentiable(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    coords = coords.requires_grad_()
    displacement = torch.rand((1, 4, 4, 2), requires_grad=True)

    elastic(Outline(types, coords), displacement).coords.sum().backward()

    assert coords.grad is not None
    assert displacement.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_elastic_preserves_cuda_device(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*(tensor.cuda() for tensor in simple_outline))
    output = ElasticTransform()(outline)
    assert output.types.device.type == "cuda"
    assert output.coords.device.type == "cuda"
