import torch

from torchfont.transforms import patchify


def test_patchify_reshapes_exact_multiple() -> None:
    types = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    coords = torch.zeros(4, 6, dtype=torch.float32)

    out_types, out_coords = patchify(types, coords, patch_size=2)

    assert out_types.shape == (2, 2)
    assert out_coords.shape == (2, 2, 6)
    assert torch.equal(out_types, types.view(2, 2))


def test_patchify_pads_when_not_exact_multiple() -> None:
    types = torch.tensor([1, 2, 3], dtype=torch.long)
    coords = torch.zeros(3, 6, dtype=torch.float32)

    out_types, out_coords = patchify(types, coords, patch_size=2)

    assert out_types.shape == (2, 2)
    assert out_coords.shape == (2, 2, 6)
    assert out_types[1, 1].item() == 0
