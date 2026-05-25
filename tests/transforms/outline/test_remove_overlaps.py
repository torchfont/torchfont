from pathlib import Path
from urllib.parse import unquote

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.io import ElementType
from torchfont.transforms import remove_overlaps, render_bitmap

GOOGLE_FONTS_ROOT = Path("data/google/fonts")


def test_remove_overlaps_merges_overlapping_subpaths() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 2.0, 0.0],
            [0, 0, 0, 0, 2.0, 2.0],
            [0, 0, 0, 0, 0.0, 2.0],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 1.0, 0.0],
            [0, 0, 0, 0, 3.0, 0.0],
            [0, 0, 0, 0, 3.0, 2.0],
            [0, 0, 0, 0, 1.0, 2.0],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = remove_overlaps(types, coords)

    assert out_types[-1].item() == ElementType.END.value
    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 1
    assert out_types.tolist().count(ElementType.CLOSE.value) == 1
    expected = torch.tensor([0.0, 0.0, 3.0, 2.0])
    actual = torch.tensor(
        [
            out_coords[:, 4].min(),
            out_coords[:, 5].min(),
            out_coords[:, 4].max(),
            out_coords[:, 5].max(),
        ]
    )
    assert torch.allclose(actual, expected)


def _transform(sample: GlyphSample) -> dict[str, Tensor]:
    out_types, out_coords = remove_overlaps(sample.types, sample.coords)
    before = render_bitmap(sample.types, sample.coords, size=64, mode="fixed")
    after = render_bitmap(out_types, out_coords, size=64, mode="fixed")
    lost = (before == 255) & (after == 0)
    gained = (before == 0) & (after == 255)
    return {
        "hard_diff": (lost | gained).sum(),
        "style_idx": torch.tensor(sample.style_idx, dtype=torch.long),
        "codepoint": torch.tensor(sample.codepoint, dtype=torch.long),
        "seq_len_before": torch.tensor(sample.types.numel(), dtype=torch.long),
        "seq_len_after": torch.tensor(out_types.numel(), dtype=torch.long),
    }


def _style_path(label_id: str) -> str:
    prefix = "style:path="
    if not label_id.startswith(prefix):
        return label_id
    path, _, _ = label_id[len(prefix) :].partition(";")
    return unquote(path)


@pytest.mark.google_fonts_full
def test_remove_overlaps_preserves_google_fonts_bitmap_coverage(
    request: pytest.FixtureRequest,
) -> None:
    if not GOOGLE_FONTS_ROOT.is_dir():
        pytest.skip(f"Google Fonts checkout not available: {GOOGLE_FONTS_ROOT}")

    raw_limit: int = request.config.getoption("--google-fonts-limit")
    limit: int | None = None if raw_limit == 0 else raw_limit

    dataset = GlyphDataset(
        root=GOOGLE_FONTS_ROOT,
        patterns=(
            "apache/*/*.ttf",
            "ofl/*/*.ttf",
            "ufl/*/*.ttf",
            "!ofl/adobeblank/*.ttf",
        ),
        transform=_transform,
    )
    metadata = dataset.metadata
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=8,
        prefetch_factor=2,
    )

    failures: list[str] = []
    checked = 0
    for batch in dataloader:
        hard_diffs: Tensor = batch["hard_diff"]
        checked += hard_diffs.numel()

        for i in hard_diffs.nonzero(as_tuple=True)[0].tolist():
            style = metadata.styles[batch["style_idx"][i].item()]
            codepoint = batch["codepoint"][i].item()
            failures.append(
                f"path={_style_path(style.label_id)!r} style={style.name!r} "
                f"codepoint=U+{codepoint:04X} char={chr(codepoint)!r} "
                f"hard_diff_pixels={hard_diffs[i].item()} "
                f"seq_len={batch['seq_len_before'][i].item()}"
                f"->{batch['seq_len_after'][i].item()}"
            )

        if len(failures) >= 20:
            break
        if limit is not None and checked >= limit:
            break

    assert not failures, (
        "remove_overlaps changed filled bitmap coverage beyond antialiasing "
        f"tolerance for {len(failures)} (of at least {checked} checked) "
        "Google Fonts samples:\n" + "\n".join(failures)
    )
