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


def _hard_diff(a: Tensor, b: Tensor) -> Tensor:
    return ((a == 255) & (b == 0)) | ((a == 0) & (b == 255))


def _transform(sample: GlyphSample) -> dict[str, Tensor]:
    tf_types, tf_coords = remove_overlaps(sample.types, sample.coords)

    orig_winding_bitmap = render_bitmap(
        sample.types, sample.coords, size=64, mode="fixed", fill_rule="winding"
    )
    tf_winding_bitmap = render_bitmap(
        tf_types, tf_coords, size=64, mode="fixed", fill_rule="winding"
    )
    tf_even_odd_bitmap = render_bitmap(
        tf_types, tf_coords, size=64, mode="fixed", fill_rule="even_odd"
    )

    # Check 1: bitmap(original) ≈ bitmap(torchfont) — hard pixel differences only
    orig_tf_hard_diff = _hard_diff(orig_winding_bitmap, tf_winding_bitmap)
    bitmap_mismatch = orig_tf_hard_diff.any()

    # Check 2: torchfont(winding) ≈ torchfont(even-odd) — hard pixel differences only
    winding_even_odd_hard_diff = _hard_diff(tf_winding_bitmap, tf_even_odd_bitmap)
    tf_has_overlaps = winding_even_odd_hard_diff.any()

    return {
        "bitmap_mismatch": bitmap_mismatch,
        "tf_has_overlaps": tf_has_overlaps,
        "hard_diff_pixels": orig_tf_hard_diff.sum(),
        "style_idx": torch.tensor(sample.style_idx, dtype=torch.long),
        "codepoint": torch.tensor(sample.codepoint, dtype=torch.long),
    }


def _style_path(label_id: str) -> str:
    prefix = "style:path="
    if not label_id.startswith(prefix):
        return label_id
    path, _, _ = label_id[len(prefix) :].partition(";")
    return unquote(path)


@pytest.mark.google_fonts
def test_remove_overlaps_google_fonts(
    request: pytest.FixtureRequest,
) -> None:
    if not GOOGLE_FONTS_ROOT.is_dir():
        pytest.fail(f"Google Fonts checkout not available: {GOOGLE_FONTS_ROOT}")

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
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
    )

    max_failures = 20
    failures: list[str] = []
    bitmap_mismatch_count = 0
    tf_has_overlaps_count = 0
    checked = 0
    for batch in dataloader:
        bitmap_mismatch: Tensor = batch["bitmap_mismatch"]
        tf_has_overlaps: Tensor = batch["tf_has_overlaps"]
        hard_diff_pixels: Tensor = batch["hard_diff_pixels"]
        checked += bitmap_mismatch.numel()

        failed = bitmap_mismatch | tf_has_overlaps
        for i in failed.nonzero(as_tuple=True)[0].tolist():
            if len(failures) >= max_failures:
                break
            style = metadata.styles[batch["style_idx"][i].item()]
            codepoint = batch["codepoint"][i].item()
            if bitmap_mismatch[i].item():
                reason = "bitmap_mismatch"
                bitmap_mismatch_count += 1
            else:
                reason = "tf_has_overlaps"
                tf_has_overlaps_count += 1
            failures.append(
                f"path={_style_path(style.label_id)!r} style={style.name!r} "
                f"codepoint=U+{codepoint:04X} char={chr(codepoint)!r} "
                f"hard_diff_pixels={hard_diff_pixels[i].item()} "
                f"reason={reason}"
            )

        if len(failures) >= max_failures:
            break
        if limit is not None and checked >= limit:
            break

    capped = len(failures) >= max_failures
    count = (
        f"{len(failures)}+ (capped at {max_failures})" if capped else str(len(failures))
    )
    assert not failures, (
        f"remove_overlaps failed for {count} / {checked} Google Fonts glyphs: "
        f"bitmap_mismatch={bitmap_mismatch_count}, "
        f"tf_has_overlaps={tf_has_overlaps_count}\n" + "\n".join(failures)
    )
