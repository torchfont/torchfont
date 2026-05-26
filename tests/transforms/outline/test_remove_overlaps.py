from pathlib import Path
from urllib.parse import unquote

import pathops
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.io import ElementType
from torchfont.transforms import remove_overlaps, render_bitmap

GOOGLE_FONTS_ROOT = Path("data/google/fonts")

_VERB_MOVE = pathops.PathVerb.MOVE.value
_VERB_LINE = pathops.PathVerb.LINE.value
_VERB_QUAD = pathops.PathVerb.QUAD.value
_VERB_CUBIC = pathops.PathVerb.CUBIC.value
_VERB_CLOSE = pathops.PathVerb.CLOSE.value


def _tensors_to_pathops(types: Tensor, coords: Tensor) -> pathops.Path:
    path: pathops.Path = pathops.Path()
    pen = path.getPen()
    in_subpath = False
    for i in range(types.numel()):
        t = int(types[i].item())
        c = coords[i]
        if t == ElementType.MOVE_TO:
            if in_subpath:
                pen.endPath()
            pen.moveTo((float(c[4]), float(c[5])))
            in_subpath = True
        elif t == ElementType.LINE_TO:
            pen.lineTo((float(c[4]), float(c[5])))
        elif t == ElementType.QUAD_TO:
            pen.qCurveTo((float(c[0]), float(c[1])), (float(c[4]), float(c[5])))
        elif t == ElementType.CURVE_TO:
            pen.curveTo(
                (float(c[0]), float(c[1])),
                (float(c[2]), float(c[3])),
                (float(c[4]), float(c[5])),
            )
        elif t == ElementType.CLOSE:
            pen.closePath()
            in_subpath = False
        elif t == ElementType.END:
            if in_subpath:
                pen.endPath()
            break
    return path


def _pathops_to_tensors(path: pathops.Path) -> tuple[Tensor, Tensor]:
    types_list: list[int] = []
    coords_list: list[list[float]] = []
    for verb, pts in path:
        if verb == _VERB_MOVE:
            types_list.append(ElementType.MOVE_TO)
            coords_list.append([0.0, 0.0, 0.0, 0.0, pts[0][0], pts[0][1]])
        elif verb == _VERB_LINE:
            types_list.append(ElementType.LINE_TO)
            coords_list.append([0.0, 0.0, 0.0, 0.0, pts[0][0], pts[0][1]])
        elif verb == _VERB_QUAD:
            types_list.append(ElementType.QUAD_TO)
            coords_list.append([pts[0][0], pts[0][1], 0.0, 0.0, pts[1][0], pts[1][1]])
        elif verb == _VERB_CUBIC:
            types_list.append(ElementType.CURVE_TO)
            coords_list.append(
                [pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1]]
            )
        elif verb == _VERB_CLOSE:
            types_list.append(ElementType.CLOSE)
            coords_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    types_list.append(ElementType.END)
    coords_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return (
        torch.tensor(types_list, dtype=torch.long),
        torch.tensor(coords_list, dtype=torch.float32),
    )


def _pathops_simplify(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    path = _tensors_to_pathops(types, coords)
    try:
        simplified = pathops.simplify(path, fix_winding=True)
    except pathops.PathOpsError:
        return types, coords
    return _pathops_to_tensors(simplified)


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
    tf_types, tf_coords = remove_overlaps(sample.types, sample.coords)

    # Check 1: bitmap(original) ≈ bitmap(torchfont) — hard pixel differences only
    orig_bitmap = render_bitmap(sample.types, sample.coords, size=64, mode="fixed")
    tf_bitmap = render_bitmap(tf_types, tf_coords, size=64, mode="fixed")
    hard_diff = ((orig_bitmap == 255) & (tf_bitmap == 0)) | (
        (orig_bitmap == 0) & (tf_bitmap == 255)
    )
    bitmap_mismatch = hard_diff.any()

    # Check 2: pathops(torchfont) == torchfont — no overlaps remain
    po_types, po_coords = _pathops_simplify(tf_types, tf_coords)
    tf_has_overlaps = torch.tensor(
        not (torch.equal(po_types, tf_types) and torch.equal(po_coords, tf_coords)),
        dtype=torch.bool,
    )

    return {
        "bitmap_mismatch": bitmap_mismatch,
        "tf_has_overlaps": tf_has_overlaps,
        "hard_diff_pixels": hard_diff.sum(),
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
