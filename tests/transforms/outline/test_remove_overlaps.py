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


def remove_overlaps_pathops(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
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
    po_types, po_coords = remove_overlaps_pathops(sample.types, sample.coords)

    orig_bitmap = render_bitmap(sample.types, sample.coords, size=64, mode="fixed")
    po_bitmap = render_bitmap(po_types, po_coords, size=64, mode="fixed")
    tf_bitmap = render_bitmap(tf_types, tf_coords, size=64, mode="fixed")

    # If orig and pathops bitmaps differ by a hard pixel (0↔255), pathops has a
    # bug — fall back to comparing torchfont against the original.  Otherwise
    # trust pathops as the reference (it normalises winding).
    # Antialiasing-only differences (values between 0 and 255) are ignored.
    po_hard_diff = ((orig_bitmap == 255) & (po_bitmap == 0)) | (
        (orig_bitmap == 0) & (po_bitmap == 255)
    )
    pathops_buggy = po_hard_diff.any()
    reference = orig_bitmap if pathops_buggy else po_bitmap

    hard_diff = ((reference == 255) & (tf_bitmap == 0)) | (
        (reference == 0) & (tf_bitmap == 255)
    )
    return {
        "hard_diff": hard_diff.sum(),
        "pathops_buggy": pathops_buggy.detach().clone(),
        "style_idx": torch.tensor(sample.style_idx, dtype=torch.long),
        "codepoint": torch.tensor(sample.codepoint, dtype=torch.long),
        "seq_len_before": torch.tensor(sample.types.numel(), dtype=torch.long),
        "seq_len_after": torch.tensor(tf_types.numel(), dtype=torch.long),
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

    max_failures = 20
    failures: list[str] = []
    checked = 0
    pathops_skipped = 0
    for batch in dataloader:
        hard_diffs: Tensor = batch["hard_diff"]
        buggy_flags: Tensor = batch["pathops_buggy"]
        checked += hard_diffs.numel()
        pathops_skipped += buggy_flags.sum().item()

        for i in hard_diffs.nonzero(as_tuple=True)[0].tolist():
            style = metadata.styles[batch["style_idx"][i].item()]
            codepoint = batch["codepoint"][i].item()
            reference = "original" if buggy_flags[i].item() else "pathops"
            failures.append(
                f"path={_style_path(style.label_id)!r} style={style.name!r} "
                f"codepoint=U+{codepoint:04X} char={chr(codepoint)!r} "
                f"hard_diff_pixels={hard_diffs[i].item()} "
                f"reference={reference} "
                f"seq_len={batch['seq_len_before'][i].item()}"
                f"->{batch['seq_len_after'][i].item()}"
            )

        if len(failures) >= max_failures:
            break
        if limit is not None and checked >= limit:
            break

    capped = len(failures) >= max_failures
    if capped:
        count = f"{len(failures)}+ (capped at {max_failures})"
    else:
        count = str(len(failures))
    assert not failures, (
        f"remove_overlaps bitmap mismatch in {count} / {checked} Google Fonts glyphs"
        f" ({pathops_skipped} used original as reference due to pathops bug):\n"
        + "\n".join(failures)
    )
