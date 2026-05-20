from collections.abc import Sequence

import torch

from torchfont.datasets import GlyphSample, NameRecord
from torchfont.io import ElementType

_ZERO_HEAD = torch.zeros(8, dtype=torch.float32)
_ZERO_HHEA = torch.zeros(10, dtype=torch.float32)
_ZERO_OS2 = torch.zeros(42, dtype=torch.float32)
_ZERO_POST = torch.zeros(4, dtype=torch.float32)
_ZERO_MAXP = torch.zeros(14, dtype=torch.float32)
_ZERO_HMTX = torch.zeros(2, dtype=torch.float32)
_ZERO_BOUNDS = torch.zeros(4, dtype=torch.float32)


def make_sample(types: torch.Tensor, coords: torch.Tensor) -> GlyphSample:
    return GlyphSample(
        types=types,
        coords=coords,
        style_idx=0,
        content_idx=0,
        head=_ZERO_HEAD,
        hhea=_ZERO_HHEA,
        os2=_ZERO_OS2,
        post=_ZERO_POST,
        maxp=_ZERO_MAXP,
        hmtx=_ZERO_HMTX,
        bounds=_ZERO_BOUNDS,
        name=NameRecord(
            copyright_notice="",
            family_name="",
            subfamily_name="",
            unique_font_identifier="",
            full_name="",
            version_string="",
            postscript_name="",
            trademark="",
            manufacturer_name="",
            designer="",
            description="",
            vendor_url="",
            designer_url="",
            license_description="",
            license_info_url="",
            compatible_full_name="",
            sample_text="",
            postscript_cid_findfont_name="",
            wws_family_name="",
            wws_subfamily_name="",
            light_background_palette="",
            dark_background_palette="",
            variations_postscript_name_prefix="",
        ),
        codepoint=0,
        glyph_name="",
    )


_Point = tuple[float, float]
_CubicSeg = tuple[_Point, _Point, _Point, _Point]
_QuadSeg = tuple[_Point, _Point, _Point]


def _lerp(
    a: tuple[float, float], b: tuple[float, float], t: float
) -> tuple[float, float]:
    return (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)


def _line_path_to_tensors(points: list[_Point]) -> tuple[torch.Tensor, torch.Tensor]:
    types = torch.tensor(
        [ElementType.MOVE_TO.value]
        + [ElementType.LINE_TO.value] * (len(points) - 1)
        + [ElementType.CLOSE.value, ElementType.END.value],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, x, y] for x, y in points] + [[0.0] * 6, [0.0] * 6],
        dtype=torch.float32,
    )
    return types, coords


def _quad_split(
    p0: _Point, p1: _Point, p2: _Point, t: float
) -> tuple[_QuadSeg, _QuadSeg]:
    q0, q1 = _lerp(p0, p1, t), _lerp(p1, p2, t)
    s = _lerp(q0, q1, t)
    return (p0, q0, s), (s, q1, p2)


def _split_quad(curve: _QuadSeg, ts: tuple[float, ...]) -> list[_QuadSeg]:
    pieces: list[_QuadSeg] = []
    current = curve
    previous_t = 0.0
    for t in ts:
        local_t = (t - previous_t) / (1.0 - previous_t)
        left, current = _quad_split(*current, local_t)
        pieces.append(left)
        previous_t = t
    pieces.append(current)
    return pieces


def _quad_segs_to_tensors(
    segs: Sequence[_QuadSeg],
) -> tuple[torch.Tensor, torch.Tensor]:
    types_list = (
        [ElementType.MOVE_TO.value]
        + [ElementType.QUAD_TO.value] * len(segs)
        + [ElementType.CLOSE.value, ElementType.END.value]
    )
    p0 = segs[0][0]
    coords_list: list[list[float]] = [[0.0, 0.0, 0.0, 0.0, p0[0], p0[1]]]
    coords_list.extend([s[1][0], s[1][1], 0.0, 0.0, s[2][0], s[2][1]] for s in segs)
    coords_list += [[0.0] * 6, [0.0] * 6]
    return (
        torch.tensor(types_list, dtype=torch.long),
        torch.tensor(coords_list, dtype=torch.float32),
    )


def _casteljau_split(
    p0: _Point,
    p1: _Point,
    p2: _Point,
    p3: _Point,
    t: float,
) -> tuple[_CubicSeg, _CubicSeg]:
    q0, q1, q2 = _lerp(p0, p1, t), _lerp(p1, p2, t), _lerp(p2, p3, t)
    r0, r1 = _lerp(q0, q1, t), _lerp(q1, q2, t)
    s = _lerp(r0, r1, t)
    return (p0, q0, r0, s), (s, r1, q2, p3)


def _cubic_segs_to_tensors(
    segs: Sequence[_CubicSeg],
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(segs)
    types_list = (
        [ElementType.MOVE_TO.value]
        + [ElementType.CURVE_TO.value] * n
        + [ElementType.CLOSE.value, ElementType.END.value]
    )
    p0 = segs[0][0]
    coords_list: list[list[float]] = [[0.0, 0.0, 0.0, 0.0, p0[0], p0[1]]]
    coords_list.extend(
        [s[1][0], s[1][1], s[2][0], s[2][1], s[3][0], s[3][1]] for s in segs
    )
    coords_list += [[0.0] * 6, [0.0] * 6]
    return (
        torch.tensor(types_list, dtype=torch.long),
        torch.tensor(coords_list, dtype=torch.float32),
    )


_CUBIC_CURVES: list[_CubicSeg] = [
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0)),
    ((0.0, 0.0), (0.5, 1.0), (1.5, 1.0), (2.0, 0.0)),
    ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
    ((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0)),
]


def _split_cubic(curve: _CubicSeg, ts: tuple[float, ...]) -> list[_CubicSeg]:
    pieces: list[_CubicSeg] = []
    current = curve
    previous_t = 0.0
    for t in ts:
        local_t = (t - previous_t) / (1.0 - previous_t)
        left, current = _casteljau_split(*current, local_t)
        pieces.append(left)
        previous_t = t
    pieces.append(current)
    return pieces


def _sub_cubic(curve: _CubicSeg, t1: float, t2: float) -> _CubicSeg:
    if t1 == 0.0:
        left, _ = _casteljau_split(*curve, t2)
        return left
    _, right = _casteljau_split(*curve, t1)
    t_adj = (t2 - t1) / (1.0 - t1)
    left, _ = _casteljau_split(*right, t_adj)
    return left


def _cubic_coords(curve: _CubicSeg) -> torch.Tensor:
    p1, p2, p3 = curve[1], curve[2], curve[3]
    return torch.tensor([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]], dtype=torch.float32)


def _assert_single_cubic_matches(
    types: torch.Tensor,
    coords: torch.Tensor,
    curve: _CubicSeg,
    *,
    atol: float = 1e-4,
) -> None:
    assert types.tolist().count(ElementType.CURVE_TO.value) == 1
    idx = types.tolist().index(ElementType.CURVE_TO.value)
    assert torch.allclose(coords[idx], _cubic_coords(curve), atol=atol)
