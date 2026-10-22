use kurbo::{Affine, Shape};
use skia_safe::{Canvas, Color, ImageInfo, Matrix, Paint, Path, PathFillType};

use super::skia::build_skia_path;
use crate::outline::{BezPath, Rect};

const FIXED_MIN: f64 = -0.25;
const FIXED_MAX: f64 = 1.25;
const MAX_BITMAP_SIDE: u32 = 4096;

#[derive(Clone, Copy)]
pub(crate) enum RenderMode {
    Fixed,
    Bbox,
    BboxSquare,
}

pub(crate) struct RenderedBitmap {
    pub(crate) data: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

#[derive(Debug)]
pub(crate) enum RenderBitmapError {
    BboxTooLarge,
}

pub(crate) fn render_bitmap(
    outline: &BezPath,
    size: u32,
    mode: RenderMode,
    fill_rule: PathFillType,
    antialias: bool,
) -> Result<RenderedBitmap, RenderBitmapError> {
    let path =
        Some(build_skia_path(outline.elements())).filter(|path| !path.segment_masks().is_empty());
    let Some(path) = path else {
        return Ok(blank_for_mode(size, mode));
    };

    let bounds = match mode {
        RenderMode::Fixed => Rect::ZERO,
        RenderMode::Bbox | RenderMode::BboxSquare => outline.bounding_box(),
    };
    let bitmap_size = f64::from(size);
    let Some((width, height, transform)) = render_target(bounds, bitmap_size, mode)? else {
        return Ok(blank_for_mode(size, mode));
    };

    Ok(draw_alpha_path(
        path, width, height, transform, fill_rule, antialias,
    ))
}

pub(crate) fn render_bitmap_in_bounds(
    outline: &BezPath,
    size: u32,
    bounds: Rect,
    fill_rule: PathFillType,
    antialias: bool,
) -> RenderedBitmap {
    let path =
        Some(build_skia_path(outline.elements())).filter(|path| !path.segment_masks().is_empty());
    let target = render_target(bounds, f64::from(size), RenderMode::BboxSquare)
        .expect("bbox-square rendering never errors");
    let (Some(path), Some((width, height, transform))) = (path, target) else {
        return blank_bitmap(size, size);
    };

    draw_alpha_path(path, width, height, transform, fill_rule, antialias)
}

fn draw_alpha_path(
    mut path: Path,
    width: u32,
    height: u32,
    transform: Affine,
    fill_rule: PathFillType,
    antialias: bool,
) -> RenderedBitmap {
    path.set_fill_type(fill_rule);
    let mut data = vec![0u8; width as usize * height as usize];
    let info = ImageInfo::new_a8((width as i32, height as i32));
    {
        let canvas = Canvas::from_raster_direct(&info, &mut data, width as usize, None)
            .expect("valid alpha bitmap");
        let mut paint = Paint::default();
        paint.set_color(Color::WHITE).set_anti_alias(antialias);
        let matrix = Matrix::from_affine(&transform.as_coeffs().map(|value| value as f32));
        canvas.concat(&matrix).draw_path(&path, &paint);
    }
    RenderedBitmap {
        data,
        width,
        height,
    }
}

fn render_target(
    bounds: Rect,
    bitmap_size: f64,
    mode: RenderMode,
) -> Result<Option<(u32, u32, Affine)>, RenderBitmapError> {
    let size = bounds.size();
    if !matches!(mode, RenderMode::Fixed) && size.is_zero_area() {
        return Ok(None);
    }
    match mode {
        RenderMode::Fixed => {
            let scale = bitmap_size / (FIXED_MAX - FIXED_MIN);
            Ok(Some((
                bitmap_size as u32,
                bitmap_size as u32,
                Affine::FLIP_Y
                    .then_scale(scale)
                    .then_translate((-FIXED_MIN * scale, bitmap_size + FIXED_MIN * scale).into()),
            )))
        }
        RenderMode::Bbox => {
            let scale = bitmap_size / (FIXED_MAX - FIXED_MIN);
            let bitmap = (size * scale).ceil();
            let bitmap_width = bitmap.width as u32;
            let bitmap_height = bitmap.height as u32;
            if bitmap_width == 0 || bitmap_height == 0 {
                return Ok(None);
            }
            if bitmap_width > MAX_BITMAP_SIDE || bitmap_height > MAX_BITMAP_SIDE {
                return Err(RenderBitmapError::BboxTooLarge);
            }
            Ok(Some((
                bitmap_width,
                bitmap_height,
                Affine::translate((-bounds.x0, -bounds.y1)).then_scale_non_uniform(scale, -scale),
            )))
        }
        RenderMode::BboxSquare => {
            let scale = bitmap_size / size.max_side();
            Ok(Some((
                bitmap_size as u32,
                bitmap_size as u32,
                Affine::translate(-bounds.center().to_vec2())
                    .then_scale_non_uniform(scale, -scale)
                    .then_translate((bitmap_size * 0.5, bitmap_size * 0.5).into()),
            )))
        }
    }
}

fn blank_bitmap(width: u32, height: u32) -> RenderedBitmap {
    RenderedBitmap {
        data: vec![0u8; (width as usize).saturating_mul(height as usize)],
        width,
        height,
    }
}

fn blank_for_mode(size: u32, mode: RenderMode) -> RenderedBitmap {
    match mode {
        RenderMode::Bbox => blank_bitmap(0, 0),
        RenderMode::Fixed | RenderMode::BboxSquare => blank_bitmap(size, size),
    }
}

#[cfg(test)]
mod tests {
    use kurbo::{Rect, Shape};

    use super::*;

    #[test]
    fn clips_to_alpha_buffers_with_unaligned_row_widths() {
        let outline = Rect::new(-1.0, -1.0, 2.0, 2.0).to_path(0.1);
        for size in [1, 17, 64] {
            for antialias in [false, true] {
                let bitmap = render_bitmap(
                    &outline,
                    size,
                    RenderMode::Fixed,
                    PathFillType::Winding,
                    antialias,
                )
                .unwrap_or_else(|_| panic!("fixed bitmap is valid"));

                assert_eq!((bitmap.width, bitmap.height), (size, size));
                assert_eq!(bitmap.data, vec![255; (size * size) as usize]);
            }
        }
    }

    #[test]
    fn square_crop_is_invariant_to_small_outline_scales() {
        let render = |extent| {
            render_bitmap(
                &Rect::new(0.0, 0.0, extent, extent).to_path(0.1),
                17,
                RenderMode::BboxSquare,
                PathFillType::Winding,
                true,
            )
            .unwrap()
            .data
        };
        assert_eq!(render(1e-10), render(1.0));
        assert_eq!(render(1.0), vec![255; 17 * 17]);
    }

    #[test]
    fn empty_outlines_have_zero_coverage() {
        for mode in [RenderMode::Fixed, RenderMode::BboxSquare, RenderMode::Bbox] {
            let bitmap = render_bitmap(&BezPath::new(), 17, mode, PathFillType::Winding, true)
                .unwrap_or_else(|_| panic!("empty outline is valid"));
            let side = if matches!(mode, RenderMode::Bbox) {
                0
            } else {
                17
            };

            assert_eq!((bitmap.width, bitmap.height), (side, side));
            assert_eq!(bitmap.data, vec![0; (side * side) as usize]);
        }
    }

    #[test]
    fn move_only_outlines_have_empty_cropped_bitmaps() {
        let mut outline = BezPath::new();
        outline.move_to((0.0, 0.0));
        outline.move_to((1.0, 1.0));
        let bitmap = render_bitmap(&outline, 17, RenderMode::Bbox, PathFillType::Winding, true)
            .unwrap_or_else(|_| panic!("move-only outline is valid"));

        assert_eq!((bitmap.width, bitmap.height), (0, 0));
        assert!(bitmap.data.is_empty());
    }
}
