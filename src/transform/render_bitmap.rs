use skia_safe::{Canvas, Color, ImageInfo, Matrix, Paint, Path, PathFillType};

use super::skia::build_skia_path;
use crate::outline::{BezPath, Bounds, bounds_from_outline};

const FIXED_MIN: f32 = -0.25;
const FIXED_MAX: f32 = 1.25;
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
    let path = build_skia_path(outline).filter(|path| !path.segment_masks().is_empty());
    let bounds = if matches!(mode, RenderMode::Fixed) {
        None
    } else {
        bounds_from_outline(outline)
    };
    let Some(path) = path else {
        return Ok(blank_for_mode(size, mode));
    };

    let bitmap_size = size as f32;
    let Some((width, height, transform)) = render_target(bounds, bitmap_size, mode)? else {
        return Ok(blank_for_mode(size, mode));
    };

    let data = draw_alpha_path(path, width, height, transform, fill_rule, antialias);
    Ok(RenderedBitmap {
        data,
        width,
        height,
    })
}

pub(crate) fn render_bitmap_in_bounds(
    outline: &BezPath,
    size: u32,
    bounds: Bounds,
    fill_rule: PathFillType,
    antialias: bool,
) -> RenderedBitmap {
    let path = build_skia_path(outline).filter(|path| !path.segment_masks().is_empty());
    let target = render_target(Some(bounds), size as f32, RenderMode::BboxSquare)
        .expect("bbox-square rendering never errors");
    let (Some(path), Some((width, height, transform))) = (path, target) else {
        return blank_bitmap(size, size);
    };

    RenderedBitmap {
        data: draw_alpha_path(path, width, height, transform, fill_rule, antialias),
        width,
        height,
    }
}

fn draw_alpha_path(
    mut path: Path,
    width: u32,
    height: u32,
    transform: Matrix,
    fill_rule: PathFillType,
    antialias: bool,
) -> Vec<u8> {
    path.set_fill_type(fill_rule);
    let mut data = vec![0u8; width as usize * height as usize];
    let info = ImageInfo::new_a8((width as i32, height as i32));
    {
        // Draw into the final one-byte-per-pixel buffer; no RGBA allocation or
        // alpha extraction is needed before transferring it to NumPy.
        let canvas = Canvas::from_raster_direct(&info, &mut data, width as usize, None)
            .expect("valid alpha bitmap");
        let mut paint = Paint::default();
        paint.set_color(Color::WHITE).set_anti_alias(antialias);
        canvas.concat(&transform).draw_path(&path, &paint);
    }
    data
}

fn render_target(
    bounds: Option<Bounds>,
    bitmap_size: f32,
    mode: RenderMode,
) -> Result<Option<(u32, u32, Matrix)>, RenderBitmapError> {
    match mode {
        RenderMode::Fixed => {
            let scale = bitmap_size / (FIXED_MAX - FIXED_MIN);
            Ok(Some((
                bitmap_size as u32,
                bitmap_size as u32,
                render_transform(scale, -FIXED_MIN * scale, bitmap_size + FIXED_MIN * scale),
            )))
        }
        RenderMode::Bbox => {
            let Some((bounds, width, height)) = nonempty_bounds(bounds) else {
                return Ok(None);
            };
            let scale = bitmap_size / (FIXED_MAX - FIXED_MIN);
            let bitmap_width = (width * scale).ceil() as u32;
            let bitmap_height = (height * scale).ceil() as u32;
            if bitmap_width == 0 || bitmap_height == 0 {
                return Ok(None);
            }
            if bitmap_width > MAX_BITMAP_SIDE || bitmap_height > MAX_BITMAP_SIDE {
                return Err(RenderBitmapError::BboxTooLarge);
            }
            Ok(Some((
                bitmap_width,
                bitmap_height,
                render_transform(scale, -bounds.x_min * scale, bounds.y_max * scale),
            )))
        }
        RenderMode::BboxSquare => {
            let Some((bounds, width, height)) = nonempty_bounds(bounds) else {
                return Ok(None);
            };
            let scale = bitmap_size / width.max(height);
            let offset_x = (bitmap_size - width * scale) * 0.5;
            let offset_y = (bitmap_size - height * scale) * 0.5;
            Ok(Some((
                bitmap_size as u32,
                bitmap_size as u32,
                render_transform(
                    scale,
                    offset_x - bounds.x_min * scale,
                    offset_y + bounds.y_max * scale,
                ),
            )))
        }
    }
}

fn nonempty_bounds(bounds: Option<Bounds>) -> Option<(Bounds, f32, f32)> {
    let b = bounds?;
    let w = b.width();
    let h = b.height();
    (w > f32::EPSILON && h > f32::EPSILON).then_some((b, w, h))
}

fn render_transform(scale: f32, tx: f32, ty: f32) -> Matrix {
    Matrix::new_all(scale, 0.0, tx, 0.0, -scale, ty, 0.0, 0.0, 1.0)
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
