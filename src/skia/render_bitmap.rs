use skia_safe::{AlphaType, Color, ColorType, ImageInfo, Matrix, Paint, PathFillType, surfaces};

use crate::geom::{Bounds, Outline};

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

pub(crate) enum RenderBitmapError {
    BboxTooLarge,
}

pub(crate) fn render_bitmap(
    outline: &Outline,
    size: u32,
    mode: RenderMode,
    fill_type: PathFillType,
) -> Result<RenderedBitmap, RenderBitmapError> {
    let Some((path, bounds)) =
        super::build_skia_path(outline, !matches!(mode, RenderMode::Fixed), fill_type)
    else {
        return Ok(blank_for_mode(size, mode));
    };

    let bitmap_size = size as f32;
    let Some((width, height, matrix)) = render_target(bounds, bitmap_size, mode)? else {
        return Ok(blank_for_mode(size, mode));
    };

    let mut data = vec![0u8; (width as usize).saturating_mul(height as usize)];
    let image_info = ImageInfo::new(
        (width as i32, height as i32),
        ColorType::Alpha8,
        AlphaType::Premul,
        None,
    );
    let Some(mut surface) = surfaces::wrap_pixels(&image_info, &mut data, width as usize, None)
    else {
        return Ok(blank_bitmap(width, height));
    };
    let canvas = surface.canvas();
    canvas.clear(Color::TRANSPARENT);
    canvas.concat(&matrix);

    let mut paint = Paint::default();
    paint.set_anti_alias(true);
    paint.set_color(Color::WHITE);
    canvas.draw_path(&path, &paint);
    drop(surface);

    Ok(RenderedBitmap {
        data,
        width,
        height,
    })
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
                render_matrix(scale, -FIXED_MIN * scale, bitmap_size + FIXED_MIN * scale),
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
                render_matrix(scale, -bounds.x_min * scale, bounds.y_max * scale),
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
                render_matrix(
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

fn render_matrix(scale: f32, tx: f32, ty: f32) -> Matrix {
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
