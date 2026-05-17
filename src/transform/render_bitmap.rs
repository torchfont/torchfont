use skia_safe::{
    AlphaType, Color, ColorType, ImageInfo, Matrix, Paint, Path, PathBuilder, PathFillType,
    surfaces,
};

use crate::bounds::{Bounds, BoundsPen};
use crate::outline::ElementType;

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

struct RenderTransform {
    sx: f32,
    ky: f32,
    sy: f32,
    tx: f32,
    ty: f32,
}

pub(crate) fn render_bitmap(
    types: &[i64],
    coords: &[f32],
    size: u32,
    mode: RenderMode,
) -> Result<RenderedBitmap, RenderBitmapError> {
    let Some((path, bounds)) = build_path(types, coords, !matches!(mode, RenderMode::Fixed)) else {
        return Ok(blank_for_mode(size, mode));
    };

    let bitmap_size = size as f32;
    let Some((width, height, transform)) = render_target(bounds, bitmap_size, mode)? else {
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
    canvas.concat(&transform.matrix());

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
) -> Result<Option<(u32, u32, RenderTransform)>, RenderBitmapError> {
    match mode {
        RenderMode::Fixed => {
            let scale = bitmap_size / (FIXED_MAX - FIXED_MIN);
            let transform = RenderTransform {
                sx: scale,
                ky: 0.0,
                sy: -scale,
                tx: -FIXED_MIN * scale,
                ty: bitmap_size + FIXED_MIN * scale,
            };
            Ok(Some((bitmap_size as u32, bitmap_size as u32, transform)))
        }
        RenderMode::Bbox => {
            let Some(bounds) = bounds else {
                return Ok(None);
            };
            let width = bounds.width();
            let height = bounds.height();
            if width <= f32::EPSILON || height <= f32::EPSILON {
                return Ok(None);
            }
            let scale = bitmap_size / (FIXED_MAX - FIXED_MIN);
            let bitmap_width = (width * scale).ceil() as u32;
            let bitmap_height = (height * scale).ceil() as u32;
            if bitmap_width == 0 || bitmap_height == 0 {
                return Ok(None);
            }
            if bitmap_width > MAX_BITMAP_SIDE || bitmap_height > MAX_BITMAP_SIDE {
                return Err(RenderBitmapError::BboxTooLarge);
            }
            let transform = RenderTransform {
                sx: scale,
                ky: 0.0,
                sy: -scale,
                tx: -bounds.x_min * scale,
                ty: bounds.y_max * scale,
            };
            Ok(Some((bitmap_width, bitmap_height, transform)))
        }
        RenderMode::BboxSquare => {
            let Some(bounds) = bounds else {
                return Ok(None);
            };
            let width = bounds.width();
            let height = bounds.height();
            if width <= f32::EPSILON || height <= f32::EPSILON {
                return Ok(None);
            }
            let scale = bitmap_size / width.max(height);
            let offset_x = (bitmap_size - width * scale) * 0.5;
            let offset_y = (bitmap_size - height * scale) * 0.5;
            let transform = RenderTransform {
                sx: scale,
                ky: 0.0,
                sy: -scale,
                tx: offset_x - bounds.x_min * scale,
                ty: offset_y + bounds.y_max * scale,
            };
            Ok(Some((bitmap_size as u32, bitmap_size as u32, transform)))
        }
    }
}

fn build_path(types: &[i64], coords: &[f32], track_bounds: bool) -> Option<(Path, Option<Bounds>)> {
    let mut builder = PathBuilder::new_with_fill_type(PathFillType::Winding);
    let mut bounds = track_bounds.then(BoundsPen::default);
    for (&element_type, values) in types.iter().zip(coords.chunks_exact(6)) {
        match element_type {
            v if v == ElementType::MoveTo as i64 => {
                if let Some(bounds) = &mut bounds {
                    bounds.move_to(values[4], values[5]);
                }
                builder.move_to((values[4], values[5]));
            }
            v if v == ElementType::LineTo as i64 => {
                if let Some(bounds) = &mut bounds {
                    bounds.line_to(values[4], values[5]);
                }
                builder.line_to((values[4], values[5]));
            }
            v if v == ElementType::QuadTo as i64 => {
                if let Some(bounds) = &mut bounds {
                    bounds.quad_to(values[0], values[1], values[4], values[5]);
                }
                builder.quad_to((values[0], values[1]), (values[4], values[5]));
            }
            v if v == ElementType::CurveTo as i64 => {
                if let Some(bounds) = &mut bounds {
                    bounds.curve_to(
                        values[0], values[1], values[2], values[3], values[4], values[5],
                    );
                }
                builder.cubic_to(
                    (values[0], values[1]),
                    (values[2], values[3]),
                    (values[4], values[5]),
                );
            }
            v if v == ElementType::Close as i64 => {
                if let Some(bounds) = &mut bounds {
                    bounds.close();
                }
                builder.close();
            }
            _ => break,
        }
    }
    (!builder.is_empty()).then(|| (builder.detach(), bounds.and_then(BoundsPen::finish)))
}

impl RenderTransform {
    fn matrix(&self) -> Matrix {
        Matrix::new_all(
            self.sx, 0.0, self.tx, self.ky, self.sy, self.ty, 0.0, 0.0, 1.0,
        )
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
