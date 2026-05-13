use tiny_skia::{FillRule, Mask, Path, PathBuilder, Transform};

use crate::bounds::{Bounds, BoundsPen};
use crate::outline::Command;

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
    types: &[i64],
    coords: &[f32],
    size: u32,
    mode: RenderMode,
) -> Result<RenderedBitmap, RenderBitmapError> {
    let Some((path, bounds)) = build_path(types, coords) else {
        return Ok(blank_for_mode(size, mode));
    };

    let bitmap_size = size as f32;
    let Some((width, height, transform)) = render_target(bounds, bitmap_size, mode)? else {
        return Ok(blank_for_mode(size, mode));
    };

    let Some(mut mask) = Mask::new(width, height) else {
        return Ok(blank_bitmap(width, height));
    };
    mask.fill_path(&path, FillRule::Winding, true, transform);
    Ok(RenderedBitmap {
        data: mask.data().to_vec(),
        width,
        height,
    })
}

fn render_target(
    bounds: Option<Bounds>,
    bitmap_size: f32,
    mode: RenderMode,
) -> Result<Option<(u32, u32, Transform)>, RenderBitmapError> {
    match mode {
        RenderMode::Fixed => {
            let scale = bitmap_size / (FIXED_MAX - FIXED_MIN);
            let transform = Transform::from_row(
                scale,
                0.0,
                0.0,
                -scale,
                -FIXED_MIN * scale,
                bitmap_size + FIXED_MIN * scale,
            );
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
            let transform = Transform::from_row(
                scale,
                0.0,
                0.0,
                -scale,
                -bounds.x_min * scale,
                bounds.y_max * scale,
            );
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
            let transform = Transform::from_row(
                scale,
                0.0,
                0.0,
                -scale,
                offset_x - bounds.x_min * scale,
                offset_y + bounds.y_max * scale,
            );
            Ok(Some((bitmap_size as u32, bitmap_size as u32, transform)))
        }
    }
}

fn build_path(types: &[i64], coords: &[f32]) -> Option<(Path, Option<Bounds>)> {
    let mut builder = PathBuilder::with_capacity(types.len(), coords.len() / 2);
    let mut bounds = BoundsPen::default();
    for (&command, values) in types.iter().zip(coords.chunks_exact(6)) {
        match command {
            v if v == Command::MoveTo as i64 => {
                bounds.move_to(values[4], values[5]);
                builder.move_to(values[4], values[5]);
            }
            v if v == Command::LineTo as i64 => {
                bounds.line_to(values[4], values[5]);
                builder.line_to(values[4], values[5]);
            }
            v if v == Command::QuadTo as i64 => {
                bounds.quad_to(values[0], values[1], values[4], values[5]);
                builder.quad_to(values[0], values[1], values[4], values[5]);
            }
            v if v == Command::CurveTo as i64 => {
                bounds.curve_to(
                    values[0], values[1], values[2], values[3], values[4], values[5],
                );
                builder.cubic_to(
                    values[0], values[1], values[2], values[3], values[4], values[5],
                );
            }
            v if v == Command::Close as i64 => {
                bounds.close();
                builder.close();
            }
            _ => break,
        }
    }
    builder.finish().map(|path| (path, bounds.finish()))
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
