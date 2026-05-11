use tiny_skia::{FillRule, Mask, Path, PathBuilder, Transform};

use crate::pen::Command;

const PADDING: f32 = 4.0;
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

pub(crate) fn render_bitmap(
    types: &[i64],
    coords: &[f32],
    size: u32,
    mode: RenderMode,
) -> RenderedBitmap {
    let Some(path) = build_path(types, coords) else {
        return blank_bitmap(size, size);
    };

    let bitmap_size = size as f32;
    let content_size = bitmap_size - 2.0 * PADDING;
    if content_size <= 0.0 {
        return blank_bitmap(size, size);
    }
    let Some((width, height, transform)) = render_target(&path, bitmap_size, content_size, mode)
    else {
        return blank_bitmap(size, size);
    };

    let Some(mut mask) = Mask::new(width, height) else {
        return blank_bitmap(width, height);
    };
    mask.fill_path(&path, FillRule::Winding, true, transform);
    RenderedBitmap {
        data: mask.data().to_vec(),
        width,
        height,
    }
}

fn render_target(
    path: &Path,
    bitmap_size: f32,
    content_size: f32,
    mode: RenderMode,
) -> Option<(u32, u32, Transform)> {
    match mode {
        RenderMode::Fixed => {
            let scale = content_size / (FIXED_MAX - FIXED_MIN);
            let transform = Transform::from_row(
                scale,
                0.0,
                0.0,
                -scale,
                PADDING - FIXED_MIN * scale,
                bitmap_size - PADDING + FIXED_MIN * scale,
            );
            Some((bitmap_size as u32, bitmap_size as u32, transform))
        }
        RenderMode::Bbox => {
            let bounds = path.compute_tight_bounds().unwrap_or_else(|| path.bounds());
            let width = bounds.width();
            let height = bounds.height();
            if width <= f32::EPSILON || height <= f32::EPSILON {
                return None;
            }
            let scale = content_size / (FIXED_MAX - FIXED_MIN);
            let bitmap_width = (width * scale + 2.0 * PADDING).ceil() as u32;
            let bitmap_height = (height * scale + 2.0 * PADDING).ceil() as u32;
            if bitmap_width == 0
                || bitmap_height == 0
                || bitmap_width > MAX_BITMAP_SIDE
                || bitmap_height > MAX_BITMAP_SIDE
            {
                return None;
            }
            let transform = Transform::from_row(
                scale,
                0.0,
                0.0,
                -scale,
                PADDING - bounds.left() * scale,
                PADDING + bounds.bottom() * scale,
            );
            Some((bitmap_width, bitmap_height, transform))
        }
        RenderMode::BboxSquare => {
            let bounds = path.compute_tight_bounds().unwrap_or_else(|| path.bounds());
            let width = bounds.width();
            let height = bounds.height();
            if width <= f32::EPSILON || height <= f32::EPSILON {
                return None;
            }
            let scale = content_size / width.max(height);
            let offset_x = (bitmap_size - width * scale) * 0.5;
            let offset_y = (bitmap_size - height * scale) * 0.5;
            let transform = Transform::from_row(
                scale,
                0.0,
                0.0,
                -scale,
                offset_x - bounds.left() * scale,
                offset_y + bounds.bottom() * scale,
            );
            Some((bitmap_size as u32, bitmap_size as u32, transform))
        }
    }
}

fn build_path(types: &[i64], coords: &[f32]) -> Option<Path> {
    let mut builder = PathBuilder::with_capacity(types.len(), coords.len() / 2);
    for (&command, values) in types.iter().zip(coords.chunks_exact(6)) {
        match command {
            v if v == Command::MoveTo as i64 => builder.move_to(values[4], values[5]),
            v if v == Command::LineTo as i64 => builder.line_to(values[4], values[5]),
            v if v == Command::QuadTo as i64 => {
                builder.quad_to(values[0], values[1], values[4], values[5])
            }
            v if v == Command::CurveTo as i64 => builder.cubic_to(
                values[0], values[1], values[2], values[3], values[4], values[5],
            ),
            v if v == Command::Close as i64 => builder.close(),
            _ => break,
        }
    }
    builder.finish()
}

fn blank_bitmap(width: u32, height: u32) -> RenderedBitmap {
    RenderedBitmap {
        data: vec![0u8; (width as usize).saturating_mul(height as usize)],
        width,
        height,
    }
}
