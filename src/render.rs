use tiny_skia::{FillRule, Mask, Path, PathBuilder, Transform};

use crate::pen::Command;

const PADDING: f32 = 4.0;

pub(crate) fn render_bitmap(types: &[i64], coords: &[f32], size: u32) -> Vec<u8> {
    let Some(path) = build_path(types, coords) else {
        return blank_bitmap(size);
    };
    let bounds = path.compute_tight_bounds().unwrap_or_else(|| path.bounds());
    let width = bounds.width();
    let height = bounds.height();
    if width <= f32::EPSILON || height <= f32::EPSILON {
        return blank_bitmap(size);
    }

    let bitmap_size = size as f32;
    let content_size = bitmap_size - 2.0 * PADDING;
    if content_size <= 0.0 {
        return blank_bitmap(size);
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

    let Some(mut mask) = Mask::new(size, size) else {
        return blank_bitmap(size);
    };
    mask.fill_path(&path, FillRule::Winding, true, transform);
    mask.data().to_vec()
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

fn blank_bitmap(size: u32) -> Vec<u8> {
    vec![0u8; (size as usize).saturating_mul(size as usize)]
}
