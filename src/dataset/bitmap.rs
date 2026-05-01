use tiny_skia::{FillRule, Mask, Path, PathBuilder, Transform};

use crate::pen::Command;

const BITMAP_SIZE: u32 = 64;
const PADDING: f32 = 4.0;

pub(super) fn render_bitmap(types: &[Command], coords: &[f32]) -> Vec<u8> {
    let Some(path) = build_path(types, coords) else {
        return blank_bitmap();
    };
    let bounds = path.compute_tight_bounds().unwrap_or_else(|| path.bounds());
    let width = bounds.width();
    let height = bounds.height();
    if width <= f32::EPSILON || height <= f32::EPSILON {
        return blank_bitmap();
    }

    let bitmap_size = BITMAP_SIZE as f32;
    let content_size = bitmap_size - PADDING.mul_add(2.0, 0.0);
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

    let mut mask = Mask::new(BITMAP_SIZE, BITMAP_SIZE).expect("bitmap size is valid");
    mask.fill_path(&path, FillRule::EvenOdd, true, transform);
    mask.data().to_vec()
}

fn blank_bitmap() -> Vec<u8> {
    vec![0; (BITMAP_SIZE * BITMAP_SIZE) as usize]
}

fn build_path(types: &[Command], coords: &[f32]) -> Option<Path> {
    let mut builder = PathBuilder::with_capacity(types.len(), coords.len() / 2);

    for (&command, values) in types.iter().zip(coords.chunks_exact(6)) {
        match command {
            Command::MoveTo => builder.move_to(values[4], values[5]),
            Command::LineTo => builder.line_to(values[4], values[5]),
            Command::QuadTo => builder.quad_to(values[0], values[1], values[4], values[5]),
            Command::CurveTo => builder.cubic_to(
                values[0], values[1], values[2], values[3], values[4], values[5],
            ),
            Command::Close => builder.close(),
            Command::End => break,
        }
    }

    builder.finish()
}
