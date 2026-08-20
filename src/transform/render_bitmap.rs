use tiny_skia::{FillRule, Mask, Path, PathBuilder, Transform};

use crate::outline::{BezPath, Bounds, PathEl, bounds_from_outline};

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
    outline: &BezPath,
    size: u32,
    mode: RenderMode,
    fill_rule: FillRule,
    antialias: bool,
) -> Result<RenderedBitmap, RenderBitmapError> {
    let path = build_path(outline);
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

    let data = draw_alpha_path(&path, width, height, transform, fill_rule, antialias);
    Ok(RenderedBitmap {
        data,
        width,
        height,
    })
}

fn build_path(outline: &BezPath) -> Option<Path> {
    let mut builder = PathBuilder::new();
    for element in outline.elements() {
        match *element {
            PathEl::MoveTo(point) => builder.move_to(point.x as f32, point.y as f32),
            PathEl::LineTo(point) => builder.line_to(point.x as f32, point.y as f32),
            PathEl::QuadTo(control, end) => builder.quad_to(
                control.x as f32,
                control.y as f32,
                end.x as f32,
                end.y as f32,
            ),
            PathEl::CurveTo(control0, control1, end) => builder.cubic_to(
                control0.x as f32,
                control0.y as f32,
                control1.x as f32,
                control1.y as f32,
                end.x as f32,
                end.y as f32,
            ),
            PathEl::ClosePath => builder.close(),
        }
    }
    builder.finish()
}

fn draw_alpha_path(
    path: &Path,
    width: u32,
    height: u32,
    transform: Transform,
    fill_rule: FillRule,
    antialias: bool,
) -> Vec<u8> {
    let mut mask = Mask::new(width, height).expect("width and height are nonzero");
    mask.fill_path(path, fill_rule, antialias, transform);
    mask.take()
}

fn render_target(
    bounds: Option<Bounds>,
    bitmap_size: f32,
    mode: RenderMode,
) -> Result<Option<(u32, u32, Transform)>, RenderBitmapError> {
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

fn render_transform(scale: f32, tx: f32, ty: f32) -> Transform {
    Transform::from_row(scale, 0.0, 0.0, -scale, tx, ty)
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
