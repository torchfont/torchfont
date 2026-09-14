use skia_safe::{Path, PathBuilder, PathFillType};

use crate::outline::{BezPath, PathEl};

pub(super) fn build_skia_path(outline: &BezPath) -> Option<Path> {
    let mut builder = PathBuilder::new_with_fill_type(PathFillType::Winding);
    for &element in outline.elements() {
        push_skia_element(&mut builder, element);
    }
    (!builder.is_empty()).then(|| builder.detach())
}

pub(super) fn push_skia_element(builder: &mut PathBuilder, element: PathEl) {
    match element {
        PathEl::MoveTo(point) => {
            builder.move_to((point.x as f32, point.y as f32));
        }
        PathEl::LineTo(point) => {
            builder.line_to((point.x as f32, point.y as f32));
        }
        PathEl::QuadTo(control, end) => {
            builder.quad_to(
                (control.x as f32, control.y as f32),
                (end.x as f32, end.y as f32),
            );
        }
        PathEl::CurveTo(control0, control1, end) => {
            builder.cubic_to(
                (control0.x as f32, control0.y as f32),
                (control1.x as f32, control1.y as f32),
                (end.x as f32, end.y as f32),
            );
        }
        PathEl::ClosePath => {
            builder.close();
        }
    };
}
