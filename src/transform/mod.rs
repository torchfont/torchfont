use skia_safe::{Path, PathBuilder, PathFillType};

use crate::bounds::{Bounds, BoundsPen};
use crate::outline::ElementType;

pub(crate) mod cubic_to_quad;
pub(crate) mod merge_curves;
pub(crate) mod quad_to_cubic;
pub(crate) mod remove_overlaps;
pub(crate) mod render_bitmap;
pub(crate) mod subpath;

pub(crate) fn build_skia_path(
    types: &[i64],
    coords: &[f32],
    track_bounds: bool,
) -> Option<(Path, Option<Bounds>)> {
    let mut builder = PathBuilder::new_with_fill_type(PathFillType::Winding);
    let mut bounds = track_bounds.then(BoundsPen::default);
    for (&element_type, values) in types.iter().zip(coords.chunks_exact(6)) {
        match element_type {
            v if v == ElementType::MoveTo as i64 => {
                if let Some(b) = &mut bounds {
                    b.move_to(values[4], values[5]);
                }
                builder.move_to((values[4], values[5]));
            }
            v if v == ElementType::LineTo as i64 => {
                if let Some(b) = &mut bounds {
                    b.line_to(values[4], values[5]);
                }
                builder.line_to((values[4], values[5]));
            }
            v if v == ElementType::QuadTo as i64 => {
                if let Some(b) = &mut bounds {
                    b.quad_to(values[0], values[1], values[4], values[5]);
                }
                builder.quad_to((values[0], values[1]), (values[4], values[5]));
            }
            v if v == ElementType::CurveTo as i64 => {
                if let Some(b) = &mut bounds {
                    b.curve_to(
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
                if let Some(b) = &mut bounds {
                    b.close();
                }
                builder.close();
            }
            _ => break,
        }
    }
    (!builder.is_empty()).then(|| (builder.detach(), bounds.and_then(BoundsPen::finish)))
}
