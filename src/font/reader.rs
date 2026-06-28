use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use memmap2::Mmap;
use skrifa::{
    GlyphId, MetadataProvider,
    instance::{LocationRef, Size},
    outline::DrawSettings,
    raw::TableProvider,
};

use crate::{
    error::Error,
    font::{extract_glyph_outline, glyph::Hmtx},
    geom::{Bounds, Outline, bounds_from_outline},
};

pub(super) struct GlyphReader {
    path: PathBuf,
    face_index: u32,
    data: Arc<Mmap>,
}

impl GlyphReader {
    pub(super) fn new(path: PathBuf, face_index: u32, data: Arc<Mmap>) -> Self {
        Self {
            path,
            face_index,
            data,
        }
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn face_index(&self) -> u32 {
        self.face_index
    }

    pub(super) fn draw_glyph(
        &self,
        glyph_id: GlyphId,
        units_per_em: u16,
        user_location: &[(String, f32)],
    ) -> Result<(Outline, Hmtx, Bounds, String), Error> {
        self.with_font_ref(|font| {
            let glyph = font.outline_glyphs().get(glyph_id).ok_or_else(|| {
                Error::Parse(format!(
                    "glyph id {} missing from '{}'",
                    glyph_id.to_u32(),
                    self.path.display()
                ))
            })?;

            let location = font.axes().location(
                user_location
                    .iter()
                    .map(|(tag, value)| (tag.as_str(), *value)),
            );
            let location_ref = LocationRef::from(&location);

            let outline = extract_glyph_outline(
                &glyph,
                DrawSettings::unhinted(Size::unscaled(), location_ref),
                units_per_em as f32,
            )
            .map_err(|err| Error::Parse(format!("failed to draw glyph: {err}")))?;

            let inv = (units_per_em as f32).recip();
            let glyph_metrics = font.glyph_metrics(Size::unscaled(), location_ref);
            let advance_width = glyph_metrics.advance_width(glyph_id).unwrap_or(f32::NAN) * inv;
            let lsb = glyph_metrics
                .left_side_bearing(glyph_id)
                .unwrap_or(f32::NAN)
                * inv;

            let nan_bounds = Bounds {
                x_min: f32::NAN,
                y_min: f32::NAN,
                x_max: f32::NAN,
                y_max: f32::NAN,
            };
            let bounds = if metrics_bounds_are_outline_based(&font) {
                bounds_from_outline(&outline).unwrap_or(nan_bounds)
            } else {
                glyph_metrics
                    .bounds(glyph_id)
                    .map_or(nan_bounds, |bb| Bounds {
                        x_min: bb.x_min * inv,
                        y_min: bb.y_min * inv,
                        x_max: bb.x_max * inv,
                        y_max: bb.y_max * inv,
                    })
            };

            let glyph_name = font
                .glyph_names()
                .get(glyph_id)
                .map(|n| n.to_string())
                .unwrap_or_default();

            Ok((outline, Hmtx { advance_width, lsb }, bounds, glyph_name))
        })
    }

    fn with_font_ref<T>(
        &self,
        f: impl FnOnce(skrifa::FontRef<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let path = &self.path;
        let face_index = self.face_index;
        let font = skrifa::FontRef::from_index(&self.data[..], face_index).map_err(|err| {
            Error::Parse(format!(
                "failed to parse '{}' (face {face_index}): {err}",
                path.display()
            ))
        })?;
        f(font)
    }
}

fn metrics_bounds_are_outline_based(font: &skrifa::FontRef<'_>) -> bool {
    font.gvar().is_ok() || font.cff().is_ok() || font.cff2().is_ok()
}
