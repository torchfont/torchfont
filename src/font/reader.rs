use std::sync::Arc;

use memmap2::Mmap;
use skrifa::{
    GlyphId, MetadataProvider,
    instance::{Location, LocationRef, Size},
    outline::DrawSettings,
    raw::TableProvider,
};

use crate::{
    error::Error,
    font::{
        extract_glyph_outline,
        glyph::{Bounds, Hmtx},
    },
    geom::{Outline, bounds_from_outline},
};

pub(super) struct GlyphReader {
    path: String,
    face_index: u32,
    data: Arc<Mmap>,
}

impl GlyphReader {
    pub(super) fn new(path: String, face_index: u32, data: Arc<Mmap>) -> Self {
        Self {
            path,
            face_index,
            data,
        }
    }

    pub(super) fn path(&self) -> &str {
        &self.path
    }

    pub(super) fn face_index(&self) -> u32 {
        self.face_index
    }

    pub(super) fn draw_glyph(
        &self,
        glyph_id: GlyphId,
        units_per_em: u16,
        locations: &[Location],
        instance_index: Option<usize>,
    ) -> Result<(Outline, Hmtx, Bounds, String), Error> {
        self.with_font_ref(|font| {
            let glyph = font.outline_glyphs().get(glyph_id).ok_or_else(|| {
                Error::Parse(format!(
                    "glyph id {} missing from '{}'",
                    glyph_id.to_u32(),
                    self.path
                ))
            })?;

            let location_ref = self.location_ref(locations, instance_index)?;

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

            let nan4 = (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
            let (x_min, y_min, x_max, y_max) = if metrics_bounds_are_outline_based(&font) {
                bounds_from_outline(&outline)
                    .map_or(nan4, |bb| (bb.x_min, bb.y_min, bb.x_max, bb.y_max))
            } else {
                glyph_metrics.bounds(glyph_id).map_or(nan4, |bb| {
                    (
                        bb.x_min * inv,
                        bb.y_min * inv,
                        bb.x_max * inv,
                        bb.y_max * inv,
                    )
                })
            };

            let glyph_name = font
                .glyph_names()
                .get(glyph_id)
                .map(|n| n.to_string())
                .unwrap_or_default();

            Ok((
                outline,
                Hmtx { advance_width, lsb },
                Bounds {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                },
                glyph_name,
            ))
        })
    }

    pub(super) fn named_instance_names(&self) -> Vec<Option<String>> {
        self.with_font_ref(|font| {
            Ok(font
                .named_instances()
                .iter()
                .map(|inst| {
                    font.localized_strings(inst.subfamily_name_id())
                        .english_or_first()
                        .map(|s| s.to_string())
                })
                .collect())
        })
        .unwrap_or_default()
    }

    fn with_font_ref<T>(
        &self,
        f: impl FnOnce(skrifa::FontRef<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let path = &self.path;
        let face_index = self.face_index;
        let font = skrifa::FontRef::from_index(&self.data[..], face_index).map_err(|err| {
            Error::Parse(format!(
                "failed to parse '{path}' (face {face_index}): {err}"
            ))
        })?;
        f(font)
    }

    fn location_ref<'a>(
        &self,
        locations: &'a [Location],
        index: Option<usize>,
    ) -> Result<LocationRef<'a>, Error> {
        index.map_or(Ok(LocationRef::default()), |idx| {
            locations
                .get(idx)
                .ok_or_else(|| {
                    Error::OutOfRange(format!(
                        "instance index {idx} out of range for '{}'",
                        self.path
                    ))
                })
                .map(LocationRef::from)
        })
    }
}

fn metrics_bounds_are_outline_based(font: &skrifa::FontRef<'_>) -> bool {
    font.gvar().is_ok() || font.cff().is_ok() || font.cff2().is_ok()
}
