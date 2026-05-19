use std::sync::Arc;

use memmap2::Mmap;
use pyo3::prelude::*;
use skrifa::{
    GlyphId, MetadataProvider,
    instance::{Location, LocationRef, Size},
    outline::DrawSettings,
    raw::TableProvider,
};

use crate::{
    error::{py_err, py_index_err},
    font::extract_glyph_outline,
    geom::bounds_from_outline,
};

use skrifa::raw::types::NameId;

pub(super) type GlyphItemData = (
    Vec<i64>,
    Vec<f32>,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    u16,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    bool,
    f32,
    String,
);

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
        units_per_em: f32,
        locations: &[Location],
        instance_index: Option<usize>,
    ) -> PyResult<GlyphItemData> {
        self.with_font_ref(|font| {
            let glyph = font.outline_glyphs().get(glyph_id).ok_or_else(|| {
                py_err(format!(
                    "glyph id {} missing from '{}'",
                    glyph_id.to_u32(),
                    self.path
                ))
            })?;

            let location_ref = self.location_ref(locations, instance_index)?;
            let inv_upem = 1.0 / units_per_em;
            let scale = |v: Option<f32>| v.map(|v| v * inv_upem).unwrap_or(f32::NAN);

            let outline = extract_glyph_outline(
                &glyph,
                DrawSettings::unhinted(Size::unscaled(), location_ref),
                units_per_em,
            )
            .map_err(|err| py_err(format!("failed to draw glyph: {err}")))?;

            let glyph_metrics = font.glyph_metrics(Size::unscaled(), location_ref);
            let advance_width = scale(glyph_metrics.advance_width(glyph_id));
            let lsb = scale(glyph_metrics.left_side_bearing(glyph_id));
            let nan4 = (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
            let (x_min, y_min, x_max, y_max) = if metrics_bounds_are_outline_based(&font) {
                bounds_from_outline(&outline)
                    .map_or(nan4, |bb| (bb.x_min, bb.y_min, bb.x_max, bb.y_max))
            } else {
                glyph_metrics.bounds(glyph_id).map_or(nan4, |bb| {
                    (
                        bb.x_min * inv_upem,
                        bb.y_min * inv_upem,
                        bb.x_max * inv_upem,
                        bb.y_max * inv_upem,
                    )
                })
            };

            let m = font.metrics(Size::unscaled(), location_ref);

            let glyph_name = font
                .glyph_names()
                .get(glyph_id)
                .map(|n| n.to_string())
                .unwrap_or_default();

            let (types, coords) = outline.encode();

            Ok((
                types,
                coords,
                advance_width,
                lsb,
                x_min,
                y_min,
                x_max,
                y_max,
                m.units_per_em,
                m.ascent * inv_upem,
                m.descent * inv_upem,
                m.leading * inv_upem,
                scale(m.cap_height),
                scale(m.x_height),
                scale(m.average_width),
                m.is_monospace,
                m.italic_angle,
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

    pub(super) fn family_name(&self) -> String {
        self.with_font_ref(|font| {
            Ok(localized_name(
                &font,
                [NameId::TYPOGRAPHIC_FAMILY_NAME, NameId::FAMILY_NAME],
            )
            .unwrap_or_default())
        })
        .unwrap_or_default()
    }

    pub(super) fn subfamily_name(&self) -> Option<String> {
        self.with_font_ref(|font| {
            Ok(localized_name(
                &font,
                [NameId::TYPOGRAPHIC_SUBFAMILY_NAME, NameId::SUBFAMILY_NAME],
            ))
        })
        .ok()
        .flatten()
    }

    fn with_font_ref<T>(&self, f: impl FnOnce(skrifa::FontRef<'_>) -> PyResult<T>) -> PyResult<T> {
        let font = skrifa::FontRef::from_index(&self.data[..], self.face_index).map_err(|err| {
            py_err(format!(
                "failed to parse '{}' (face {}): {err}",
                self.path, self.face_index
            ))
        })?;
        f(font)
    }

    fn location_ref<'a>(
        &self,
        locations: &'a [Location],
        index: Option<usize>,
    ) -> PyResult<LocationRef<'a>> {
        index.map_or(Ok(LocationRef::default()), |idx| {
            locations
                .get(idx)
                .ok_or_else(|| {
                    py_index_err(format!(
                        "instance index {idx} out of range for '{}'",
                        self.path
                    ))
                })
                .map(LocationRef::from)
        })
    }
}

fn localized_name(font: &skrifa::FontRef<'_>, ids: [NameId; 2]) -> Option<String> {
    ids.into_iter().find_map(|id| {
        font.localized_strings(id)
            .english_or_first()
            .map(|s| s.to_string())
    })
}

fn metrics_bounds_are_outline_based(font: &skrifa::FontRef<'_>) -> bool {
    font.gvar().is_ok() || font.cff().is_ok() || font.cff2().is_ok()
}
