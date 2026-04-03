use pyo3::prelude::*;
use skrifa::{
    GlyphId, MetadataProvider,
    instance::{Location, LocationRef, Size},
    outline::DrawSettings,
};

use crate::{
    dataset::io::map_font,
    error::{py_err, py_index_err},
    pen::SegmentPen,
};

pub(super) struct GlyphReader {
    path: String,
    face_index: u32,
}

impl GlyphReader {
    pub(super) fn new(path: String, face_index: u32) -> Self {
        Self { path, face_index }
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
    ) -> PyResult<(Vec<i32>, Vec<f32>, Vec<f32>)> {
        self.with_font_ref(|font| {
            let glyph = font.outline_glyphs().get(glyph_id).ok_or_else(|| {
                py_err(format!(
                    "glyph id {} missing from '{}'",
                    glyph_id.to_u32(),
                    self.path
                ))
            })?;

            let scale = units_per_em.recip();
            let glyph_metrics = font.glyph_metrics(
                Size::unscaled(),
                self.location_ref(locations, instance_index)?,
            );
            let advance_width = glyph_metrics.advance_width(glyph_id).unwrap_or(0.0) * scale;
            let lsb = glyph_metrics.left_side_bearing(glyph_id).unwrap_or(0.0) * scale;
            let bbox = glyph_metrics.bounds(glyph_id).unwrap_or_default();

            let mut pen = SegmentPen::new(units_per_em);
            glyph
                .draw(
                    DrawSettings::unhinted(
                        Size::unscaled(),
                        self.location_ref(locations, instance_index)?,
                    ),
                    &mut pen,
                )
                .map_err(|err| py_err(format!("failed to draw glyph: {err}")))?;

            let (types, coords) = pen.finish();
            let metrics = vec![
                advance_width,
                lsb,
                bbox.x_min * scale,
                bbox.y_min * scale,
                bbox.x_max * scale,
                bbox.y_max * scale,
            ];
            Ok((types, coords, metrics))
        })
    }

    pub(super) fn named_instance_names(&self) -> Vec<Option<String>> {
        self.with_font_ref(|font| {
            Ok(font
                .named_instances()
                .iter()
                .map(|inst| {
                    let name_id = inst.subfamily_name_id();
                    font.localized_strings(name_id)
                        .english_or_first()
                        .map(|s| s.to_string())
                })
                .collect())
        })
        .unwrap_or_default()
    }

    pub(super) fn family_name(&self) -> String {
        self.with_font_ref(|font| {
            Ok([
                skrifa::raw::types::NameId::TYPOGRAPHIC_FAMILY_NAME,
                skrifa::raw::types::NameId::FAMILY_NAME,
            ]
            .into_iter()
            .find_map(|id| {
                font.localized_strings(id)
                    .english_or_first()
                    .map(|s| s.to_string())
            })
            .unwrap_or_default())
        })
        .unwrap_or_default()
    }

    pub(super) fn subfamily_name(&self) -> Option<String> {
        self.with_font_ref(|font| {
            Ok([
                skrifa::raw::types::NameId::TYPOGRAPHIC_SUBFAMILY_NAME,
                skrifa::raw::types::NameId::SUBFAMILY_NAME,
            ]
            .into_iter()
            .find_map(|id| {
                font.localized_strings(id)
                    .english_or_first()
                    .map(|s| s.to_string())
            }))
        })
        .ok()
        .flatten()
    }

    fn with_font_ref<T>(&self, f: impl FnOnce(skrifa::FontRef<'_>) -> PyResult<T>) -> PyResult<T> {
        let data = map_font(&self.path)?;
        let font = skrifa::FontRef::from_index(&data[..], self.face_index).map_err(|err| {
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
        if let Some(idx) = index {
            let location = locations.get(idx).ok_or_else(|| {
                py_index_err(format!(
                    "instance index {idx} out of range for '{}'",
                    self.path
                ))
            })?;
            Ok(LocationRef::from(location))
        } else {
            Ok(LocationRef::default())
        }
    }
}
