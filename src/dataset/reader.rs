use std::sync::Arc;

use memmap2::Mmap;
use pyo3::prelude::*;
use skrifa::{
    GlyphId, MetadataProvider,
    instance::{Location, LocationRef, Size},
    outline::DrawSettings,
};

use super::bitmap::render_bitmap;
use crate::{
    error::{py_err, py_index_err},
    pen::{Command, SegmentPen},
};

pub(super) struct GlyphItemData {
    pub(super) types: Vec<Command>,
    pub(super) coords: Vec<f32>,
    pub(super) bitmap: Vec<u8>,
    pub(super) advance_width: f32,
    pub(super) lsb: f32,
    pub(super) x_min: f32,
    pub(super) y_min: f32,
    pub(super) x_max: f32,
    pub(super) y_max: f32,
    pub(super) units_per_em: u16,
    pub(super) ascent: f32,
    pub(super) descent: f32,
    pub(super) leading: f32,
    pub(super) cap_height: f32,
    pub(super) x_height: f32,
    pub(super) average_width: f32,
    pub(super) is_monospace: bool,
    pub(super) italic_angle: f32,
    pub(super) glyph_name: String,
}

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

            let mut pen = SegmentPen::new(units_per_em);
            glyph
                .draw(
                    DrawSettings::unhinted(Size::unscaled(), location_ref),
                    &mut pen,
                )
                .map_err(|err| py_err(format!("failed to draw glyph: {err}")))?;
            let (types, coords) = pen.finish();
            let bitmap = render_bitmap(&types, &coords);

            let glyph_metrics = font.glyph_metrics(Size::unscaled(), location_ref);
            let advance_width = glyph_metrics
                .advance_width(glyph_id)
                .map(|v| v * inv_upem)
                .unwrap_or(f32::NAN);
            let lsb = glyph_metrics
                .left_side_bearing(glyph_id)
                .map(|v| v * inv_upem)
                .unwrap_or(f32::NAN);
            let (x_min, y_min, x_max, y_max) = glyph_metrics.bounds(glyph_id).map_or(
                (f32::NAN, f32::NAN, f32::NAN, f32::NAN),
                |bb| {
                    (
                        bb.x_min * inv_upem,
                        bb.y_min * inv_upem,
                        bb.x_max * inv_upem,
                        bb.y_max * inv_upem,
                    )
                },
            );

            let m = font.metrics(Size::unscaled(), location_ref);

            let glyph_name = font
                .glyph_names()
                .get(glyph_id)
                .map(|n| n.to_string())
                .unwrap_or_default();

            Ok(GlyphItemData {
                types,
                coords,
                bitmap,
                advance_width,
                lsb,
                x_min,
                y_min,
                x_max,
                y_max,
                units_per_em: m.units_per_em,
                ascent: m.ascent * inv_upem,
                descent: m.descent * inv_upem,
                leading: m.leading * inv_upem,
                cap_height: m.cap_height.map(|v| v * inv_upem).unwrap_or(f32::NAN),
                x_height: m.x_height.map(|v| v * inv_upem).unwrap_or(f32::NAN),
                average_width: m.average_width.map(|v| v * inv_upem).unwrap_or(f32::NAN),
                is_monospace: m.is_monospace,
                italic_angle: m.italic_angle,
                glyph_name,
            })
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
