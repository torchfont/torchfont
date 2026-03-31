use std::sync::Arc;

use memmap2::Mmap;
use pyo3::prelude::*;
use skrifa::{
    GlyphId, MetadataProvider,
    instance::{Location, LocationRef, Size},
    outline::DrawSettings,
};

use crate::{
    error::{py_err, py_index_err},
    pen::SegmentPen,
};

pub(super) struct GlyphReader {
    data: Arc<Mmap>,
    path: String,
    face_index: u32,
}

impl GlyphReader {
    pub(super) fn new(data: Arc<Mmap>, path: String, face_index: u32) -> Self {
        Self {
            data,
            path,
            face_index,
        }
    }

    pub(super) fn path(&self) -> &str {
        &self.path
    }

    pub(super) fn draw_glyph(
        &self,
        glyph_id: GlyphId,
        units_per_em: f32,
        locations: &[Location],
        instance_index: Option<usize>,
    ) -> PyResult<(Vec<i32>, Vec<f32>)> {
        let font = self.font_ref()?;
        let glyph = font.outline_glyphs().get(glyph_id).ok_or_else(|| {
            py_err(format!(
                "glyph id {} missing from '{}'",
                glyph_id.to_u32(),
                self.path
            ))
        })?;

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

        Ok(pen.finish())
    }

    pub(super) fn named_instance_names(&self) -> Vec<Option<String>> {
        let font = match self.font_ref() {
            Ok(f) => f,
            Err(_) => return vec![],
        };

        font.named_instances()
            .iter()
            .map(|inst| {
                let name_id = inst.subfamily_name_id();
                font.localized_strings(name_id)
                    .english_or_first()
                    .map(|s| s.to_string())
            })
            .collect()
    }

    pub(super) fn family_name(&self) -> String {
        let font = match self.font_ref() {
            Ok(f) => f,
            Err(_) => return String::new(),
        };

        [
            skrifa::raw::types::NameId::TYPOGRAPHIC_FAMILY_NAME,
            skrifa::raw::types::NameId::FAMILY_NAME,
        ]
        .into_iter()
        .find_map(|id| {
            font.localized_strings(id)
                .english_or_first()
                .map(|s| s.to_string())
        })
        .unwrap_or_default()
    }

    pub(super) fn subfamily_name(&self) -> Option<String> {
        let font = self.font_ref().ok()?;

        [
            skrifa::raw::types::NameId::TYPOGRAPHIC_SUBFAMILY_NAME,
            skrifa::raw::types::NameId::SUBFAMILY_NAME,
        ]
        .into_iter()
        .find_map(|id| {
            font.localized_strings(id)
                .english_or_first()
                .map(|s| s.to_string())
        })
    }

    fn font_ref(&self) -> PyResult<skrifa::FontRef<'_>> {
        skrifa::FontRef::from_index(&self.data[..], self.face_index).map_err(|err| {
            py_err(format!(
                "failed to parse '{}' (face {}): {err}",
                self.path, self.face_index
            ))
        })
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
