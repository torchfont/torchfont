use std::sync::{Arc, RwLock};

use memmap2::Mmap;
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
    data: RwLock<Option<Arc<Mmap>>>,
    path: String,
    face_index: u32,
}

impl GlyphReader {
    pub(super) fn new(path: String, face_index: u32) -> Self {
        Self {
            data: RwLock::new(None),
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
        self.with_font_ref(|font| {
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
        let data = self.load_data()?;
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

    fn load_data(&self) -> PyResult<Arc<Mmap>> {
        if let Some(mapped) = self
            .data
            .read()
            .map_err(|_| py_err("glyph reader lock poisoned"))?
            .as_ref()
        {
            return Ok(Arc::clone(mapped));
        }

        let mut guard = self
            .data
            .write()
            .map_err(|_| py_err("glyph reader lock poisoned"))?;
        if let Some(mapped) = guard.as_ref() {
            return Ok(Arc::clone(mapped));
        }
        let mapped = map_font(&self.path)?;
        let result = Arc::clone(&mapped);
        *guard = Some(mapped);
        Ok(result)
    }
}
