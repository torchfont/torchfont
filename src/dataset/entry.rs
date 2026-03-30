use std::sync::Arc;

use memmap2::Mmap;
use pyo3::prelude::*;
use skrifa::raw::{FileRef, TableProvider};
use skrifa::{
    GlyphId, MetadataProvider,
    instance::{Location, LocationRef, Size},
    outline::DrawSettings,
};

use super::io::map_font;
use crate::{
    error::{py_err, py_index_err},
    pen::SegmentPen,
};

pub(super) struct FontEntry {
    data: Arc<Mmap>,
    face_index: u32,
    pub(super) path: String,
    pub(super) codepoints: Vec<u32>,
    glyph_ids: Vec<GlyphId>,
    units_per_em: f32,
    locations: Vec<Location>,
}

impl FontEntry {
    pub(super) fn load_faces(path: &str, filter: Option<&[u32]>) -> PyResult<Vec<Self>> {
        let mapped = map_font(path)?;
        let parsed = FileRef::new(&mapped[..])
            .map_err(|err| py_err(format!("failed to parse '{path}': {err}")))?;

        let entries = parsed
            .fonts()
            .enumerate()
            .map(|(face_index, face)| {
                let font = face.map_err(|err| {
                    py_err(format!(
                        "failed to parse '{path}' (face {face_index}): {err}",
                        face_index = face_index
                    ))
                })?;
                Self::from_face(path, Arc::clone(&mapped), face_index as u32, &font, filter)
            })
            .collect::<PyResult<Vec<_>>>()?;

        if entries.is_empty() {
            return Err(py_err(format!(
                "font file '{path}' does not contain any fonts"
            )));
        }
        Ok(entries)
    }

    pub(super) fn glyph(
        &self,
        codepoint: u32,
        instance_index: Option<usize>,
    ) -> PyResult<(Vec<i32>, Vec<f32>)> {
        let glyph_id = self.lookup_glyph(codepoint)?;
        let font = skrifa::FontRef::from_index(&self.data[..], self.face_index).map_err(|err| {
            py_err(format!(
                "failed to parse '{}' (face {}): {err}",
                self.path, self.face_index
            ))
        })?;
        let glyph = font.outline_glyphs().get(glyph_id).ok_or_else(|| {
            py_err(format!(
                "glyph id {} missing from '{}'",
                glyph_id.to_u32(),
                self.path
            ))
        })?;

        let mut pen = SegmentPen::new(self.units_per_em);
        glyph
            .draw(
                DrawSettings::unhinted(Size::unscaled(), self.location_ref(instance_index)?),
                &mut pen,
            )
            .map_err(|err| py_err(format!("failed to draw glyph: {err}")))?;

        Ok(pen.finish())
    }

    pub(super) fn instance_count(&self) -> usize {
        self.locations.len().max(1)
    }

    pub(super) fn is_variable(&self) -> bool {
        !self.locations.is_empty()
    }

    pub(super) fn named_instance_names(&self) -> Vec<Option<String>> {
        if !self.is_variable() {
            return vec![];
        }

        let font = match skrifa::FontRef::from_index(&self.data[..], self.face_index) {
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
        let font = match skrifa::FontRef::from_index(&self.data[..], self.face_index) {
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
        let font = skrifa::FontRef::from_index(&self.data[..], self.face_index).ok()?;

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

    fn from_face(
        base_path: &str,
        data: Arc<Mmap>,
        face_index: u32,
        font: &skrifa::FontRef<'_>,
        filter: Option<&[u32]>,
    ) -> PyResult<Self> {
        let upem = font
            .head()
            .map_err(|_| {
                py_err(format!(
                    "font '{base_path}' (face_index {face_index}) is missing a head table"
                ))
            })?
            .units_per_em();

        let outline_glyphs = font.outline_glyphs();
        let mut mappings: Vec<_> = font
            .charmap()
            .mappings()
            .filter(|(codepoint, _)| {
                filter.is_none_or(|values| values.binary_search(codepoint).is_ok())
            })
            .filter(|(_, glyph_id)| outline_glyphs.get(*glyph_id).is_some())
            .collect();
        mappings.sort_unstable_by_key(|entry| entry.0);
        let (codepoints, glyph_ids): (Vec<_>, Vec<_>) = mappings.into_iter().unzip();

        let locations = font
            .named_instances()
            .iter()
            .map(|inst| inst.location())
            .collect();

        Ok(Self {
            path: base_path.to_string(),
            data,
            face_index,
            codepoints,
            glyph_ids,
            units_per_em: upem as f32,
            locations,
        })
    }

    fn lookup_glyph(&self, codepoint: u32) -> PyResult<GlyphId> {
        self.codepoints
            .binary_search(&codepoint)
            .map(|idx| self.glyph_ids[idx])
            .map_err(|_| {
                py_index_err(format!(
                    "codepoint U+{codepoint:04X} missing from '{}'",
                    self.path
                ))
            })
    }

    fn location_ref(&self, index: Option<usize>) -> PyResult<LocationRef<'_>> {
        if let Some(idx) = index {
            let location = self.locations.get(idx).ok_or_else(|| {
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
