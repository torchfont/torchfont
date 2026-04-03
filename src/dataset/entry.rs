use pyo3::prelude::*;
use skrifa::raw::{FileRef, TableProvider};
use skrifa::{GlyphId, MetadataProvider, instance::Location};

use super::{io::map_font, reader::GlyphReader};
use crate::error::{py_err, py_index_err};

pub(super) struct GlyphIndex {
    codepoints: Vec<u32>,
    glyph_ids: Vec<GlyphId>,
}

pub(super) struct FontEntry {
    index: GlyphIndex,
    reader: GlyphReader,
    units_per_em: f32,
    locations: Vec<Location>,
    style_axes: Vec<Vec<(String, f32)>>,
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
                Self::from_face(path, face_index as u32, &font, filter)
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
        self.reader
            .draw_glyph(glyph_id, self.units_per_em, &self.locations, instance_index)
    }

    pub(super) fn instance_count(&self) -> usize {
        self.style_axes.len()
    }

    pub(super) fn is_variable(&self) -> bool {
        !self.locations.is_empty()
    }

    pub(super) fn path(&self) -> &str {
        self.reader.path()
    }

    pub(super) fn face_index(&self) -> u32 {
        self.reader.face_index()
    }

    pub(super) fn codepoints(&self) -> &[u32] {
        &self.index.codepoints
    }

    pub(super) fn codepoint_count(&self) -> usize {
        self.index.codepoints.len()
    }

    pub(super) fn named_instance_names(&self) -> Vec<Option<String>> {
        if !self.is_variable() {
            return vec![];
        }
        self.reader.named_instance_names()
    }

    pub(super) fn style_axes(&self) -> &[Vec<(String, f32)>] {
        &self.style_axes
    }

    pub(super) fn family_name(&self) -> String {
        self.reader.family_name()
    }

    pub(super) fn subfamily_name(&self) -> Option<String> {
        self.reader.subfamily_name()
    }

    fn from_face(
        base_path: &str,
        face_index: u32,
        font: &skrifa::FontRef<'_>,
        filter: Option<&[u32]>,
    ) -> PyResult<Self> {
        let upem = font
            .head()
            .map_err(|_| {
                py_err(format!(
                    "font '{base_path}' (face {face_index}) is missing a head table"
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

        let axis_tags: Vec<String> = font
            .axes()
            .iter()
            .map(|axis| axis.tag().to_string())
            .collect();
        let named_instances = font.named_instances();
        let locations: Vec<Location> = named_instances.iter().map(|inst| inst.location()).collect();
        let style_axes = if locations.is_empty() {
            vec![vec![]]
        } else {
            named_instances
                .iter()
                .map(|inst| axis_tags.iter().cloned().zip(inst.user_coords()).collect())
                .collect()
        };

        Ok(Self {
            index: GlyphIndex {
                codepoints,
                glyph_ids,
            },
            reader: GlyphReader::new(base_path.to_string(), face_index),
            units_per_em: upem as f32,
            locations,
            style_axes,
        })
    }

    fn lookup_glyph(&self, codepoint: u32) -> PyResult<GlyphId> {
        self.index
            .codepoints
            .binary_search(&codepoint)
            .map(|idx| self.index.glyph_ids[idx])
            .map_err(|_| {
                py_index_err(format!(
                    "codepoint U+{codepoint:04X} missing from '{}'",
                    self.reader.path()
                ))
            })
    }
}
