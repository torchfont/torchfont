use std::path::{Path, PathBuf};

use skrifa::{
    GlyphId, MetadataProvider,
    instance::{LocationRef, Size},
    outline::{DrawSettings, OutlineGlyph},
    raw::FileRef,
};

use crate::error::Error;
use crate::font::{count_glyph_elements, map_font_file};

/// One discovered face and the codepoint/glyph id pairs its `cmap` contributes.
pub(crate) struct DiscoveredCodepoints {
    path: PathBuf,
    face_index: u32,
    codepoints: Vec<u32>,
    glyph_ids: Vec<u32>,
}

/// One discovered face and every glyph id it draws an outline for.
pub(crate) struct DiscoveredGlyphs {
    path: PathBuf,
    face_index: u32,
    glyph_ids: Vec<u32>,
}

impl DiscoveredCodepoints {
    pub(crate) fn from_file(
        path: &Path,
        filter: Option<&[u32]>,
        max_length: Option<usize>,
    ) -> Result<Vec<Self>, Error> {
        read_faces(path, |face_index, font| {
            Self::from_font(path, face_index, font, filter, max_length)
        })
    }

    pub(crate) fn into_parts(self) -> (PathBuf, u32, Vec<u32>, Vec<u32>) {
        (self.path, self.face_index, self.codepoints, self.glyph_ids)
    }

    pub(crate) fn codepoint_count(&self) -> usize {
        self.codepoints.len()
    }

    fn from_font(
        path: &Path,
        face_index: u32,
        font: &skrifa::FontRef<'_>,
        filter: Option<&[u32]>,
        max_length: Option<usize>,
    ) -> Result<Self, Error> {
        let outline_glyphs = font.outline_glyphs();
        let mut mappings = Vec::new();
        for (codepoint, glyph_id) in font.charmap().mappings() {
            if filter.is_some_and(|values| values.binary_search(&codepoint).is_err()) {
                continue;
            }
            let Some(glyph) = outline_glyphs.get(glyph_id) else {
                continue;
            };
            if let Some(max_length) = max_length
                && !fits_max_length(path, face_index, glyph_id, &glyph, max_length)?
            {
                continue;
            }
            mappings.push((codepoint, glyph_id));
        }
        mappings.sort_unstable_by_key(|entry| entry.0);
        let (codepoints, glyph_ids) = mappings
            .into_iter()
            .map(|(codepoint, glyph_id)| (codepoint, glyph_id.to_u32()))
            .unzip();
        Ok(Self {
            path: path.to_path_buf(),
            face_index,
            codepoints,
            glyph_ids,
        })
    }
}

impl DiscoveredGlyphs {
    pub(crate) fn from_file(path: &Path, max_length: Option<usize>) -> Result<Vec<Self>, Error> {
        read_faces(path, |face_index, font| {
            Self::from_font(path, face_index, font, max_length)
        })
    }

    pub(crate) fn into_parts(self) -> (PathBuf, u32, Vec<u32>) {
        (self.path, self.face_index, self.glyph_ids)
    }

    pub(crate) fn glyph_count(&self) -> usize {
        self.glyph_ids.len()
    }

    fn from_font(
        path: &Path,
        face_index: u32,
        font: &skrifa::FontRef<'_>,
        max_length: Option<usize>,
    ) -> Result<Self, Error> {
        let mut glyph_ids = Vec::new();
        for (glyph_id, glyph) in font.outline_glyphs().iter() {
            if let Some(max_length) = max_length
                && !fits_max_length(path, face_index, glyph_id, &glyph, max_length)?
            {
                continue;
            }
            glyph_ids.push(glyph_id.to_u32());
        }
        Ok(Self {
            path: path.to_path_buf(),
            face_index,
            glyph_ids,
        })
    }
}

/// Whether the encoded sequence of one glyph fits within `max_length`.
///
/// The elements are counted at the face default location, where variations
/// move points without changing how many elements a glyph draws.
fn fits_max_length(
    path: &Path,
    face_index: u32,
    glyph_id: GlyphId,
    glyph: &OutlineGlyph<'_>,
    max_length: usize,
) -> Result<bool, Error> {
    let length = count_glyph_elements(
        glyph,
        DrawSettings::unhinted(Size::unscaled(), LocationRef::default()),
    )
    .map_err(|err| {
        Error::Parse(format!(
            "failed to draw glyph id {} of '{}' (face_index {face_index}): {err}",
            glyph_id.to_u32(),
            path.display()
        ))
    })?;
    Ok(length <= max_length)
}

/// Parses every face in one font file and builds one entry per face.
fn read_faces<T>(
    path: &Path,
    build: impl Fn(u32, &skrifa::FontRef<'_>) -> Result<T, Error>,
) -> Result<Vec<T>, Error> {
    let mapped = map_font_file(path)?;
    let parsed = FileRef::new(&mapped[..])
        .map_err(|err| Error::Parse(format!("failed to parse '{}': {err}", path.display())))?;
    let entries = parsed
        .fonts()
        .enumerate()
        .map(|(face_index, font_result)| {
            let font = font_result.map_err(|err| {
                Error::Parse(format!(
                    "failed to parse '{}' (face_index {face_index}): {err}",
                    path.display()
                ))
            })?;
            build(face_index as u32, &font)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if entries.is_empty() {
        return Err(Error::Parse(format!(
            "font file '{}' does not contain any fonts",
            path.display()
        )));
    }
    Ok(entries)
}
