use std::path::{Path, PathBuf};

use skrifa::{MetadataProvider, raw::FileRef};

use crate::error::Error;
use crate::font::map_font;

/// One discovered face and the codepoint/glyph id pairs its `cmap` contributes.
pub(crate) struct DiscoveredCodepoints {
    path: PathBuf,
    ttc_index: u32,
    codepoints: Vec<u32>,
    glyph_ids: Vec<u32>,
}

/// One discovered face and every glyph id it draws an outline for.
pub(crate) struct DiscoveredGlyphs {
    path: PathBuf,
    ttc_index: u32,
    glyph_ids: Vec<u32>,
}

impl DiscoveredCodepoints {
    pub(crate) fn from_file(path: &Path, filter: Option<&[u32]>) -> Result<Vec<Self>, Error> {
        read_faces(path, |ttc_index, font| {
            Self::from_font(path, ttc_index, font, filter)
        })
    }

    pub(crate) fn into_parts(self) -> (PathBuf, u32, Vec<u32>, Vec<u32>) {
        (self.path, self.ttc_index, self.codepoints, self.glyph_ids)
    }

    pub(crate) fn codepoint_count(&self) -> usize {
        self.codepoints.len()
    }

    fn from_font(
        path: &Path,
        ttc_index: u32,
        font: &skrifa::FontRef<'_>,
        filter: Option<&[u32]>,
    ) -> Self {
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
        let (codepoints, glyph_ids) = mappings
            .into_iter()
            .map(|(codepoint, glyph_id)| (codepoint, glyph_id.to_u32()))
            .unzip();
        Self {
            path: path.to_path_buf(),
            ttc_index,
            codepoints,
            glyph_ids,
        }
    }
}

impl DiscoveredGlyphs {
    pub(crate) fn from_file(path: &Path) -> Result<Vec<Self>, Error> {
        read_faces(path, |ttc_index, font| {
            Self::from_font(path, ttc_index, font)
        })
    }

    pub(crate) fn into_parts(self) -> (PathBuf, u32, Vec<u32>) {
        (self.path, self.ttc_index, self.glyph_ids)
    }

    pub(crate) fn glyph_count(&self) -> usize {
        self.glyph_ids.len()
    }

    fn from_font(path: &Path, ttc_index: u32, font: &skrifa::FontRef<'_>) -> Self {
        let glyph_ids = font
            .outline_glyphs()
            .iter()
            .map(|(glyph_id, _)| glyph_id.to_u32())
            .collect();
        Self {
            path: path.to_path_buf(),
            ttc_index,
            glyph_ids,
        }
    }
}

/// Parses every face in one font file and builds one entry per face.
fn read_faces<T>(
    path: &Path,
    build: impl Fn(u32, &skrifa::FontRef<'_>) -> T,
) -> Result<Vec<T>, Error> {
    let mapped = map_font(path)?;
    let parsed = FileRef::new(&mapped[..])
        .map_err(|err| Error::Parse(format!("failed to parse '{}': {err}", path.display())))?;
    let entries = parsed
        .fonts()
        .enumerate()
        .map(|(ttc_index, font_result)| {
            let font = font_result.map_err(|err| {
                Error::Parse(format!(
                    "failed to parse '{}' (ttc_index {ttc_index}): {err}",
                    path.display()
                ))
            })?;
            Ok(build(ttc_index as u32, &font))
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
