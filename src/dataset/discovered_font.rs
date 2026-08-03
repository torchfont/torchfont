use std::path::{Path, PathBuf};

use skrifa::{MetadataProvider, raw::FileRef};

use crate::error::Error;
use crate::font::map_font;

pub(crate) struct DiscoveredFont {
    path: PathBuf,
    ttc_index: u32,
    codepoints: Vec<u32>,
}

impl DiscoveredFont {
    pub(crate) fn from_file(path: &Path, filter: Option<&[u32]>) -> Result<Vec<Self>, Error> {
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
                Ok(Self::from_font(path, ttc_index as u32, &font, filter))
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

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn ttc_index(&self) -> u32 {
        self.ttc_index
    }

    pub(crate) fn codepoints(&self) -> &[u32] {
        &self.codepoints
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
        Self {
            path: path.to_path_buf(),
            ttc_index,
            codepoints: mappings
                .into_iter()
                .map(|(codepoint, _)| codepoint)
                .collect(),
        }
    }
}
