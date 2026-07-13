use std::path::{Path, PathBuf};

use skrifa::{MetadataProvider, raw::FileRef, raw::types::NameId};

use crate::error::Error;
use crate::font::map_font;

struct Name {
    family_name: String,
    subfamily_name: String,
    typographic_family_name: String,
    typographic_subfamily_name: String,
}

pub(crate) struct DiscoveredFont {
    path: PathBuf,
    ttc_index: u32,
    codepoints: Vec<u32>,
    name: Name,
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

    pub(crate) fn family_name(&self) -> &str {
        if self.name.typographic_family_name.is_empty() {
            &self.name.family_name
        } else {
            &self.name.typographic_family_name
        }
    }

    pub(crate) fn subfamily_name(&self) -> &str {
        if self.name.typographic_subfamily_name.is_empty() {
            &self.name.subfamily_name
        } else {
            &self.name.typographic_subfamily_name
        }
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
            name: parse_name_table(font),
        }
    }
}

fn parse_name_table(font: &skrifa::FontRef<'_>) -> Name {
    let one = |id: NameId| {
        font.localized_strings(id)
            .english_or_first()
            .map(|value| value.to_string())
            .unwrap_or_default()
    };
    Name {
        family_name: one(NameId::FAMILY_NAME),
        subfamily_name: one(NameId::SUBFAMILY_NAME),
        typographic_family_name: one(NameId::TYPOGRAPHIC_FAMILY_NAME),
        typographic_subfamily_name: one(NameId::TYPOGRAPHIC_SUBFAMILY_NAME),
    }
}
