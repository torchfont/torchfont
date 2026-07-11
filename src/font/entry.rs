use std::{fs, path::Path, path::PathBuf};

use memmap2::Mmap;
use skrifa::MetadataProvider;
use skrifa::raw::FileRef;
use skrifa::raw::types::NameId;

use crate::error::Error;

/// Font-level strings used for dataset style display names.
pub(crate) struct Name {
    pub(crate) family_name: String,
    pub(crate) subfamily_name: String,
    pub(crate) typographic_family_name: String,
    pub(crate) typographic_subfamily_name: String,
}

pub(crate) struct FontEntry {
    path: PathBuf,
    ttc_index: u32,
    codepoints: Vec<u32>,
    pub(crate) name: Name,
}

impl FontEntry {
    pub(crate) fn load_fonts(path: &Path, filter: Option<&[u32]>) -> Result<Vec<Self>, Error> {
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
                Self::from_font(path, ttc_index as u32, &font, filter)
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
        base_path: &Path,
        ttc_index: u32,
        font: &skrifa::FontRef<'_>,
        filter: Option<&[u32]>,
    ) -> Result<Self, Error> {
        let name = parse_name_table(font);

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
        let codepoints: Vec<_> = mappings
            .into_iter()
            .map(|(codepoint, _glyph_id)| codepoint)
            .collect();

        Ok(Self {
            path: base_path.to_path_buf(),
            ttc_index,
            codepoints,
            name,
        })
    }
}

fn parse_name_table(font: &skrifa::FontRef<'_>) -> Name {
    let one = |id: NameId| -> String {
        font.localized_strings(id)
            .english_or_first()
            .map(|s| s.to_string())
            .unwrap_or_default()
    };
    Name {
        family_name: one(NameId::FAMILY_NAME),
        subfamily_name: one(NameId::SUBFAMILY_NAME),
        typographic_family_name: one(NameId::TYPOGRAPHIC_FAMILY_NAME),
        typographic_subfamily_name: one(NameId::TYPOGRAPHIC_SUBFAMILY_NAME),
    }
}

pub(crate) fn map_font(path: &Path) -> Result<Mmap, Error> {
    let file = fs::File::open(path)
        .map_err(|err| Error::Io(format!("failed to open '{}': {err}", path.display())))?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|err| Error::Io(format!("failed to map '{}': {err}", path.display())))?;
    Ok(mmap)
}

pub(crate) fn parse_font_ref<'a>(
    data: &'a [u8],
    path: &Path,
    ttc_index: u32,
) -> Result<skrifa::FontRef<'a>, Error> {
    skrifa::FontRef::from_index(data, ttc_index).map_err(|err| {
        Error::Parse(format!(
            "failed to parse '{}' (ttc_index {ttc_index}): {err}",
            path.display()
        ))
    })
}
