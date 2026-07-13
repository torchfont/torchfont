use std::{collections::BTreeMap, path::Path};

use skrifa::{
    MetadataProvider,
    instance::{LocationRef, Size},
    outline::DrawSettings,
    raw::TableProvider,
};

use crate::{
    error::Error,
    font::{canonicalize_location, extract_glyph_outline, map_font, parse_font_ref},
    outline::Outline,
};

pub(crate) fn load_glyph_outline(
    path: &Path,
    ttc_index: u32,
    codepoint: u32,
    location: Option<&BTreeMap<String, f32>>,
) -> Result<Outline, Error> {
    let data = map_font(path)?;
    let font = parse_font_ref(&data[..], path, ttc_index)?;
    let units_per_em = font
        .head()
        .map_err(|err| {
            Error::Parse(format!(
                "font '{}' (ttc_index {ttc_index}) 'head' table error: {err}",
                path.display()
            ))
        })?
        .units_per_em();
    if units_per_em == 0 {
        return Err(Error::Parse(format!(
            "font '{}' (ttc_index {ttc_index}) has zero units per em",
            path.display()
        )));
    }
    let glyph_id = font.charmap().map(codepoint).ok_or_else(|| {
        Error::OutOfRange(format!(
            "codepoint U+{codepoint:04X} missing from '{}'",
            path.display()
        ))
    })?;
    let user_location = canonicalize_location(&font, path, ttc_index, location)?;
    let glyph = font.outline_glyphs().get(glyph_id).ok_or_else(|| {
        Error::Parse(format!(
            "glyph id {} missing from '{}'",
            glyph_id.to_u32(),
            path.display()
        ))
    })?;
    let location = font.axes().location(
        user_location
            .iter()
            .map(|(tag, value)| (tag.as_str(), *value)),
    );
    extract_glyph_outline(
        &glyph,
        DrawSettings::unhinted(Size::unscaled(), LocationRef::from(&location)),
        units_per_em as f32,
    )
    .map_err(|err| Error::Parse(format!("failed to draw glyph: {err}")))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::error::Error;

    use super::load_glyph_outline;

    fn test_font() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fonts/lato/Lato-Regular.ttf")
    }

    #[test]
    fn loads_outline_without_python() {
        let outline = load_glyph_outline(&test_font(), 0, 'A' as u32, None).unwrap();
        assert!(!outline.subpaths().is_empty());
    }

    #[test]
    fn reports_missing_codepoint_as_out_of_range() {
        let error = load_glyph_outline(&test_font(), 0, 0x10ffff, None).unwrap_err();
        assert!(matches!(error, Error::OutOfRange(_)));
    }
}
