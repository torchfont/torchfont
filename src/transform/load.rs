use std::{collections::BTreeMap, path::Path};

use skrifa::{
    GlyphId, MetadataProvider,
    instance::{LocationRef, Size},
    outline::DrawSettings,
    raw::TableProvider,
};

use crate::{
    error::Error,
    font::{canonicalize_location, extract_glyph_outline, map_font, parse_font_ref},
    outline::BezPath,
};

pub(crate) fn load_glyph_outline(
    path: &Path,
    face_index: u32,
    glyph_id: u32,
    location: Option<&BTreeMap<String, f32>>,
) -> Result<BezPath, Error> {
    let data = map_font(path)?;
    let font = parse_font_ref(&data[..], path, face_index)?;
    let units_per_em = font
        .head()
        .map_err(|err| {
            Error::Parse(format!(
                "font '{}' (face_index {face_index}) 'head' table error: {err}",
                path.display()
            ))
        })?
        .units_per_em();
    if units_per_em == 0 {
        return Err(Error::Parse(format!(
            "font '{}' (face_index {face_index}) has zero units per em",
            path.display()
        )));
    }
    let user_location = canonicalize_location(&font, path, face_index, location)?;
    let glyph = font
        .outline_glyphs()
        .get(GlyphId::new(glyph_id))
        .ok_or_else(|| {
            Error::OutOfRange(format!(
                "glyph id {glyph_id} missing from '{}' (face_index {face_index})",
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
    use std::path::{Path, PathBuf};

    use skrifa::MetadataProvider as _;

    use crate::{
        error::Error,
        font::{map_font, parse_font_ref},
    };

    use super::load_glyph_outline;

    fn test_font() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fonts/source-sans/SourceSans3-Regular.ttf")
    }

    fn glyph_id_for(path: &Path, codepoint: u32) -> u32 {
        let data = map_font(path).unwrap();
        let font = parse_font_ref(&data[..], path, 0).unwrap();
        font.charmap().map(codepoint).unwrap().to_u32()
    }

    #[test]
    fn loads_outline_without_python() {
        let glyph_id = glyph_id_for(&test_font(), 'A' as u32);
        let outline = load_glyph_outline(&test_font(), 0, glyph_id, None).unwrap();
        assert!(outline.subpaths().next().is_some());
    }

    #[test]
    fn reports_missing_glyph_id_as_out_of_range() {
        let error = load_glyph_outline(&test_font(), 0, u32::MAX, None).unwrap_err();
        assert!(matches!(error, Error::OutOfRange(_)));
    }
}
