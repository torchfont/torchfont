use std::{collections::BTreeMap, path::Path};

use kurbo::BezPath;
use skrifa::{
    GlyphId, MetadataProvider, Tag,
    attribute::Style,
    instance::{LocationRef, Size},
    outline::DrawSettings,
    raw::TableProvider,
    setting::VariationSetting,
};

use crate::{
    error::Error,
    font::{extract_glyph_outline, map_font_file, parse_font_ref},
};

const REGISTERED_AXIS_TAGS: [Tag; 5] = [
    Tag::new(b"wght"),
    Tag::new(b"wdth"),
    Tag::new(b"ital"),
    Tag::new(b"slnt"),
    Tag::new(b"opsz"),
];

type LoadedGlyph = (BezPath, Vec<VariationSetting>, [f32; 5]);

pub(crate) fn load_glyph(
    path: &Path,
    face_index: u32,
    glyph_id: u32,
    location: Option<&BTreeMap<String, f32>>,
) -> Result<LoadedGlyph, Error> {
    let data = map_font_file(path)?;
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
    let location = canonicalize_location(&font, location)?;
    let glyph = font
        .outline_glyphs()
        .get(GlyphId::new(glyph_id))
        .ok_or_else(|| {
            Error::OutOfRange(format!(
                "glyph id {glyph_id} missing from '{}' (face_index {face_index})",
                path.display()
            ))
        })?;
    let skrifa_location = font.axes().location(&location);
    let outline = extract_glyph_outline(
        &glyph,
        DrawSettings::unhinted(Size::unscaled(), LocationRef::from(&skrifa_location)),
        units_per_em as f32,
    )
    .map_err(|err| Error::Parse(format!("failed to draw glyph: {err}")))?;
    let values = registered_axis_values(&font, &location);
    Ok((outline, location, values))
}

fn canonicalize_location(
    font: &skrifa::FontRef<'_>,
    location: Option<&BTreeMap<String, f32>>,
) -> Result<Vec<VariationSetting>, Error> {
    for (tag, value) in location.into_iter().flatten() {
        if !value.is_finite() {
            return Err(Error::Parse(format!(
                "variation axis '{tag}' value must be finite"
            )));
        }
    }
    let axes = font.axes();
    let defaults = axes
        .iter()
        .map(|axis| VariationSetting::new(axis.tag(), axis.default_value()));
    let requested = location
        .into_iter()
        .flatten()
        .map(|(tag, &value)| (tag.as_str(), value).into());
    Ok(axes.filter(defaults.chain(requested)).collect())
}

fn registered_axis_values(font: &skrifa::FontRef<'_>, location: &[VariationSetting]) -> [f32; 5] {
    let attributes = font.attributes();
    let mut values = [
        attributes.weight.value(),
        attributes.stretch.percentage(),
        f32::from(matches!(attributes.style, Style::Italic)),
        font.post()
            .map_or(f32::NAN, |post| post.italic_angle().to_f64() as f32),
        f32::NAN,
    ];
    for setting in location {
        if let Some(index) = REGISTERED_AXIS_TAGS
            .iter()
            .position(|&tag| tag == setting.selector)
        {
            values[index] = setting.value;
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use skrifa::MetadataProvider as _;

    use crate::{
        error::Error,
        font::{map_font_file, parse_font_ref},
    };

    use super::{load_glyph, registered_axis_values};

    fn test_font() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fonts/source-sans/SourceSans3-Regular.ttf")
    }

    fn glyph_id_for(path: &Path, codepoint: u32) -> u32 {
        let data = map_font_file(path).unwrap();
        let font = parse_font_ref(&data[..], path, 0).unwrap();
        font.charmap().map(codepoint).unwrap().to_u32()
    }

    #[test]
    fn loads_outline_without_python() {
        let glyph_id = glyph_id_for(&test_font(), 'A' as u32);
        let (outline, _, _) = load_glyph(&test_font(), 0, glyph_id, None).unwrap();
        assert!(outline.subpaths().next().is_some());
    }

    #[test]
    fn reports_missing_glyph_id_as_out_of_range() {
        let Err(error) = load_glyph(&test_font(), 0, u32::MAX, None) else {
            panic!("missing glyph id must fail");
        };
        assert!(matches!(error, Error::OutOfRange(_)));
    }

    #[test]
    fn variation_location_overrides_font_attributes() {
        let path = test_font();
        let data = map_font_file(&path).unwrap();
        let font = parse_font_ref(&data[..], &path, 0).unwrap();
        let values = registered_axis_values(
            &font,
            &[
                ("wght", 700.0).into(),
                ("wdth", 75.0).into(),
                ("ital", 0.5).into(),
                ("slnt", -10.0).into(),
                ("opsz", 12.0).into(),
            ],
        );
        assert_eq!(values, [700.0, 75.0, 0.5, -10.0, 12.0]);
    }
}
