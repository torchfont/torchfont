use numpy::{IntoPyArray as _, PyArray1};
use pyo3::prelude::*;
use skrifa::MetadataProvider;
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::DrawSettings;
use skrifa::raw::TableProvider;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::font::{
    axis_info, canonicalize_location, default_location, extract_glyph_outline, grid_location_count,
    grid_locations, map_font, named_locations, parse_font_ref,
};

type GlyphOutlineArrays = (Py<PyArray1<i64>>, Py<PyArray1<f32>>);

#[pyfunction]
pub(crate) fn load_glyph(
    py: Python<'_>,
    path: PathBuf,
    ttc_index: u32,
    codepoint: u32,
    location: Option<HashMap<String, f32>>,
) -> PyResult<GlyphOutlineArrays> {
    let data = map_font(&path)?;
    let font = parse_font_ref(&data[..], &path, ttc_index)?;
    let units_per_em = font
        .head()
        .map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "font '{}' (ttc_index {ttc_index}) 'head' table error: {err}",
                path.display()
            ))
        })?
        .units_per_em();
    let glyph_id = font.charmap().map(codepoint).ok_or_else(|| {
        pyo3::exceptions::PyIndexError::new_err(format!(
            "codepoint U+{codepoint:04X} missing from '{}'",
            path.display()
        ))
    })?;
    let user_location = canonicalize_location(&font, &path, ttc_index, location.as_ref())?;
    let glyph = font.outline_glyphs().get(glyph_id).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
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
    let location_ref = LocationRef::from(&location);
    let outline = extract_glyph_outline(
        &glyph,
        DrawSettings::unhinted(Size::unscaled(), location_ref),
        units_per_em as f32,
    )
    .map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("failed to draw glyph: {err}"))
    })?;

    let (types, coords) = outline.encode();
    Ok((
        types.into_pyarray(py).unbind(),
        coords.into_pyarray(py).unbind(),
    ))
}

#[pyfunction]
pub(crate) fn variation_axes(
    path: PathBuf,
    ttc_index: u32,
) -> PyResult<Vec<(String, f32, f32, f32)>> {
    with_font_ref(&path, ttc_index, |font| {
        Ok(axis_info(&font)
            .into_iter()
            .map(|axis| (axis.tag, axis.min, axis.default, axis.max))
            .collect())
    })
}

#[pyfunction]
pub(crate) fn default_location_for_font(
    path: PathBuf,
    ttc_index: u32,
) -> PyResult<Vec<(String, f32)>> {
    with_font_ref(&path, ttc_index, |font| Ok(default_location(&font)))
}

#[pyfunction]
pub(crate) fn named_instance_locations_for_font(
    path: PathBuf,
    ttc_index: u32,
) -> PyResult<Vec<Vec<(String, f32)>>> {
    with_font_ref(&path, ttc_index, |font| {
        named_locations(&font, &path, ttc_index).map_err(Into::into)
    })
}

#[pyfunction]
pub(crate) fn grid_locations_for_font(
    path: PathBuf,
    ttc_index: u32,
    axes: HashMap<String, i64>,
) -> PyResult<Vec<Vec<(String, f32)>>> {
    with_font_ref(&path, ttc_index, |font| {
        grid_locations(&font, &axes).map_err(Into::into)
    })
}

#[pyfunction]
pub(crate) fn grid_location_count_for_font(
    path: PathBuf,
    ttc_index: u32,
    axes: HashMap<String, i64>,
) -> PyResult<usize> {
    with_font_ref(&path, ttc_index, |font| {
        grid_location_count(&font, &axes).map_err(Into::into)
    })
}

fn with_font_ref<T>(
    path: &Path,
    ttc_index: u32,
    f: impl FnOnce(skrifa::FontRef<'_>) -> PyResult<T>,
) -> PyResult<T> {
    let data = map_font(path)?;
    let font = parse_font_ref(&data[..], path, ttc_index)?;
    f(font)
}
