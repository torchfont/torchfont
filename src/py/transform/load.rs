use numpy::{IntoPyArray as _, PyArray1};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::font::{axis_info, map_font, parse_font_ref};
use crate::transform::load::load_glyph_outline;

type GlyphOutlineArrays = (Py<PyArray1<i64>>, Py<PyArray1<f32>>);

#[pyfunction]
pub(crate) fn variation_axes(
    path: PathBuf,
    ttc_index: u32,
) -> PyResult<Vec<(String, f32, f32, f32)>> {
    let data = map_font(&path)?;
    let font = parse_font_ref(&data[..], &path, ttc_index)?;
    Ok(axis_info(&font)
        .into_iter()
        .map(|axis| (axis.tag, axis.min, axis.default, axis.max))
        .collect())
}

#[pyfunction]
pub(crate) fn load_glyph(
    py: Python<'_>,
    path: PathBuf,
    ttc_index: u32,
    codepoint: u32,
    location: Option<BTreeMap<String, f32>>,
) -> PyResult<GlyphOutlineArrays> {
    let (types, coords) =
        load_glyph_outline(&path, ttc_index, codepoint, location.as_ref())?.encode();
    Ok((
        types.into_pyarray(py).unbind(),
        coords.into_pyarray(py).unbind(),
    ))
}
