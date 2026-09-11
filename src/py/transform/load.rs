use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::font::{axis_info, map_font_file, parse_font_ref};
use crate::transform::load::load_glyph as load;

#[pyfunction]
pub(crate) fn variation_axes(
    py: Python<'_>,
    path: PathBuf,
    face_index: u32,
) -> PyResult<Vec<(String, f32, f32, f32)>> {
    py.detach(|| {
        let data = map_font_file(&path)?;
        let font = parse_font_ref(&data[..], &path, face_index)?;
        Ok(axis_info(&font)
            .into_iter()
            .map(|axis| (axis.tag, axis.min_value, axis.default_value, axis.max_value))
            .collect())
    })
}

type AxisValues = (f32, f32, f32, f32, f32);

type LoadedGlyphArrays<'py> = (super::OutlineArrays<'py>, Vec<(String, f32)>, AxisValues);

#[pyfunction]
pub(crate) fn load_glyph<'py>(
    py: Python<'py>,
    path: PathBuf,
    face_index: u32,
    glyph_id: u32,
    location: Option<BTreeMap<String, f32>>,
) -> PyResult<LoadedGlyphArrays<'py>> {
    let (outline, location, axis_values) =
        py.detach(|| load(&path, face_index, glyph_id, location.as_ref()))?;
    Ok((super::encode(py, &outline), location, axis_values))
}
