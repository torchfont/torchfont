use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::font::{
    axis_info, canonicalize_location, map_font, parse_font_ref, registered_axis_values,
};
use crate::transform::load::load_glyph_outline;

#[pyfunction]
pub(crate) fn variation_axes(
    py: Python<'_>,
    path: PathBuf,
    face_index: u32,
) -> PyResult<Vec<(String, f32, f32, f32)>> {
    py.detach(|| {
        let data = map_font(&path)?;
        let font = parse_font_ref(&data[..], &path, face_index)?;
        Ok(axis_info(&font)
            .into_iter()
            .map(|axis| (axis.tag, axis.min, axis.default, axis.max))
            .collect())
    })
}

type GlyphTargets = (f32, f32, f32, f32, f32);

#[pyfunction]
pub(crate) fn glyph_targets(
    py: Python<'_>,
    path: PathBuf,
    face_index: u32,
    location: BTreeMap<String, f32>,
) -> PyResult<GlyphTargets> {
    py.detach(|| {
        let data = map_font(&path)?;
        let font = parse_font_ref(&data[..], &path, face_index)?;
        let location = canonicalize_location(&font, &path, face_index, Some(&location))?;
        let values = registered_axis_values(&font, &location);
        Ok((
            values.weight,
            values.width,
            values.italic,
            values.slant,
            values.optical_size,
        ))
    })
}

#[pyfunction]
pub(crate) fn load_glyph<'py>(
    py: Python<'py>,
    path: PathBuf,
    face_index: u32,
    glyph_id: u32,
    location: Option<BTreeMap<String, f32>>,
) -> PyResult<super::OutlineArrays<'py>> {
    let outline =
        py.detach(|| load_glyph_outline(&path, face_index, glyph_id, location.as_ref()))?;
    Ok(super::encode(py, &outline))
}
