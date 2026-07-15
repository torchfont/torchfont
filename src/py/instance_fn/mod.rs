pub(crate) mod callback;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pyo3::{
    Bound, PyResult, pyfunction,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};

use crate::font::{default_location, map_font, parse_font_ref};
use crate::instance_fn::{grid_location_count, grid_locations, named_locations};

#[pyfunction]
fn default_location_for_font(path: PathBuf, ttc_index: u32) -> PyResult<Vec<(String, f32)>> {
    with_font_ref(&path, ttc_index, |font| Ok(default_location(&font)))
}

#[pyfunction]
fn named_instance_locations_for_font(
    path: PathBuf,
    ttc_index: u32,
) -> PyResult<Vec<Vec<(String, f32)>>> {
    with_font_ref(&path, ttc_index, |font| {
        named_locations(&font, &path, ttc_index).map_err(Into::into)
    })
}

#[pyfunction]
fn grid_locations_for_font(
    path: PathBuf,
    ttc_index: u32,
    axes: BTreeMap<String, i64>,
) -> PyResult<Vec<Vec<(String, f32)>>> {
    with_font_ref(&path, ttc_index, |font| {
        grid_locations(&font, &axes).map_err(Into::into)
    })
}

#[pyfunction]
fn grid_location_count_for_font(
    path: PathBuf,
    ttc_index: u32,
    axes: BTreeMap<String, i64>,
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(default_location_for_font, m)?)?;
    m.add_function(wrap_pyfunction!(named_instance_locations_for_font, m)?)?;
    m.add_function(wrap_pyfunction!(grid_locations_for_font, m)?)?;
    m.add_function(wrap_pyfunction!(grid_location_count_for_font, m)?)?;
    Ok(())
}
