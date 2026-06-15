use google_fonts_glyphsets::{GF_LATIN_CORE, GF_LATIN_KERNEL};
use pyo3::prelude::*;

/// Returns the codepoints of a glyphset as a list of integers.
///
/// If the glyphset is not found, raises a ValueError.
#[pyfunction]
fn get_glyphset_codepoints(glyphset_name: &str) -> PyResult<Vec<u32>> {
    google_fonts_glyphsets::GLYPHSETS
        .get(glyphset_name)
        .map(|glyphset| glyphset.iter_codepoints().collect())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Glyphset '{}' not found",
                glyphset_name
            ))
        })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "LATIN_CORE",
        GF_LATIN_CORE.iter_codepoints().collect::<Vec<_>>(),
    )?;
    m.add(
        "LATIN_KERNEL",
        GF_LATIN_KERNEL.iter_codepoints().collect::<Vec<_>>(),
    )?;
    m.add_function(wrap_pyfunction!(get_glyphset_codepoints, m)?)?;
    Ok(())
}
