pub(crate) mod callable;
pub(crate) mod dataset;
pub(crate) mod glyphsets;
pub(crate) mod targets;
pub(crate) mod transforms;

use pyo3::{Bound, PyResult, types::PyModule, types::PyModuleMethods, wrap_pyfunction};

pub(crate) fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dataset::load_glyph, m)?)?;
    m.add_function(wrap_pyfunction!(dataset::variation_axes, m)?)?;
    m.add_function(wrap_pyfunction!(dataset::default_location_for_font, m)?)?;
    m.add_function(wrap_pyfunction!(
        dataset::named_instance_locations_for_font,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(dataset::grid_locations_for_font, m)?)?;
    m.add_function(wrap_pyfunction!(dataset::grid_location_count_for_font, m)?)?;
    glyphsets::register(m)?;
    targets::register(m)?;
    transforms::register(m)?;
    Ok(())
}
