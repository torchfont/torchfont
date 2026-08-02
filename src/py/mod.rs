pub(crate) mod dataset;
mod error;
pub(crate) mod glyphsets;
pub(crate) mod instances;
pub(crate) mod transform;

use pyo3::{Bound, PyResult, types::PyModule};

pub(crate) fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dataset::register(m)?;
    instances::register(m)?;
    glyphsets::register(m)?;
    transform::register(m)?;
    Ok(())
}
