pub(crate) mod dataset;
mod error;
pub(crate) mod glyphsets;
pub(crate) mod instance_fn;
pub(crate) mod transform;

use pyo3::{Bound, PyResult, types::PyModule};

pub(crate) fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dataset::register(m)?;
    instance_fn::register(m)?;
    glyphsets::register(m)?;
    transform::register(m)?;
    Ok(())
}
