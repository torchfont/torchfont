pub(crate) mod dataset;
pub(crate) mod transforms;

use pyo3::{Bound, PyResult, types::PyModule, types::PyModuleMethods};

pub(crate) fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<dataset::GlyphDatasetBackend>()?;
    m.add_class::<dataset::GlyphItem>()?;
    transforms::register(m)?;
    Ok(())
}
