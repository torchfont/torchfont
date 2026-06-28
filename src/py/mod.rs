pub(crate) mod dataset;
pub(crate) mod glyphsets;
pub(crate) mod transforms;

use pyo3::{Bound, PyResult, types::PyModule, types::PyModuleMethods};

pub(crate) fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<dataset::GlyphDatasetBackend>()?;
    m.add_class::<dataset::GlyphItem>()?;
    m.add_class::<dataset::DefaultInstantiation>()?;
    m.add_class::<dataset::NamedInstantiation>()?;
    m.add_class::<dataset::GridInstantiation>()?;
    glyphsets::register(m)?;
    transforms::register(m)?;
    Ok(())
}
