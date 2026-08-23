mod index;

use pyo3::{
    Bound, PyResult,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(index::index_fonts, m)?)?;
    Ok(())
}
