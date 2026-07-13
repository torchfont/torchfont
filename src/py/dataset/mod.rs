mod build;
mod fixed;
mod variable;

use pyo3::{
    Bound, PyErr, PyResult,
    types::{PyModule, PyModuleMethods},
};

use crate::dataset::IndexOverflow;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<fixed::FixedGlyphIndex>()?;
    m.add_class::<variable::VariableGlyphIndex>()?;
    Ok(())
}

fn index_error(idx: usize, len: usize) -> PyErr {
    pyo3::exceptions::PyIndexError::new_err(format!("sample index {idx} out of range (len={len})"))
}

fn overflow_error(kind: IndexOverflow) -> PyErr {
    let message = match kind {
        IndexOverflow::SampleCount => "dataset sample count overflowed usize",
        IndexOverflow::StyleCount => "dataset style count overflowed usize",
    };
    pyo3::exceptions::PyOverflowError::new_err(message)
}
