mod dataset;
mod error;
mod pen;

use dataset::{GlyphDataset, GlyphItem};
use pyo3::{Bound, prelude::*, types::PyModule};

#[pymodule]
fn _torchfont(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GlyphDataset>()?;
    m.add_class::<GlyphItem>()?;
    Ok(())
}
