mod dataset;
mod error;
mod pen;

use dataset::{GlyphDataset, GlyphItem};
use pyo3::{Bound, prelude::*, types::PyModule};

#[pyfunction]
fn render_bitmap(types_bytes: &[u8], coords_bytes: &[u8], size: u32) -> Vec<u8> {
    dataset::render_bitmap_from_bytes(types_bytes, coords_bytes, size)
}

#[pymodule]
fn _torchfont(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GlyphDataset>()?;
    m.add_class::<GlyphItem>()?;
    m.add_function(wrap_pyfunction!(render_bitmap, &m)?)?;
    Ok(())
}
