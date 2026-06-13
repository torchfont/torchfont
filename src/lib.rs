mod curves;
mod dataset;
mod error;
mod font;
mod geom;
mod py;
mod transform;

use pyo3::{Bound, prelude::*, types::PyModule};

#[pymodule]
fn _torchfont(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    py::register_module(&m)
}
