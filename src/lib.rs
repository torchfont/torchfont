mod dataset;
mod error;
mod font;
mod instance_fn;
mod outline;
mod py;
mod transform;

use pyo3::{Bound, prelude::*, types::PyModule};

#[pymodule]
fn _torchfont(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    py::register_module(&m)
}
