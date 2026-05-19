use pyo3::prelude::*;

pub fn py_err(msg: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(msg.into())
}

pub fn py_index_err(msg: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIndexError, _>(msg.into())
}
