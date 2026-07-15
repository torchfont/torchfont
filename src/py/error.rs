use crate::error::Error;

impl From<Error> for pyo3::PyErr {
    fn from(error: Error) -> Self {
        match error {
            Error::OutOfRange(_) => {
                pyo3::PyErr::new::<pyo3::exceptions::PyIndexError, _>(error.to_string())
            }
            Error::Parse(_) => {
                pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(error.to_string())
            }
            Error::Io(err) => err.into(),
        }
    }
}
