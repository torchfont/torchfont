use std::fmt;

#[derive(Debug)]
pub enum Error {
    Parse(String),
    Io(String),
    OutOfRange(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(msg) | Self::Io(msg) | Self::OutOfRange(msg) => write!(f, "{msg}"),
        }
    }
}

impl From<Error> for pyo3::PyErr {
    fn from(e: Error) -> Self {
        match e {
            Error::OutOfRange(_) => {
                pyo3::PyErr::new::<pyo3::exceptions::PyIndexError, _>(e.to_string())
            }
            Error::Parse(_) | Error::Io(_) => {
                pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
            }
        }
    }
}
