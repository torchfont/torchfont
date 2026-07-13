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
