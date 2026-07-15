use std::fmt;

#[derive(Debug)]
pub enum Error {
    Parse(String),
    Io(std::io::Error),
    OutOfRange(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(msg) | Self::OutOfRange(msg) => write!(f, "{msg}"),
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}
