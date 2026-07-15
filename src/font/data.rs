use std::{fs, path::Path};

use memmap2::Mmap;

use crate::error::Error;

pub(crate) fn map_font(path: &Path) -> Result<Mmap, Error> {
    let file = fs::File::open(path).map_err(|err| {
        Error::Io(std::io::Error::new(
            err.kind(),
            format!("failed to open '{}': {err}", path.display()),
        ))
    })?;
    // SAFETY: callers only access the map while parsing and TorchFont documents
    // modification of indexed font files during use as unsupported.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
        Error::Io(std::io::Error::new(
            err.kind(),
            format!("failed to map '{}': {err}", path.display()),
        ))
    })?;
    Ok(mmap)
}

pub(crate) fn parse_font_ref<'a>(
    data: &'a [u8],
    path: &Path,
    ttc_index: u32,
) -> Result<skrifa::FontRef<'a>, Error> {
    skrifa::FontRef::from_index(data, ttc_index).map_err(|err| {
        Error::Parse(format!(
            "failed to parse '{}' (ttc_index {ttc_index}): {err}",
            path.display()
        ))
    })
}
