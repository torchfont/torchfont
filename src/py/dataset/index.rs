use std::path::PathBuf;

use numpy::{IntoPyArray as _, PyArray1};
use pyo3::{Py, PyResult, Python, pyfunction};

use crate::dataset::{DiscoveredFont, FontIndex, canonicalize_root, discover_font_files};

type FontIndexArrays = (
    Vec<(PathBuf, u32)>,
    Vec<i64>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
);

#[pyfunction]
pub(super) fn index_fonts(
    py: Python<'_>,
    root: &str,
    codepoints: Option<Vec<u32>>,
    patterns: Option<Vec<String>>,
) -> PyResult<FontIndexArrays> {
    let index = py.detach(|| {
        let fonts = discover_fonts(root, codepoints, patterns)?;
        Ok::<_, pyo3::PyErr>(FontIndex::build(
            fonts.into_iter().map(DiscoveredFont::into_parts).collect(),
        ))
    })?;
    Ok((
        index.fonts,
        index.offsets,
        index
            .character_codepoints
            .into_boxed_slice()
            .into_pyarray(py)
            .unbind(),
        index.character_index.into_pyarray(py).unbind(),
    ))
}

fn discover_fonts(
    root: &str,
    codepoints: Option<Vec<u32>>,
    patterns: Option<Vec<String>>,
) -> PyResult<Vec<DiscoveredFont>> {
    let filter = codepoints.map(|mut values| {
        values.sort_unstable();
        values.dedup();
        values
    });
    let root = canonicalize_root(root)?;
    let mut entries = Vec::new();
    for path in discover_font_files(&root, patterns.as_deref())? {
        entries.extend(
            DiscoveredFont::from_file(&path, filter.as_deref())?
                .into_iter()
                .filter(|entry| entry.codepoint_count() > 0),
        );
    }
    Ok(entries)
}
