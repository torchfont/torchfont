use std::path::PathBuf;

use numpy::{IntoPyArray as _, PyArray1};
use pyo3::{Py, PyResult, Python, pyfunction};

use crate::dataset::{
    CodepointIndex, GlyphIndex, build_from_files, canonicalize_root, discover_codepoints,
    discover_font_files, discover_glyphs,
};

use crate::error::Error;

type CodepointIndexArrays = (
    Vec<(PathBuf, u32)>,
    Vec<i64>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
);

type GlyphIndexArrays = (
    Vec<(PathBuf, u32)>,
    Vec<i64>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
);

#[pyfunction]
pub(super) fn index_codepoints(
    py: Python<'_>,
    root: &str,
    codepoints: Option<Vec<u32>>,
    max_length: Option<usize>,
    patterns: Option<Vec<String>>,
) -> PyResult<CodepointIndexArrays> {
    let index = py.detach(|| {
        let filter = codepoints.map(|mut values| {
            values.sort_unstable();
            values.dedup();
            values
        });
        let root = canonicalize_root(root)?;
        let files = discover_font_files(&root, patterns.as_deref())?;
        let mut fonts = build_from_files(&files, |path| {
            discover_codepoints(path, filter.as_deref(), max_length)
        })?;
        fonts.retain(|face| !face.codepoints.is_empty());
        Ok::<_, Error>(CodepointIndex::build(fonts))
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
        index.glyph_ids.into_pyarray(py).unbind(),
        index.outline_lengths.into_pyarray(py).unbind(),
    ))
}

#[pyfunction]
pub(super) fn index_glyphs(
    py: Python<'_>,
    root: &str,
    max_length: Option<usize>,
    patterns: Option<Vec<String>>,
) -> PyResult<GlyphIndexArrays> {
    let index = py.detach(|| {
        let root = canonicalize_root(root)?;
        let files = discover_font_files(&root, patterns.as_deref())?;
        let mut fonts = build_from_files(&files, |path| discover_glyphs(path, max_length))?;
        fonts.retain(|face| !face.glyph_ids.is_empty());
        Ok::<_, Error>(GlyphIndex::build(fonts))
    })?;
    Ok((
        index.fonts,
        index.offsets,
        index.glyph_ids.into_pyarray(py).unbind(),
        index.outline_lengths.into_pyarray(py).unbind(),
    ))
}
