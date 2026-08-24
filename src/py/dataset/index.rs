use std::path::PathBuf;

use numpy::{IntoPyArray as _, PyArray1};
use pyo3::{Py, PyResult, Python, pyfunction};

use crate::dataset::{
    CodepointIndex, DiscoveredCodepoints, DiscoveredGlyphs, GlyphIndex, build_from_files,
    canonicalize_root, discover_font_files,
};

type CodepointIndexArrays = (
    Vec<(PathBuf, u32)>,
    Vec<i64>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
);

type GlyphIndexArrays = (Vec<(PathBuf, u32)>, Vec<i64>, Py<PyArray1<u32>>);

#[pyfunction]
pub(super) fn index_codepoints(
    py: Python<'_>,
    root: &str,
    codepoints: Option<Vec<u32>>,
    max_length: Option<usize>,
    patterns: Option<Vec<String>>,
) -> PyResult<CodepointIndexArrays> {
    let index = py.detach(|| {
        let fonts = discover_codepoint_fonts(root, codepoints, max_length, patterns)?;
        Ok::<_, pyo3::PyErr>(CodepointIndex::build(
            fonts
                .into_iter()
                .map(DiscoveredCodepoints::into_parts)
                .collect(),
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
        index.glyph_ids.into_pyarray(py).unbind(),
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
        let fonts = discover_glyph_fonts(root, max_length, patterns)?;
        Ok::<_, pyo3::PyErr>(GlyphIndex::build(
            fonts
                .into_iter()
                .map(DiscoveredGlyphs::into_parts)
                .collect(),
        ))
    })?;
    Ok((
        index.fonts,
        index.offsets,
        index.glyph_ids.into_pyarray(py).unbind(),
    ))
}

fn discover_codepoint_fonts(
    root: &str,
    codepoints: Option<Vec<u32>>,
    max_length: Option<usize>,
    patterns: Option<Vec<String>>,
) -> PyResult<Vec<DiscoveredCodepoints>> {
    let filter = codepoints.map(|mut values| {
        values.sort_unstable();
        values.dedup();
        values
    });
    let root = canonicalize_root(root)?;
    let files = discover_font_files(&root, patterns.as_deref())?;
    Ok(build_from_files(&files, |path| {
        Ok(
            DiscoveredCodepoints::from_file(path, filter.as_deref(), max_length)?
                .into_iter()
                .filter(|entry| entry.codepoint_count() > 0)
                .collect(),
        )
    })?)
}

fn discover_glyph_fonts(
    root: &str,
    max_length: Option<usize>,
    patterns: Option<Vec<String>>,
) -> PyResult<Vec<DiscoveredGlyphs>> {
    let root = canonicalize_root(root)?;
    let files = discover_font_files(&root, patterns.as_deref())?;
    Ok(build_from_files(&files, |path| {
        Ok(DiscoveredGlyphs::from_file(path, max_length)?
            .into_iter()
            .filter(|entry| entry.glyph_count() > 0)
            .collect())
    })?)
}
