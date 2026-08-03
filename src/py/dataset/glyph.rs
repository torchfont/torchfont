use std::path::PathBuf;

use numpy::{IntoPyArray as _, PyArray1};
use pyo3::{Bound, prelude::*, types::PyType};

use crate::dataset::{FontEntry, GlyphIndex as CoreGlyphIndex};

use super::{build, index_error, overflow_error};

type FontArg = (PathBuf, u32, Vec<u32>);
type LocationArg = (PathBuf, u32, usize, u32, usize);

#[pyclass(frozen, module = "torchfont._torchfont")]
pub(super) struct GlyphIndex {
    inner: CoreGlyphIndex,
}

#[pymethods]
impl GlyphIndex {
    #[new]
    fn new(fonts: Vec<FontArg>) -> PyResult<Self> {
        Self::from_entries(fonts.into_iter().map(font_entry).collect())
    }

    #[classmethod]
    fn from_root(
        _cls: &Bound<'_, PyType>,
        root: String,
        codepoints: Option<Vec<u32>>,
        patterns: Option<Vec<String>>,
    ) -> PyResult<Self> {
        Self::from_entries(build::build_entries(&root, codepoints, patterns)?)
    }

    #[getter]
    fn sample_count(&self) -> usize {
        self.inner.sample_count()
    }

    fn font_refs(&self) -> Vec<(PathBuf, u32)> {
        self.inner
            .fonts()
            .iter()
            .map(|font| (font.path.clone(), font.ttc_index))
            .collect()
    }

    fn character_codepoints(&self) -> Vec<u32> {
        self.inner.character_codepoints().to_vec()
    }

    fn locate(&self, idx: usize) -> PyResult<LocationArg> {
        let sample = self
            .inner
            .locate(idx)
            .ok_or_else(|| index_error(idx, self.inner.sample_count()))?;
        Ok((
            sample.path.to_path_buf(),
            sample.ttc_index,
            sample.font_idx,
            sample.codepoint,
            sample.character_idx,
        ))
    }

    fn font_targets(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        self.inner.font_targets().into_pyarray(py).unbind()
    }

    fn character_targets(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        self.inner.character_targets().into_pyarray(py).unbind()
    }

    fn __getnewargs__(&self) -> (Vec<FontArg>,) {
        (self
            .inner
            .fonts()
            .iter()
            .map(|font| (font.path.clone(), font.ttc_index, font.codepoints.clone()))
            .collect(),)
    }
}

impl GlyphIndex {
    fn from_entries(fonts: Vec<FontEntry>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreGlyphIndex::new(fonts).map_err(overflow_error)?,
        })
    }
}

fn font_entry((path, ttc_index, codepoints): FontArg) -> FontEntry {
    FontEntry {
        path,
        ttc_index,
        codepoints,
    }
}
