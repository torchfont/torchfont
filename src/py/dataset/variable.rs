use std::path::PathBuf;

use numpy::{IntoPyArray as _, PyArray1};
use pyo3::{
    Bound,
    prelude::*,
    types::{PyAny, PyType},
};

use crate::dataset::{VariableFontEntry, VariableIndex};

use super::{build, index_error, overflow_error};

type VariableFontArg = (PathBuf, u32, Vec<u32>, usize);
type VariableLocationArg = (PathBuf, u32, usize, u32, usize);

#[pyclass(frozen, module = "torchfont._torchfont")]
pub(super) struct VariableGlyphIndex {
    inner: VariableIndex,
}

#[pymethods]
impl VariableGlyphIndex {
    #[new]
    fn new(fonts: Vec<VariableFontArg>) -> PyResult<Self> {
        Self::from_entries(fonts.into_iter().map(variable_entry).collect())
    }

    #[classmethod]
    fn from_root(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        root: String,
        codepoints: Option<Vec<u32>>,
        patterns: Option<Vec<String>>,
        instance_fn: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Self::from_entries(build::build_variable_entries(
            py,
            &root,
            codepoints,
            patterns,
            &instance_fn,
        )?)
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

    fn locate(&self, idx: usize) -> PyResult<VariableLocationArg> {
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

    fn __getnewargs__(&self) -> (Vec<VariableFontArg>,) {
        (self
            .inner
            .fonts()
            .iter()
            .map(|font| {
                (
                    font.path.clone(),
                    font.ttc_index,
                    font.codepoints.clone(),
                    font.instance_count,
                )
            })
            .collect(),)
    }
}

impl VariableGlyphIndex {
    fn from_entries(fonts: Vec<VariableFontEntry>) -> PyResult<Self> {
        Ok(Self {
            inner: VariableIndex::new(fonts).map_err(overflow_error)?,
        })
    }
}

fn variable_entry(args: VariableFontArg) -> VariableFontEntry {
    let (path, ttc_index, codepoints, instance_count) = args;
    VariableFontEntry {
        path,
        ttc_index,
        codepoints,
        instance_count,
    }
}
