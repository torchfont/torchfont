use std::path::PathBuf;

use numpy::{IntoPyArray as _, PyArray1};
use pyo3::{
    Bound,
    prelude::*,
    types::{PyAny, PyType},
};

use crate::dataset::{FixedFontEntry, FixedIndex};
use crate::font::{Location, map_font, parse_font_ref, registered_axis_values};

use super::{build, index_error, overflow_error};

type FixedFontArg = (PathBuf, u32, Vec<u32>, Vec<Location>, String, String);
type FixedLocationArg = (
    PathBuf,
    u32,
    usize,
    u32,
    Location,
    usize,
    usize,
    Option<f32>,
    Option<f32>,
    Option<f32>,
    Option<f32>,
    Option<f32>,
);

#[pyclass(frozen, module = "torchfont._torchfont")]
pub(super) struct FixedGlyphIndex {
    inner: FixedIndex,
}

#[pymethods]
impl FixedGlyphIndex {
    #[new]
    fn new(fonts: Vec<FixedFontArg>) -> PyResult<Self> {
        Self::from_entries(fonts.into_iter().map(fixed_entry).collect())
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
        Self::from_entries(build::build_fixed_entries(
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

    #[getter]
    fn style_count(&self) -> usize {
        self.inner.style_count()
    }

    fn font_refs(&self) -> Vec<(PathBuf, u32)> {
        self.inner
            .fonts()
            .iter()
            .map(|font| (font.path.clone(), font.ttc_index))
            .collect()
    }

    fn style_classes(&self) -> Vec<String> {
        self.inner.style_classes()
    }

    fn character_codepoints(&self) -> Vec<u32> {
        self.inner.character_codepoints().to_vec()
    }

    fn locate(&self, idx: usize) -> PyResult<FixedLocationArg> {
        let sample = self
            .inner
            .locate(idx)
            .ok_or_else(|| index_error(idx, self.inner.sample_count()))?;
        let data = map_font(sample.path)?;
        let font_ref = parse_font_ref(&data, sample.path, sample.ttc_index)?;
        let values = registered_axis_values(&font_ref, sample.location);
        Ok((
            sample.path.to_path_buf(),
            sample.ttc_index,
            sample.font_idx,
            sample.codepoint,
            sample.location.clone(),
            sample.style_idx,
            sample.character_idx,
            finite_or_none(values.weight),
            finite_or_none(values.width),
            finite_or_none(values.italic),
            finite_or_none(values.slant),
            finite_or_none(values.optical_size),
        ))
    }

    fn font_targets(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        self.inner.font_targets().into_pyarray(py).unbind()
    }

    fn style_targets(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        self.inner.style_targets().into_pyarray(py).unbind()
    }

    fn character_targets(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        self.inner.character_targets().into_pyarray(py).unbind()
    }

    fn weight_targets(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(self.inner.weight_targets()?.into_pyarray(py).unbind())
    }

    fn width_targets(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(self.inner.width_targets()?.into_pyarray(py).unbind())
    }

    fn italic_targets(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(self.inner.italic_targets()?.into_pyarray(py).unbind())
    }

    fn slant_targets(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(self.inner.slant_targets()?.into_pyarray(py).unbind())
    }

    fn optical_size_targets(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f32>>> {
        Ok(self.inner.optical_size_targets()?.into_pyarray(py).unbind())
    }

    fn __getnewargs__(&self) -> (Vec<FixedFontArg>,) {
        (self
            .inner
            .fonts()
            .iter()
            .map(|font| {
                (
                    font.path.clone(),
                    font.ttc_index,
                    font.codepoints.clone(),
                    font.locations.clone(),
                    font.family_name.clone(),
                    font.subfamily_name.clone(),
                )
            })
            .collect(),)
    }
}

impl FixedGlyphIndex {
    fn from_entries(fonts: Vec<FixedFontEntry>) -> PyResult<Self> {
        Ok(Self {
            inner: FixedIndex::new(fonts).map_err(overflow_error)?,
        })
    }
}

fn finite_or_none(value: f32) -> Option<f32> {
    value.is_finite().then_some(value)
}

fn fixed_entry(args: FixedFontArg) -> FixedFontEntry {
    let (path, ttc_index, codepoints, locations, family_name, subfamily_name) = args;
    FixedFontEntry {
        path,
        ttc_index,
        codepoints,
        locations,
        family_name,
        subfamily_name,
    }
}
