//! Dataset sample indexing and deterministic target tensor construction.
//!
//! This module owns font discovery for dataset construction, Python callable
//! resolution, the flattened sample order, class vocabularies, and target
//! expansion.

use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;

use numpy::{IntoPyArray as _, PyArray1};
use pyo3::{
    Bound,
    prelude::*,
    types::{PyAny, PyModule, PyType},
};

use crate::dataset::{canonicalize_root, discover_font_files};
use crate::font::{FontEntry, canonicalize_location, map_font, parse_font_ref};
use crate::py::callable::CallableBridge;

type Location = Vec<(String, f32)>;
type FixedFontArg = (PathBuf, u32, Vec<u32>, Vec<Location>, String, String);
type FixedLocationArg = (Py<PyAny>, u32, usize, u32, Location, usize, usize);
type VariableFontArg = (PathBuf, u32, Vec<u32>, usize);
type VariableLocationArg = (Py<PyAny>, u32, usize, u32, usize);

struct FixedFontPlan {
    path: PathBuf,
    ttc_index: u32,
    codepoints: Vec<u32>,
    locations: Vec<Location>,
    family_name: String,
    subfamily_name: String,
}

struct VariableFontPlan {
    path: PathBuf,
    ttc_index: u32,
    codepoints: Vec<u32>,
    instance_count: usize,
}

#[pyclass(module = "torchfont._torchfont")]
pub(crate) struct FixedGlyphIndex {
    fonts: Vec<FixedFontPlan>,
    sample_starts: Vec<usize>,
    style_starts: Vec<usize>,
    sample_count: usize,
    style_count: usize,
    character_codepoints: Vec<u32>,
}

#[pymethods]
impl FixedGlyphIndex {
    #[new]
    pub(crate) fn new(fonts: Vec<FixedFontArg>) -> PyResult<Self> {
        let fonts: Vec<_> = fonts
            .into_iter()
            .map(
                |(path, ttc_index, codepoints, locations, family_name, subfamily_name)| {
                    FixedFontPlan {
                        path,
                        ttc_index,
                        codepoints,
                        locations,
                        family_name,
                        subfamily_name,
                    }
                },
            )
            .collect();
        Self::from_plans(fonts)
    }

    #[classmethod]
    pub(crate) fn from_root(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        root: String,
        codepoints: Option<Vec<u32>>,
        patterns: Option<Vec<String>>,
        instances: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let bridge = CallableBridge::new(py)?;
        let native_fonts = font_entries(&root, codepoints, patterns)?;
        let mut fonts = Vec::new();
        for native_font in native_fonts {
            let locations = instance_locations(&bridge, &instances, &native_font)?;
            if locations.is_empty() {
                continue;
            }
            fonts.push(FixedFontPlan {
                path: native_font.path().to_path_buf(),
                ttc_index: native_font.ttc_index(),
                codepoints: native_font.codepoints().to_vec(),
                locations,
                family_name: native_font.family_name().to_string(),
                subfamily_name: native_font.subfamily_name().to_string(),
            });
        }
        Self::from_plans(fonts)
    }

    #[getter]
    pub(crate) fn sample_count(&self) -> usize {
        self.sample_count
    }

    #[getter]
    pub(crate) fn style_count(&self) -> usize {
        self.style_count
    }

    pub(crate) fn font_refs(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, u32)>> {
        self.fonts
            .iter()
            .map(|font| {
                Ok((
                    font.path.clone().into_pyobject(py)?.unbind(),
                    font.ttc_index,
                ))
            })
            .collect()
    }

    pub(crate) fn style_classes(&self) -> Vec<String> {
        self.fonts
            .iter()
            .flat_map(|font| {
                font.locations
                    .iter()
                    .map(|location| style_name(&font.family_name, &font.subfamily_name, location))
            })
            .collect()
    }

    pub(crate) fn character_codepoints(&self) -> Vec<u32> {
        self.character_codepoints.clone()
    }

    pub(crate) fn locate(&self, py: Python<'_>, idx: usize) -> PyResult<FixedLocationArg> {
        if idx >= self.sample_count {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "sample index {idx} out of range (len={})",
                self.sample_count
            )));
        }

        let font_idx = self.sample_starts.partition_point(|&start| start <= idx) - 1;
        let font_start = self.sample_starts[font_idx];
        let font = &self.fonts[font_idx];
        let codepoint_count = font.codepoints.len();
        let local = idx - font_start;
        let location_idx = local / codepoint_count;
        let codepoint_idx = local % codepoint_count;

        Ok((
            font.path.clone().into_pyobject(py)?.unbind(),
            font.ttc_index,
            font_idx,
            font.codepoints[codepoint_idx],
            font.locations[location_idx].clone(),
            self.style_starts[font_idx] + location_idx,
            self.character_index(font.codepoints[codepoint_idx]),
        ))
    }

    pub(crate) fn font_targets(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        let mut out = Vec::with_capacity(self.sample_count);
        for (font_idx, font) in self.fonts.iter().enumerate() {
            out.extend(std::iter::repeat_n(
                font_idx as i64,
                font.codepoints.len() * font.locations.len(),
            ));
        }
        out.into_pyarray(py).unbind()
    }

    pub(crate) fn style_targets(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        let mut out = Vec::with_capacity(self.sample_count);
        for (font, &style_start) in self.fonts.iter().zip(&self.style_starts) {
            for location_idx in 0..font.locations.len() {
                out.extend(std::iter::repeat_n(
                    (style_start + location_idx) as i64,
                    font.codepoints.len(),
                ));
            }
        }
        out.into_pyarray(py).unbind()
    }

    pub(crate) fn character_targets(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        let mut out = Vec::with_capacity(self.sample_count);
        for font in &self.fonts {
            let indices: Vec<_> = font
                .codepoints
                .iter()
                .map(|&codepoint| self.character_index(codepoint) as i64)
                .collect();
            for _ in 0..font.locations.len() {
                out.extend_from_slice(&indices);
            }
        }
        out.into_pyarray(py).unbind()
    }

    pub(crate) fn __getnewargs__(&self) -> (Vec<FixedFontArg>,) {
        (self
            .fonts
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

#[pyclass(module = "torchfont._torchfont")]
pub(crate) struct VariableGlyphIndex {
    fonts: Vec<VariableFontPlan>,
    sample_starts: Vec<usize>,
    sample_count: usize,
    character_codepoints: Vec<u32>,
}

#[pymethods]
impl VariableGlyphIndex {
    #[new]
    pub(crate) fn new(fonts: Vec<VariableFontArg>) -> PyResult<Self> {
        let fonts: Vec<_> = fonts
            .into_iter()
            .map(
                |(path, ttc_index, codepoints, instance_count)| VariableFontPlan {
                    path,
                    ttc_index,
                    codepoints,
                    instance_count,
                },
            )
            .collect();
        Self::from_plans(fonts)
    }

    #[classmethod]
    pub(crate) fn from_root(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        root: String,
        codepoints: Option<Vec<u32>>,
        patterns: Option<Vec<String>>,
        instance_count: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let bridge = CallableBridge::new(py)?;
        let native_fonts = font_entries(&root, codepoints, patterns)?;
        let mut fonts = Vec::new();
        for native_font in native_fonts {
            let count = bridge.call_instance_count(
                &instance_count,
                native_font.path(),
                native_font.ttc_index(),
            )?;
            if count == 0 {
                continue;
            }
            fonts.push(VariableFontPlan {
                path: native_font.path().to_path_buf(),
                ttc_index: native_font.ttc_index(),
                codepoints: native_font.codepoints().to_vec(),
                instance_count: count,
            });
        }
        Self::from_plans(fonts)
    }

    #[getter]
    pub(crate) fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub(crate) fn font_refs(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, u32)>> {
        self.fonts
            .iter()
            .map(|font| {
                Ok((
                    font.path.clone().into_pyobject(py)?.unbind(),
                    font.ttc_index,
                ))
            })
            .collect()
    }

    pub(crate) fn character_codepoints(&self) -> Vec<u32> {
        self.character_codepoints.clone()
    }

    pub(crate) fn locate(&self, py: Python<'_>, idx: usize) -> PyResult<VariableLocationArg> {
        if idx >= self.sample_count {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "sample index {idx} out of range (len={})",
                self.sample_count
            )));
        }

        let font_idx = self.sample_starts.partition_point(|&start| start <= idx) - 1;
        let font_start = self.sample_starts[font_idx];
        let font = &self.fonts[font_idx];
        let codepoint_count = font.codepoints.len();
        let local = idx - font_start;
        let codepoint_idx = local % codepoint_count;

        Ok((
            font.path.clone().into_pyobject(py)?.unbind(),
            font.ttc_index,
            font_idx,
            font.codepoints[codepoint_idx],
            self.character_index(font.codepoints[codepoint_idx]),
        ))
    }

    pub(crate) fn font_targets(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        let mut out = Vec::with_capacity(self.sample_count);
        for (font_idx, font) in self.fonts.iter().enumerate() {
            out.extend(std::iter::repeat_n(
                font_idx as i64,
                font.codepoints.len() * font.instance_count,
            ));
        }
        out.into_pyarray(py).unbind()
    }

    pub(crate) fn character_targets(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        let mut out = Vec::with_capacity(self.sample_count);
        for font in &self.fonts {
            let indices: Vec<_> = font
                .codepoints
                .iter()
                .map(|&codepoint| self.character_index(codepoint) as i64)
                .collect();
            for _ in 0..font.instance_count {
                out.extend_from_slice(&indices);
            }
        }
        out.into_pyarray(py).unbind()
    }

    pub(crate) fn __getnewargs__(&self) -> (Vec<VariableFontArg>,) {
        (self
            .fonts
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

impl FixedGlyphIndex {
    fn from_plans(fonts: Vec<FixedFontPlan>) -> PyResult<Self> {
        let (sample_starts, sample_count) = sample_starts_for_fixed(&fonts)?;
        let (style_starts, style_count) = style_starts_for_fixed(&fonts)?;
        let character_codepoints =
            character_index(fonts.iter().map(|font| font.codepoints.as_slice()));

        Ok(Self {
            fonts,
            sample_starts,
            style_starts,
            sample_count,
            style_count,
            character_codepoints,
        })
    }
}

impl VariableGlyphIndex {
    fn from_plans(fonts: Vec<VariableFontPlan>) -> PyResult<Self> {
        let (sample_starts, sample_count) = sample_starts_for_variable(&fonts)?;
        let character_codepoints =
            character_index(fonts.iter().map(|font| font.codepoints.as_slice()));

        Ok(Self {
            fonts,
            sample_starts,
            sample_count,
            character_codepoints,
        })
    }
}

impl FixedGlyphIndex {
    fn character_index(&self, codepoint: u32) -> usize {
        self.character_codepoints
            .binary_search(&codepoint)
            .expect("character index was built from all codepoints")
    }
}

impl VariableGlyphIndex {
    fn character_index(&self, codepoint: u32) -> usize {
        self.character_codepoints
            .binary_search(&codepoint)
            .expect("character index was built from all codepoints")
    }
}

fn sample_starts_for_fixed(fonts: &[FixedFontPlan]) -> PyResult<(Vec<usize>, usize)> {
    let mut starts = Vec::with_capacity(fonts.len());
    let mut offset = 0usize;
    for font in fonts {
        starts.push(offset);
        offset = offset
            .checked_add(fixed_font_sample_count(font)?)
            .ok_or_else(sample_count_overflow)?;
    }
    Ok((starts, offset))
}

fn font_entries(
    root: &str,
    codepoints: Option<Vec<u32>>,
    patterns: Option<Vec<String>>,
) -> PyResult<Vec<FontEntry>> {
    let filter = codepoints.map(|mut values| {
        values.sort_unstable();
        values.dedup();
        values
    });

    let root_path = canonicalize_root(root)?;
    let files = discover_font_files(&root_path, patterns.as_deref())?;
    let mut entries = Vec::new();
    for path in files {
        entries.extend(
            FontEntry::load_fonts(&path, filter.as_deref())?
                .into_iter()
                .filter(|entry| entry.codepoint_count() > 0),
        );
    }
    Ok(entries)
}

fn instance_locations(
    bridge: &CallableBridge<'_>,
    instances: &Bound<'_, PyAny>,
    font_entry: &FontEntry,
) -> PyResult<Vec<Location>> {
    let raw_locations =
        bridge.call_instances(instances, font_entry.path(), font_entry.ttc_index())?;
    let data = map_font(font_entry.path())?;
    let font = parse_font_ref(&data[..], font_entry.path(), font_entry.ttc_index())?;
    let locations = raw_locations
        .iter()
        .map(|location| {
            canonicalize_location(
                &font,
                font_entry.path(),
                font_entry.ttc_index(),
                Some(location),
            )
            .map_err(Into::into)
        })
        .collect::<PyResult<Vec<_>>>()?;
    reject_duplicate_locations(font_entry, &locations)?;
    Ok(locations)
}

fn reject_duplicate_locations(font_entry: &FontEntry, locations: &[Location]) -> PyResult<()> {
    let mut seen = HashSet::new();
    for location in locations {
        let key: Vec<_> = location
            .iter()
            .map(|(tag, value)| (tag.clone(), value.to_bits()))
            .collect();
        if !seen.insert(key) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "InstanceFn returned duplicate variation locations for '{}' \
                 (ttc_index={}) after canonicalization",
                font_entry.path().display(),
                font_entry.ttc_index()
            )));
        }
    }
    Ok(())
}

fn style_starts_for_fixed(fonts: &[FixedFontPlan]) -> PyResult<(Vec<usize>, usize)> {
    let mut starts = Vec::with_capacity(fonts.len());
    let mut offset = 0usize;
    for font in fonts {
        starts.push(offset);
        offset = offset
            .checked_add(font.locations.len())
            .ok_or_else(style_count_overflow)?;
    }
    Ok((starts, offset))
}

fn sample_starts_for_variable(fonts: &[VariableFontPlan]) -> PyResult<(Vec<usize>, usize)> {
    let mut starts = Vec::with_capacity(fonts.len());
    let mut offset = 0usize;
    for font in fonts {
        starts.push(offset);
        offset = offset
            .checked_add(variable_font_sample_count(font)?)
            .ok_or_else(sample_count_overflow)?;
    }
    Ok((starts, offset))
}

fn fixed_font_sample_count(font: &FixedFontPlan) -> PyResult<usize> {
    font.codepoints
        .len()
        .checked_mul(font.locations.len())
        .ok_or_else(sample_count_overflow)
}

fn variable_font_sample_count(font: &VariableFontPlan) -> PyResult<usize> {
    font.codepoints
        .len()
        .checked_mul(font.instance_count)
        .ok_or_else(sample_count_overflow)
}

fn character_index<'a>(codepoints: impl Iterator<Item = &'a [u32]> + Clone) -> Vec<u32> {
    codepoints
        .clone()
        .flat_map(|font_codepoints| font_codepoints.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn style_name(family_name: &str, subfamily_name: &str, location: &[(String, f32)]) -> String {
    if !location.is_empty() {
        return format!("{family_name} {}", format_location(location));
    }
    if !subfamily_name.is_empty() {
        return format!("{family_name} {subfamily_name}");
    }
    family_name.to_string()
}

fn format_location(location: &[(String, f32)]) -> String {
    location
        .iter()
        .map(|(tag, value)| format!("{tag}={}", format_axis_value(*value)))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_axis_value(value: f32) -> String {
    let mut formatted = format!("{value:.6}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    formatted
}

fn sample_count_overflow() -> PyErr {
    pyo3::exceptions::PyOverflowError::new_err("dataset sample count overflowed usize")
}

fn style_count_overflow() -> PyErr {
    pyo3::exceptions::PyOverflowError::new_err("dataset style count overflowed usize")
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FixedGlyphIndex>()?;
    m.add_class::<VariableGlyphIndex>()?;
    Ok(())
}
