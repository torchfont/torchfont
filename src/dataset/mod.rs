mod entry;
mod index;
mod io;
mod reader;

use crate::error::py_index_err;
use entry::FontEntry;
use index::{DatasetIndex, load_entries_and_index};
use io::{canonicalize_root, discover_font_files};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;

type LocateResult = (
    String,
    u32,
    Option<usize>,
    u32,
    usize,
    usize,
    Vec<(String, f32)>,
);

type ItemResult = (Vec<i32>, Vec<f32>, Vec<f32>, usize, usize);

#[pyclass]
pub struct FontDataset {
    entries: Vec<FontEntry>,
    index: DatasetIndex,
}

#[pymethods]
impl FontDataset {
    #[new]
    pub fn new(
        root: String,
        codepoints: Option<Vec<u32>>,
        patterns: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let filter = codepoints.map(|mut values| {
            values.sort_unstable();
            values.dedup();
            values
        });

        let root_path = canonicalize_root(&root)?;
        let files = discover_font_files(&root_path, patterns.as_deref())?;
        let (entries, index) = load_entries_and_index(files, filter.as_deref())?;

        Ok(Self { entries, index })
    }

    #[getter]
    pub fn sample_count(&self) -> usize {
        self.index.sample_offsets.last().copied().unwrap_or(0)
    }

    #[getter]
    pub fn content_class_count(&self) -> usize {
        self.index.content_classes.len()
    }

    pub fn content_metadata_rows(&self) -> PyResult<Vec<(String, String, u32)>> {
        self.index
            .content_classes
            .iter()
            .copied()
            .map(|codepoint| {
                let ch = char::from_u32(codepoint).ok_or_else(|| {
                    crate::error::py_err(format!(
                        "indexed codepoint U+{codepoint:04X} is not a Unicode scalar value"
                    ))
                })?;
                Ok((
                    format!("content:U+{codepoint:04X}"),
                    ch.to_string(),
                    codepoint,
                ))
            })
            .collect()
    }

    #[getter]
    pub fn style_class_count(&self) -> usize {
        self.index.inst_offsets.last().copied().unwrap_or(0)
    }

    pub fn style_metadata_rows(&self, root: String) -> PyResult<Vec<(String, String)>> {
        let root_path = Path::new(&root);
        self.style_rows()
            .into_iter()
            .map(|(name, path, face_idx, instance_idx)| {
                Ok((
                    name,
                    style_label_id(root_path, Path::new(&path), face_idx, instance_idx)?,
                ))
            })
            .collect()
    }

    #[getter]
    pub fn style_axes(&self) -> Vec<Vec<(String, f32)>> {
        let mut axes = Vec::new();
        for entry in self.entries.iter() {
            axes.extend(entry.style_axes().iter().cloned());
        }
        axes
    }

    pub fn locate(&self, idx: usize) -> PyResult<LocateResult> {
        let (font_idx, instance_idx, codepoint, style_idx, content_idx) = self.locate_parts(idx)?;
        let entry = &self.entries[font_idx];
        let axes = instance_idx
            .map(|inst_idx| entry.style_axes()[inst_idx].clone())
            .unwrap_or_default();
        Ok((
            entry.path().to_owned(),
            entry.face_index(),
            instance_idx,
            codepoint,
            style_idx,
            content_idx,
            axes,
        ))
    }

    pub fn item(&self, idx: usize) -> PyResult<ItemResult> {
        let (font_idx, inst_idx, codepoint, style_idx, content_idx) = self.locate_parts(idx)?;
        self.entries[font_idx]
            .glyph(codepoint, inst_idx)
            .map(|(types, coords, metrics)| (types, coords, metrics, style_idx, content_idx))
    }

    pub fn targets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let total = self.sample_count();
        let capacity = total
            .checked_mul(2)
            .and_then(|n| n.checked_mul(std::mem::size_of::<i64>()))
            .ok_or_else(|| {
                pyo3::exceptions::PyOverflowError::new_err("target buffer size overflowed usize")
            })?;
        let mut bytes = Vec::with_capacity(capacity);
        for (font_idx, entry) in self.entries.iter().enumerate() {
            let inst_offset = self.index.inst_offsets[font_idx];
            for inst_idx in 0..entry.instance_count() {
                let style_idx = inst_offset + inst_idx;
                for &cp in entry.codepoints() {
                    let content_idx = self.index.content_index(cp)?;
                    bytes.extend_from_slice(&(style_idx as i64).to_ne_bytes());
                    bytes.extend_from_slice(&(content_idx as i64).to_ne_bytes());
                }
            }
        }
        Ok(PyBytes::new(py, &bytes))
    }
}

impl FontDataset {
    fn style_rows(&self) -> Vec<(String, String, u32, Option<usize>)> {
        let mut rows = Vec::new();
        for entry in self.entries.iter() {
            let path = entry.path().to_owned();
            let face_idx = entry.face_index();
            let family_name = entry.family_name();

            if entry.is_variable() {
                let instance_names = entry.named_instance_names();
                if instance_names.is_empty() {
                    let display_name = if let Some(subfamily) = entry.subfamily_name() {
                        format!("{family_name} {subfamily}")
                    } else {
                        family_name
                    };
                    rows.push((display_name, path, face_idx, None));
                } else {
                    for (inst_idx, name_opt) in instance_names.iter().enumerate() {
                        let instance_name = name_opt.as_deref().unwrap_or("");
                        let display_name = if instance_name.is_empty() {
                            family_name.clone()
                        } else {
                            format!("{family_name} {instance_name}")
                        };
                        rows.push((display_name, path.clone(), face_idx, Some(inst_idx)));
                    }
                }
            } else if let Some(subfamily) = entry.subfamily_name() {
                rows.push((format!("{family_name} {subfamily}"), path, face_idx, None));
            } else {
                rows.push((family_name, path, face_idx, None));
            }
        }
        rows
    }

    fn locate_parts(&self, idx: usize) -> PyResult<(usize, Option<usize>, u32, usize, usize)> {
        let total = self.sample_count();
        if idx >= total {
            return Err(py_index_err(format!(
                "sample index {idx} out of range (len={total})"
            )));
        }

        let font_idx = self
            .index
            .sample_offsets
            .partition_point(|offset| *offset <= idx)
            - 1;

        let entry = &self.entries[font_idx];
        let font_start = self.index.sample_offsets[font_idx];
        let sample_idx = idx - font_start;
        let cp_count = entry.codepoint_count();
        debug_assert!(
            cp_count > 0,
            "font '{}' has no indexed code points",
            entry.path()
        );

        let inst_start = self.index.inst_offsets[font_idx];
        let inst_idx = sample_idx / cp_count;
        debug_assert!(
            inst_idx < entry.instance_count(),
            "instance index {} out of range for font '{}'",
            inst_idx,
            entry.path()
        );

        let cp_offset = sample_idx % cp_count;
        let cp = entry.codepoints()[cp_offset];
        let style_idx = inst_start + inst_idx;
        let content_idx = self.index.content_index(cp)?;
        let instance = entry.is_variable().then_some(inst_idx);

        Ok((font_idx, instance, cp, style_idx, content_idx))
    }
}

fn style_label_id(
    root: &Path,
    font_path: &Path,
    face_idx: u32,
    instance_idx: Option<usize>,
) -> PyResult<String> {
    let relative_path = font_path.strip_prefix(root).map_err(|_| {
        crate::error::py_err(format!(
            "font path '{}' is not under dataset root '{}'",
            font_path.display(),
            root.display()
        ))
    })?;
    let quoted_path = relative_path
        .components()
        .map(|component| urlencoding::encode(&component.as_os_str().to_string_lossy()).into_owned())
        .collect::<Vec<_>>()
        .join("/");
    let instance_value = instance_idx.map_or_else(|| "static".to_string(), |idx| idx.to_string());
    Ok(format!(
        "style:path={quoted_path};face={face_idx};instance={instance_value}"
    ))
}
