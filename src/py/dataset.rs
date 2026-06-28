use numpy::{IntoPyArray as _, PyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::dataset::{
    DatasetIndex, FontEntry, canonicalize_root, discover_font_files, load_entries_and_index,
    structure_fingerprint,
};
use crate::font::VariationInstantiation;

#[pyclass(module = "torchfont._torchfont")]
pub(crate) struct DefaultInstantiation;

#[pymethods]
impl DefaultInstantiation {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }
}

#[pyclass(module = "torchfont._torchfont")]
pub(crate) struct NamedInstantiation;

#[pymethods]
impl NamedInstantiation {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }
}

#[pyclass(module = "torchfont._torchfont")]
pub(crate) struct GridInstantiation {
    #[pyo3(get)]
    pub axes: HashMap<String, u32>,
}

#[pymethods]
impl GridInstantiation {
    #[new]
    pub fn new(axes: HashMap<String, u32>) -> PyResult<Self> {
        if axes.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "GridInstantiation requires at least one axis in axes; \
                 use DefaultInstantiation to pin every axis to its default",
            ));
        }
        if axes.values().any(|&count| count == 0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "GridInstantiation axis densities must be greater than zero",
            ));
        }
        Ok(Self { axes })
    }

    pub fn __getnewargs__(&self) -> (HashMap<String, u32>,) {
        (self.axes.clone(),)
    }
}

#[derive(FromPyObject)]
pub(crate) enum PyVariationInstantiation<'py> {
    Default(PyRef<'py, DefaultInstantiation>),
    Named(PyRef<'py, NamedInstantiation>),
    Grid(PyRef<'py, GridInstantiation>),
}

impl From<PyVariationInstantiation<'_>> for VariationInstantiation {
    fn from(value: PyVariationInstantiation<'_>) -> Self {
        match value {
            PyVariationInstantiation::Default(_inst) => Self::Default,
            PyVariationInstantiation::Named(_inst) => Self::Named,
            PyVariationInstantiation::Grid(grid) => Self::Grid {
                axes: grid.axes.clone(),
            },
        }
    }
}

#[pyclass(get_all)]
pub(crate) struct GlyphItem {
    pub types: Py<PyArray1<i64>>,
    pub coords: Py<PyArray1<f32>>,
    pub style_idx: usize,
    pub content_idx: usize,
    pub head: Py<PyArray1<f32>>,
    pub hhea: Py<PyArray1<f32>>,
    pub os2: Py<PyArray1<f32>>,
    pub post: Py<PyArray1<f32>>,
    pub maxp: Py<PyArray1<f32>>,
    pub hmtx: Py<PyArray1<f32>>,
    pub bounds: Py<PyArray1<f32>>,
    pub name: Py<PyDict>,
    pub codepoint: u32,
    pub glyph_name: String,
}

#[pyclass]
pub(crate) struct GlyphDatasetBackend {
    root: PathBuf,
    entries: Vec<FontEntry>,
    index: DatasetIndex,
    fingerprint: u64,
}

#[pymethods]
impl GlyphDatasetBackend {
    #[new]
    pub fn new(
        root: String,
        codepoints: Option<Vec<u32>>,
        patterns: Option<Vec<String>>,
        variation_instantiation: Option<PyVariationInstantiation<'_>>,
    ) -> PyResult<Self> {
        let filter = codepoints.map(|mut values| {
            values.sort_unstable();
            values.dedup();
            values
        });

        let root_path = canonicalize_root(&root)?;
        let files = discover_font_files(&root_path, patterns.as_deref())?;
        let variation_instantiation = match variation_instantiation {
            Some(value) => value.into(),
            None => VariationInstantiation::Named,
        };
        let (entries, index) =
            load_entries_and_index(files, filter.as_deref(), &variation_instantiation)?;
        let fingerprint = structure_fingerprint(&entries);

        Ok(Self {
            root: root_path,
            entries,
            index,
            fingerprint,
        })
    }

    #[getter]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    #[getter]
    pub fn sample_count(&self) -> usize {
        self.index.sample_offsets.last().copied().unwrap_or(0)
    }

    #[getter]
    pub fn content_class_count(&self) -> usize {
        self.index.content_classes.len()
    }

    pub fn content_metadata_rows(&self) -> Vec<(String, String, u32)> {
        self.index
            .content_classes
            .iter()
            .copied()
            .map(|codepoint| {
                let ch = char::from_u32(codepoint)
                    .expect("codepoints in content index are pre-validated Unicode scalar values");
                (
                    format!("content:U+{codepoint:04X}"),
                    ch.to_string(),
                    codepoint,
                )
            })
            .collect()
    }

    #[getter]
    pub fn style_class_count(&self) -> usize {
        self.index.inst_offsets.last().copied().unwrap_or(0)
    }

    pub fn style_metadata_rows(&self) -> PyResult<Vec<(String, String)>> {
        self.style_rows()
            .into_iter()
            .map(|(name, path, face_idx, instance_label)| {
                Ok((
                    name,
                    style_label_id(&self.root, &path, face_idx, instance_label)?,
                ))
            })
            .collect()
    }

    #[getter]
    pub fn style_axes(&self) -> Vec<Vec<(String, f32)>> {
        self.entries
            .iter()
            .flat_map(|entry| {
                (0..entry.instance_count()).map(move |index| entry.style_axes_at(index))
            })
            .collect()
    }

    pub fn item(&self, py: Python<'_>, idx: usize) -> PyResult<GlyphItem> {
        let (font_idx, inst_idx, codepoint, style_idx, content_idx) = self.locate_parts(idx)?;
        let entry = &self.entries[font_idx];
        let (outline, hmtx, bounds, glyph_name) = entry.glyph_complete(codepoint, inst_idx)?;

        let upem = entry.head.units_per_em as f32;
        let opt_u16 = |v: Option<u16>| v.map_or(f32::NAN, |v| v as f32);

        let head_arr: Vec<f32> = vec![
            upem,
            entry.head.flags as f32,
            entry.head.x_min,
            entry.head.y_min,
            entry.head.x_max,
            entry.head.y_max,
            entry.head.mac_style as f32,
            entry.head.lowest_rec_ppem as f32,
        ];

        let hhea_arr: Vec<f32> = vec![
            entry.hhea.ascender,
            entry.hhea.descender,
            entry.hhea.line_gap,
            entry.hhea.advance_width_max,
            entry.hhea.min_left_side_bearing,
            entry.hhea.min_right_side_bearing,
            entry.hhea.x_max_extent,
            entry.hhea.caret_slope_rise as f32,
            entry.hhea.caret_slope_run as f32,
            entry.hhea.caret_offset,
        ];

        let mut os2_arr: Vec<f32> = vec![
            entry.os2.weight_class as f32,
            entry.os2.width_class as f32,
            entry.os2.fs_type as f32,
            entry.os2.fs_selection as f32,
            entry.os2.typo_ascender,
            entry.os2.typo_descender,
            entry.os2.typo_line_gap,
            entry.os2.win_ascent,
            entry.os2.win_descent,
            entry.os2.avg_char_width,
            entry.os2.y_subscript_x_size,
            entry.os2.y_subscript_y_size,
            entry.os2.y_subscript_x_offset,
            entry.os2.y_subscript_y_offset,
            entry.os2.y_superscript_x_size,
            entry.os2.y_superscript_y_size,
            entry.os2.y_superscript_x_offset,
            entry.os2.y_superscript_y_offset,
            entry.os2.y_strikeout_size,
            entry.os2.y_strikeout_position,
            entry.os2.s_family_class as f32,
        ];
        os2_arr.extend(entry.os2.panose.iter().map(|&b| b as f32));
        os2_arr.extend(entry.os2.vend_id.iter().map(|&b| b as f32));
        os2_arr.extend([
            entry.os2.us_first_char_index as f32,
            entry.os2.us_last_char_index as f32,
            entry.os2.x_height.unwrap_or(f32::NAN),
            entry.os2.cap_height.unwrap_or(f32::NAN),
            opt_u16(entry.os2.us_default_char),
            opt_u16(entry.os2.us_break_char),
            opt_u16(entry.os2.us_max_context),
        ]);

        let post_arr: Vec<f32> = vec![
            entry.post.italic_angle,
            f32::from(entry.post.is_fixed_pitch),
            entry.post.underline_position,
            entry.post.underline_thickness,
        ];

        let maxp_arr: Vec<f32> = vec![
            entry.maxp.num_glyphs as f32,
            opt_u16(entry.maxp.max_points),
            opt_u16(entry.maxp.max_contours),
            opt_u16(entry.maxp.max_composite_points),
            opt_u16(entry.maxp.max_composite_contours),
            opt_u16(entry.maxp.max_zones),
            opt_u16(entry.maxp.max_twilight_points),
            opt_u16(entry.maxp.max_storage),
            opt_u16(entry.maxp.max_function_defs),
            opt_u16(entry.maxp.max_instruction_defs),
            opt_u16(entry.maxp.max_stack_elements),
            opt_u16(entry.maxp.max_size_of_instructions),
            opt_u16(entry.maxp.max_component_elements),
            opt_u16(entry.maxp.max_component_depth),
        ];

        let hmtx_arr: Vec<f32> = vec![hmtx.advance_width, hmtx.lsb];

        let bounds_arr: Vec<f32> = vec![bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max];

        let (types, coords) = outline.encode();

        let n = &entry.name;
        let name = {
            let d = PyDict::new(py);
            d.set_item("copyright_notice", &n.copyright_notice)?;
            d.set_item("family_name", &n.family_name)?;
            d.set_item("subfamily_name", &n.subfamily_name)?;
            d.set_item("unique_id", &n.unique_id)?;
            d.set_item("full_name", &n.full_name)?;
            d.set_item("version_string", &n.version_string)?;
            d.set_item("postscript_name", &n.postscript_name)?;
            d.set_item("trademark", &n.trademark)?;
            d.set_item("manufacturer", &n.manufacturer)?;
            d.set_item("designer", &n.designer)?;
            d.set_item("description", &n.description)?;
            d.set_item("vendor_url", &n.vendor_url)?;
            d.set_item("designer_url", &n.designer_url)?;
            d.set_item("license_description", &n.license_description)?;
            d.set_item("license_url", &n.license_url)?;
            d.set_item("reserved", &n.reserved)?;
            d.set_item("typographic_family_name", &n.typographic_family_name)?;
            d.set_item("typographic_subfamily_name", &n.typographic_subfamily_name)?;
            d.set_item("compatible_full_name", &n.compatible_full_name)?;
            d.set_item("sample_text", &n.sample_text)?;
            d.set_item("postscript_cid_name", &n.postscript_cid_name)?;
            d.set_item("wws_family_name", &n.wws_family_name)?;
            d.set_item("wws_subfamily_name", &n.wws_subfamily_name)?;
            d.set_item("light_background_palette", &n.light_background_palette)?;
            d.set_item("dark_background_palette", &n.dark_background_palette)?;
            d.set_item(
                "variations_postscript_name_prefix",
                &n.variations_postscript_name_prefix,
            )?;
            d.unbind()
        };

        Ok(GlyphItem {
            types: types.into_pyarray(py).unbind(),
            coords: coords.into_pyarray(py).unbind(),
            style_idx,
            content_idx,
            head: head_arr.into_pyarray(py).unbind(),
            hhea: hhea_arr.into_pyarray(py).unbind(),
            os2: os2_arr.into_pyarray(py).unbind(),
            post: post_arr.into_pyarray(py).unbind(),
            maxp: maxp_arr.into_pyarray(py).unbind(),
            hmtx: hmtx_arr.into_pyarray(py).unbind(),
            bounds: bounds_arr.into_pyarray(py).unbind(),
            name,
            codepoint,
            glyph_name,
        })
    }

    pub fn targets<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<i64>>> {
        let total = self.sample_count();
        let mut pairs: Vec<i64> = Vec::with_capacity(total * 2);
        for (font_idx, entry) in self.entries.iter().enumerate() {
            let inst_offset = self.index.inst_offsets[font_idx];
            for inst_idx in 0..entry.instance_count() {
                let style_idx = inst_offset + inst_idx;
                for &cp in entry.codepoints() {
                    let content_idx = self.index.content_index(cp)?;
                    pairs.extend([style_idx as i64, content_idx as i64]);
                }
            }
        }
        Ok(pairs.into_pyarray(py).unbind())
    }
}

impl GlyphDatasetBackend {
    fn style_rows(&self) -> Vec<(String, PathBuf, u32, Option<String>)> {
        let mut rows = Vec::new();
        for entry in self.entries.iter() {
            let path = entry.path().to_owned();
            let face_idx = entry.face_index();
            let n = &entry.name;
            let family = if !n.typographic_family_name.is_empty() {
                &n.typographic_family_name
            } else {
                &n.family_name
            };
            if entry.is_variable() {
                for index in 0..entry.instance_count() {
                    let axes = entry.style_axes_at(index);
                    let location = axes_label(&axes);
                    let display_name = if location.is_empty() {
                        family.clone()
                    } else {
                        format!("{family} {location}")
                    };
                    rows.push((display_name, path.clone(), face_idx, Some(location)));
                }
            } else {
                let sub = if !n.typographic_subfamily_name.is_empty() {
                    &n.typographic_subfamily_name
                } else {
                    &n.subfamily_name
                };
                let display_name = if sub.is_empty() {
                    family.clone()
                } else {
                    format!("{family} {sub}")
                };
                rows.push((display_name, path, face_idx, None));
            }
        }
        rows
    }

    fn locate_parts(&self, idx: usize) -> PyResult<(usize, Option<usize>, u32, usize, usize)> {
        let total = self.sample_count();
        if idx >= total {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "sample index {idx} out of range (len={total})"
            )));
        }

        let font_idx = self
            .index
            .sample_offsets
            .partition_point(|&offset| offset <= idx)
            - 1;

        let entry = &self.entries[font_idx];
        let font_start = self.index.sample_offsets[font_idx];
        let sample_idx = idx - font_start;
        let cp_count = entry.codepoint_count();
        debug_assert!(
            cp_count > 0,
            "font '{}' has no indexed code points",
            entry.path().display()
        );

        let inst_start = self.index.inst_offsets[font_idx];
        let inst_idx = sample_idx / cp_count;
        debug_assert!(
            inst_idx < entry.instance_count(),
            "instance index {} out of range for font '{}'",
            inst_idx,
            entry.path().display()
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
    instance_label: Option<String>,
) -> PyResult<String> {
    let relative_path = font_path.strip_prefix(root).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "font path '{}' is not under dataset root '{}'",
            font_path.display(),
            root.display()
        ))
    })?;
    let quoted_path = relative_path
        .components()
        .map(|component| {
            urlencoding::encode_binary(component.as_os_str().as_encoded_bytes()).into_owned()
        })
        .collect::<Vec<_>>()
        .join("/");
    let instance_value = instance_label.unwrap_or_else(|| "static".to_string());
    Ok(format!(
        "style:path={quoted_path};face={face_idx};instance={}",
        urlencoding::encode(&instance_value),
    ))
}

fn axes_label(axes: &[(String, f32)]) -> String {
    axes.iter()
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
