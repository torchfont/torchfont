use pyo3::{Bound, PyResult, Python, types::PyAny};

use crate::dataset::{
    DiscoveredFont, FixedFontEntry, VariableFontEntry, canonicalize_root, discover_font_files,
};
use crate::font::{Location, map_font, parse_font_ref};
use crate::instance_fn::canonicalize_locations;
use crate::py::instance_fn::callback::InstanceFunctionsBridge;

pub(super) fn build_fixed_entries(
    py: Python<'_>,
    root: &str,
    codepoints: Option<Vec<u32>>,
    patterns: Option<Vec<String>>,
    instance_fn: &Bound<'_, PyAny>,
) -> PyResult<Vec<FixedFontEntry>> {
    let bridge = InstanceFunctionsBridge::new(py)?;
    let mut entries = Vec::new();
    for font in discover_fonts(root, codepoints, patterns)? {
        let locations = instance_locations(&bridge, instance_fn, &font)?;
        if !locations.is_empty() {
            entries.push(FixedFontEntry {
                path: font.path().to_path_buf(),
                ttc_index: font.ttc_index(),
                codepoints: font.codepoints().to_vec(),
                locations,
                family_name: font.family_name().to_string(),
                subfamily_name: font.subfamily_name().to_string(),
            });
        }
    }
    Ok(entries)
}

pub(super) fn build_variable_entries(
    py: Python<'_>,
    root: &str,
    codepoints: Option<Vec<u32>>,
    patterns: Option<Vec<String>>,
    instance_fn: &Bound<'_, PyAny>,
) -> PyResult<Vec<VariableFontEntry>> {
    let bridge = InstanceFunctionsBridge::new(py)?;
    let mut entries = Vec::new();
    for font in discover_fonts(root, codepoints, patterns)? {
        let count = bridge.call_instance_count(instance_fn, font.path(), font.ttc_index())?;
        if count != 0 {
            entries.push(VariableFontEntry {
                path: font.path().to_path_buf(),
                ttc_index: font.ttc_index(),
                codepoints: font.codepoints().to_vec(),
                instance_count: count,
            });
        }
    }
    Ok(entries)
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

fn instance_locations(
    bridge: &InstanceFunctionsBridge<'_>,
    instance_fn: &Bound<'_, PyAny>,
    font: &DiscoveredFont,
) -> PyResult<Vec<Location>> {
    let raw = bridge.call_instance_locations(instance_fn, font.path(), font.ttc_index())?;
    let data = map_font(font.path())?;
    let font_ref = parse_font_ref(&data[..], font.path(), font.ttc_index())?;
    canonicalize_locations(&font_ref, font.path(), font.ttc_index(), &raw).map_err(Into::into)
}
