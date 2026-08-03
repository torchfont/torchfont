use pyo3::PyResult;

use crate::dataset::{DiscoveredFont, FontEntry, canonicalize_root, discover_font_files};

pub(super) fn build_entries(
    root: &str,
    codepoints: Option<Vec<u32>>,
    patterns: Option<Vec<String>>,
) -> PyResult<Vec<FontEntry>> {
    Ok(discover_fonts(root, codepoints, patterns)?
        .into_iter()
        .map(|font| FontEntry {
            path: font.path().to_path_buf(),
            ttc_index: font.ttc_index(),
            codepoints: font.codepoints().to_vec(),
        })
        .collect())
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
