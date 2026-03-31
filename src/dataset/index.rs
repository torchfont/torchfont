use super::entry::FontEntry;
use crate::error::py_index_err;
use pyo3::prelude::*;

pub(super) struct DatasetIndex {
    pub(super) sample_offsets: Vec<usize>,
    pub(super) inst_offsets: Vec<usize>,
    pub(super) content_classes: Vec<u32>,
}

impl DatasetIndex {
    pub(super) fn content_index(&self, codepoint: u32) -> PyResult<usize> {
        self.content_classes
            .binary_search(&codepoint)
            .map_err(|_| py_index_err(format!("codepoint U+{codepoint:04X} missing from index")))
    }
}

pub(super) fn load_entries_and_index(
    files: Vec<String>,
    filter: Option<&[u32]>,
) -> PyResult<(Vec<FontEntry>, DatasetIndex)> {
    let mut entries = Vec::new();
    let mut all_cps = Vec::new();

    for path in files {
        let faces = FontEntry::load_faces(&path, filter)?;
        for entry in faces
            .into_iter()
            .filter(|entry| entry.codepoint_count() > 0)
        {
            all_cps.extend(entry.codepoints().iter().copied());
            entries.push(entry);
        }
    }

    let sample_offsets = std::iter::once(0)
        .chain(
            entries
                .iter()
                .map(|entry| entry.codepoint_count() * entry.instance_count()),
        )
        .scan(0usize, |total, delta| {
            *total += delta;
            Some(*total)
        })
        .collect();

    let inst_offsets = std::iter::once(0)
        .chain(entries.iter().map(FontEntry::instance_count))
        .scan(0usize, |total, delta| {
            *total += delta;
            Some(*total)
        })
        .collect();

    let mut content_classes = all_cps;
    content_classes.sort_unstable();
    content_classes.dedup();

    let index = DatasetIndex {
        sample_offsets,
        inst_offsets,
        content_classes,
    };

    Ok((entries, index))
}
