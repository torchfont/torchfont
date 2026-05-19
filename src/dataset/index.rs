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
        for entry in FontEntry::load_faces(&path, filter)?
            .into_iter()
            .filter(|e| e.codepoint_count() > 0)
        {
            all_cps.extend(entry.codepoints().iter().copied());
            entries.push(entry);
        }
    }

    let sample_offsets = cumulative_sums(
        entries
            .iter()
            .map(|e| e.codepoint_count() * e.instance_count()),
    );
    let inst_offsets = cumulative_sums(entries.iter().map(FontEntry::instance_count));

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

fn cumulative_sums(deltas: impl Iterator<Item = usize>) -> Vec<usize> {
    std::iter::once(0)
        .chain(deltas)
        .scan(0usize, |acc, d| {
            *acc += d;
            Some(*acc)
        })
        .collect()
}
