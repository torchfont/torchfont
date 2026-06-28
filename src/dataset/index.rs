use crate::error::Error;
use crate::font::{FontEntry, VariationInstantiation};
use std::path::PathBuf;

pub(crate) struct DatasetIndex {
    pub(crate) sample_offsets: Vec<usize>,
    pub(crate) inst_offsets: Vec<usize>,
    pub(crate) content_classes: Vec<u32>,
}

impl DatasetIndex {
    pub(crate) fn content_index(&self, codepoint: u32) -> Result<usize, Error> {
        self.content_classes.binary_search(&codepoint).map_err(|_| {
            Error::OutOfRange(format!("codepoint U+{codepoint:04X} missing from index"))
        })
    }
}

pub(crate) fn load_entries_and_index(
    files: Vec<PathBuf>,
    filter: Option<&[u32]>,
    variation_instantiation: &VariationInstantiation,
) -> Result<(Vec<FontEntry>, DatasetIndex), Error> {
    let mut entries = Vec::new();
    let mut all_cps = Vec::new();

    for path in files {
        for entry in FontEntry::load_faces(&path, filter, variation_instantiation)?
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

/// Hash everything that determines the sample -> (style, content) mapping:
/// entry order, paths, face indices, code point lists, and instance counts.
pub(crate) fn structure_fingerprint(entries: &[FontEntry]) -> u64 {
    let mut hash = Fnv1a::new();
    hash.write_u64(entries.len() as u64);
    for entry in entries {
        let path = entry.path().as_os_str().as_encoded_bytes();
        hash.write_u64(path.len() as u64);
        hash.write(path);
        hash.write_u32(entry.face_index());
        hash.write_u64(entry.instance_count() as u64);
        hash.write_u64(entry.codepoint_count() as u64);
        for &codepoint in entry.codepoints() {
            hash.write_u32(codepoint);
        }
    }
    hash.finish()
}

/// FNV-1a (64-bit). Fingerprints are compared across processes and library
/// versions, so the hash must stay stable, unlike `std::hash::DefaultHasher`.
struct Fnv1a(u64);

impl Fnv1a {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    fn finish(&self) -> u64 {
        self.0
    }
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
