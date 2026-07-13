use std::path::{Path, PathBuf};

use super::IndexOverflow;
use super::classes::character_index;
use super::counts::checked_add_samples;
use super::targets::expand_character_targets;

pub(crate) struct VariableFontEntry {
    pub(crate) path: PathBuf,
    pub(crate) ttc_index: u32,
    pub(crate) codepoints: Vec<u32>,
    pub(crate) instance_count: usize,
}

pub(crate) struct VariableIndex {
    fonts: Vec<VariableFontEntry>,
    sample_starts: Vec<usize>,
    sample_count: usize,
    character_codepoints: Vec<u32>,
}

pub(crate) struct VariableSample<'a> {
    pub(crate) path: &'a Path,
    pub(crate) ttc_index: u32,
    pub(crate) font_idx: usize,
    pub(crate) codepoint: u32,
    pub(crate) character_idx: usize,
}

impl VariableIndex {
    pub(crate) fn new(fonts: Vec<VariableFontEntry>) -> Result<Self, IndexOverflow> {
        let mut sample_starts = Vec::with_capacity(fonts.len());
        let mut sample_count = 0usize;
        for font in &fonts {
            sample_starts.push(sample_count);
            sample_count =
                checked_add_samples(sample_count, font.codepoints.len(), font.instance_count)?;
        }
        let character_codepoints = character_index(&fonts, |font| &font.codepoints);
        Ok(Self {
            fonts,
            sample_starts,
            sample_count,
            character_codepoints,
        })
    }

    pub(crate) fn fonts(&self) -> &[VariableFontEntry] {
        &self.fonts
    }

    pub(crate) fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub(crate) fn character_codepoints(&self) -> &[u32] {
        &self.character_codepoints
    }

    pub(crate) fn locate(&self, idx: usize) -> Option<VariableSample<'_>> {
        if idx >= self.sample_count {
            return None;
        }
        let font_idx = self.sample_starts.partition_point(|&start| start <= idx) - 1;
        let font = &self.fonts[font_idx];
        let codepoint =
            font.codepoints[(idx - self.sample_starts[font_idx]) % font.codepoints.len()];
        Some(VariableSample {
            path: &font.path,
            ttc_index: font.ttc_index,
            font_idx,
            codepoint,
            character_idx: self.character_index(codepoint),
        })
    }

    pub(crate) fn font_targets(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.sample_count);
        for (font_idx, font) in self.fonts.iter().enumerate() {
            out.extend(std::iter::repeat_n(
                font_idx as i64,
                font.codepoints.len() * font.instance_count,
            ));
        }
        out
    }

    pub(crate) fn character_targets(&self) -> Vec<i64> {
        expand_character_targets(
            &self.fonts,
            self.sample_count,
            |font| &font.codepoints,
            |font| font.instance_count,
            |codepoint| self.character_index(codepoint),
        )
    }

    fn character_index(&self, codepoint: u32) -> usize {
        self.character_codepoints
            .binary_search(&codepoint)
            .expect("character index was built from all codepoints")
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{VariableFontEntry, VariableIndex};

    #[test]
    fn expands_instance_multiplicity_and_targets() {
        let index = VariableIndex::new(vec![
            VariableFontEntry {
                path: PathBuf::from("a.ttf"),
                ttc_index: 0,
                codepoints: vec![65, 67],
                instance_count: 2,
            },
            VariableFontEntry {
                path: PathBuf::from("b.ttf"),
                ttc_index: 0,
                codepoints: vec![66],
                instance_count: 1,
            },
        ])
        .unwrap();
        assert_eq!(index.sample_count(), 5);
        assert_eq!(index.character_codepoints(), &[65, 66, 67]);
        assert_eq!(index.font_targets(), vec![0, 0, 0, 0, 1]);
        assert_eq!(index.character_targets(), vec![0, 2, 0, 2, 1]);
        assert_eq!(index.locate(2).unwrap().codepoint, 65);
        assert!(index.locate(5).is_none());
    }

    #[test]
    fn handles_empty_index() {
        let index = VariableIndex::new(vec![]).unwrap();
        assert_eq!(index.sample_count(), 0);
        assert!(index.locate(0).is_none());
    }
}
