use std::path::{Path, PathBuf};

use super::{IndexOverflow, classes::character_index};

pub(crate) struct FontEntry {
    pub(crate) path: PathBuf,
    pub(crate) ttc_index: u32,
    pub(crate) codepoints: Vec<u32>,
}

pub(crate) struct GlyphIndex {
    fonts: Vec<FontEntry>,
    sample_starts: Vec<usize>,
    sample_count: usize,
    character_codepoints: Vec<u32>,
}

pub(crate) struct GlyphSample<'a> {
    pub(crate) path: &'a Path,
    pub(crate) ttc_index: u32,
    pub(crate) font_idx: usize,
    pub(crate) codepoint: u32,
    pub(crate) character_idx: usize,
}

impl GlyphIndex {
    pub(crate) fn new(fonts: Vec<FontEntry>) -> Result<Self, IndexOverflow> {
        let mut sample_starts = Vec::with_capacity(fonts.len());
        let mut sample_count = 0usize;
        for font in &fonts {
            sample_starts.push(sample_count);
            sample_count = sample_count
                .checked_add(font.codepoints.len())
                .ok_or(IndexOverflow::SampleCount)?;
        }
        let character_codepoints = character_index(&fonts, |font| &font.codepoints);
        Ok(Self {
            fonts,
            sample_starts,
            sample_count,
            character_codepoints,
        })
    }

    pub(crate) fn fonts(&self) -> &[FontEntry] {
        &self.fonts
    }

    pub(crate) fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub(crate) fn character_codepoints(&self) -> &[u32] {
        &self.character_codepoints
    }

    pub(crate) fn locate(&self, idx: usize) -> Option<GlyphSample<'_>> {
        if idx >= self.sample_count {
            return None;
        }
        let font_idx = self.sample_starts.partition_point(|&start| start <= idx) - 1;
        let font = &self.fonts[font_idx];
        let codepoint = font.codepoints[idx - self.sample_starts[font_idx]];
        Some(GlyphSample {
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
            out.extend(std::iter::repeat_n(font_idx as i64, font.codepoints.len()));
        }
        out
    }

    pub(crate) fn character_targets(&self) -> Vec<i64> {
        self.fonts
            .iter()
            .flat_map(|font| font.codepoints.iter())
            .map(|&codepoint| self.character_index(codepoint) as i64)
            .collect()
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

    use super::{FontEntry, GlyphIndex};

    #[test]
    fn indexes_each_face_codepoint_once() {
        let index = GlyphIndex::new(vec![
            FontEntry {
                path: PathBuf::from("a.ttf"),
                ttc_index: 0,
                codepoints: vec![65, 67],
            },
            FontEntry {
                path: PathBuf::from("b.ttf"),
                ttc_index: 1,
                codepoints: vec![66],
            },
        ])
        .unwrap();
        assert_eq!(index.sample_count(), 3);
        assert_eq!(index.character_codepoints(), &[65, 66, 67]);
        assert_eq!(index.font_targets(), vec![0, 0, 1]);
        assert_eq!(index.character_targets(), vec![0, 2, 1]);
        assert_eq!(index.locate(2).unwrap().codepoint, 66);
        assert!(index.locate(3).is_none());
    }
}
