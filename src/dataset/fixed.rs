use std::path::{Path, PathBuf};

use crate::font::Location;

use super::IndexOverflow;
use super::classes::{character_index, style_name};
use super::counts::{checked_add_samples, checked_add_styles};
use super::targets::expand_character_targets;

pub(crate) struct FixedFontEntry {
    pub(crate) path: PathBuf,
    pub(crate) ttc_index: u32,
    pub(crate) codepoints: Vec<u32>,
    pub(crate) locations: Vec<Location>,
    pub(crate) family_name: String,
    pub(crate) subfamily_name: String,
}

pub(crate) struct FixedIndex {
    fonts: Vec<FixedFontEntry>,
    sample_starts: Vec<usize>,
    style_starts: Vec<usize>,
    sample_count: usize,
    style_count: usize,
    character_codepoints: Vec<u32>,
}

pub(crate) struct FixedSample<'a> {
    pub(crate) path: &'a Path,
    pub(crate) ttc_index: u32,
    pub(crate) font_idx: usize,
    pub(crate) codepoint: u32,
    pub(crate) location: &'a Location,
    pub(crate) style_idx: usize,
    pub(crate) character_idx: usize,
}

impl FixedIndex {
    pub(crate) fn new(fonts: Vec<FixedFontEntry>) -> Result<Self, IndexOverflow> {
        let mut sample_starts = Vec::with_capacity(fonts.len());
        let mut style_starts = Vec::with_capacity(fonts.len());
        let mut sample_count = 0usize;
        let mut style_count = 0usize;
        for font in &fonts {
            sample_starts.push(sample_count);
            style_starts.push(style_count);
            sample_count =
                checked_add_samples(sample_count, font.codepoints.len(), font.locations.len())?;
            style_count = checked_add_styles(style_count, font.locations.len())?;
        }
        let character_codepoints = character_index(&fonts, |font| &font.codepoints);
        Ok(Self {
            fonts,
            sample_starts,
            style_starts,
            sample_count,
            style_count,
            character_codepoints,
        })
    }

    pub(crate) fn fonts(&self) -> &[FixedFontEntry] {
        &self.fonts
    }

    pub(crate) fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub(crate) fn style_count(&self) -> usize {
        self.style_count
    }

    pub(crate) fn character_codepoints(&self) -> &[u32] {
        &self.character_codepoints
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

    pub(crate) fn locate(&self, idx: usize) -> Option<FixedSample<'_>> {
        if idx >= self.sample_count {
            return None;
        }
        let font_idx = self.sample_starts.partition_point(|&start| start <= idx) - 1;
        let font = &self.fonts[font_idx];
        let local = idx - self.sample_starts[font_idx];
        let location_idx = local / font.codepoints.len();
        let codepoint = font.codepoints[local % font.codepoints.len()];
        Some(FixedSample {
            path: &font.path,
            ttc_index: font.ttc_index,
            font_idx,
            codepoint,
            location: &font.locations[location_idx],
            style_idx: self.style_starts[font_idx] + location_idx,
            character_idx: self.character_index(codepoint),
        })
    }

    pub(crate) fn font_targets(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.sample_count);
        for (font_idx, font) in self.fonts.iter().enumerate() {
            out.extend(std::iter::repeat_n(
                font_idx as i64,
                font.codepoints.len() * font.locations.len(),
            ));
        }
        out
    }

    pub(crate) fn style_targets(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.sample_count);
        for (font, &style_start) in self.fonts.iter().zip(&self.style_starts) {
            for location_idx in 0..font.locations.len() {
                out.extend(std::iter::repeat_n(
                    (style_start + location_idx) as i64,
                    font.codepoints.len(),
                ));
            }
        }
        out
    }

    pub(crate) fn character_targets(&self) -> Vec<i64> {
        expand_character_targets(
            &self.fonts,
            self.sample_count,
            |font| &font.codepoints,
            |font| font.locations.len(),
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

    use super::{FixedFontEntry, FixedIndex};

    fn index() -> FixedIndex {
        FixedIndex::new(vec![
            FixedFontEntry {
                path: PathBuf::from("a.ttf"),
                ttc_index: 0,
                codepoints: vec![65, 66],
                locations: vec![vec![], vec![("wght".to_string(), 700.0)]],
                family_name: "A".to_string(),
                subfamily_name: "Regular".to_string(),
            },
            FixedFontEntry {
                path: PathBuf::from("b.ttf"),
                ttc_index: 1,
                codepoints: vec![66, 67],
                locations: vec![vec![]],
                family_name: "B".to_string(),
                subfamily_name: "Bold".to_string(),
            },
        ])
        .unwrap()
    }

    #[test]
    fn expands_samples_and_targets_in_font_location_codepoint_order() {
        let index = index();
        assert_eq!(index.sample_count(), 6);
        assert_eq!(index.style_count(), 3);
        assert_eq!(index.character_codepoints(), &[65, 66, 67]);
        assert_eq!(index.font_targets(), vec![0, 0, 0, 0, 1, 1]);
        assert_eq!(index.style_targets(), vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(index.character_targets(), vec![0, 1, 0, 1, 1, 2]);
        let sample = index.locate(3).unwrap();
        assert_eq!(
            (sample.codepoint, sample.style_idx, sample.character_idx),
            (66, 1, 1)
        );
    }

    #[test]
    fn handles_empty_index_and_out_of_range() {
        let index = FixedIndex::new(vec![]).unwrap();
        assert_eq!(index.sample_count(), 0);
        assert!(index.locate(0).is_none());
        assert!(index.font_targets().is_empty());
    }
}
