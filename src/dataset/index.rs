use std::path::PathBuf;

/// One discovered face: its file, TTC index, and the codepoint/glyph id pairs
/// it contributes, sorted by codepoint.
pub(crate) type IndexedCodepointFace = (PathBuf, u32, Vec<u32>, Vec<u32>);

/// One discovered face: its file, TTC index, and the glyph ids it contributes,
/// sorted in ascending order.
pub(crate) type IndexedGlyphFace = (PathBuf, u32, Vec<u32>);

/// Flat sample index over the `cmap` entries of discovered font faces.
///
/// Samples are laid out face by face: face `i` owns the half-open sample range
/// `offsets[i]..offsets[i + 1]`, so `offsets` has one more element than `fonts`
/// and its last element is the total sample count. Sample `s` draws glyph
/// `glyph_ids[s]`, whose codepoint is `character_codepoints[character_index[s]]`.
pub(crate) struct CodepointIndex {
    pub(crate) fonts: Vec<(PathBuf, u32)>,
    pub(crate) offsets: Vec<i64>,
    pub(crate) character_codepoints: Vec<u32>,
    pub(crate) character_index: Vec<u32>,
    pub(crate) glyph_ids: Vec<u32>,
}

/// Flat sample index over every outline glyph of discovered font faces.
///
/// Samples are laid out face by face like [`CodepointIndex`], but sample `s` names
/// glyph `glyph_ids[s]` directly, including glyphs no codepoint maps to.
pub(crate) struct GlyphIndex {
    pub(crate) fonts: Vec<(PathBuf, u32)>,
    pub(crate) offsets: Vec<i64>,
    pub(crate) glyph_ids: Vec<u32>,
}

impl CodepointIndex {
    /// Builds an index from faces whose codepoints are sorted in ascending order.
    pub(crate) fn build(faces: Vec<IndexedCodepointFace>) -> Self {
        let mut fonts = Vec::with_capacity(faces.len());
        let mut offsets = Vec::with_capacity(faces.len() + 1);
        let sample_count = faces.iter().map(|face| face.2.len()).sum();
        let mut codepoints = Vec::with_capacity(sample_count);
        let mut glyph_ids = Vec::with_capacity(sample_count);
        offsets.push(0);
        for (path, ttc_index, face_codepoints, face_glyph_ids) in faces {
            debug_assert!(face_codepoints.is_sorted());
            debug_assert_eq!(face_codepoints.len(), face_glyph_ids.len());
            fonts.push((path, ttc_index));
            codepoints.extend(face_codepoints);
            glyph_ids.extend(face_glyph_ids);
            offsets.push(i64::try_from(codepoints.len()).expect("Vec length fits in i64"));
        }
        let mut character_codepoints = codepoints.clone();
        character_codepoints.sort_unstable();
        character_codepoints.dedup();
        let mut character_index = Vec::with_capacity(codepoints.len());
        for face in offsets.windows(2) {
            let mut base = 0;
            for &codepoint in &codepoints[face[0] as usize..face[1] as usize] {
                base += character_codepoints[base..].partition_point(|&other| other < codepoint);
                character_index
                    .push(u32::try_from(base).expect("unique u32 codepoint index fits in u32"));
            }
        }
        Self {
            fonts,
            offsets,
            character_codepoints,
            character_index,
            glyph_ids,
        }
    }
}

impl GlyphIndex {
    /// Builds an index from faces whose glyph ids are sorted in ascending order.
    pub(crate) fn build(faces: Vec<IndexedGlyphFace>) -> Self {
        let mut fonts = Vec::with_capacity(faces.len());
        let mut offsets = Vec::with_capacity(faces.len() + 1);
        let mut glyph_ids = Vec::with_capacity(faces.iter().map(|face| face.2.len()).sum());
        offsets.push(0);
        for (path, ttc_index, face_glyph_ids) in faces {
            debug_assert!(face_glyph_ids.is_sorted());
            fonts.push((path, ttc_index));
            glyph_ids.extend(face_glyph_ids);
            offsets.push(i64::try_from(glyph_ids.len()).expect("Vec length fits in i64"));
        }
        Self {
            fonts,
            offsets,
            glyph_ids,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{CodepointIndex, GlyphIndex};

    #[test]
    fn indexes_each_face_codepoint_once() {
        let index = CodepointIndex::build(vec![
            (PathBuf::from("a.ttf"), 0, vec![65, 67], vec![36, 38]),
            (PathBuf::from("b.ttf"), 1, vec![66], vec![5]),
        ]);
        assert_eq!(
            index.fonts,
            vec![(PathBuf::from("a.ttf"), 0), (PathBuf::from("b.ttf"), 1)]
        );
        assert_eq!(index.offsets, vec![0, 2, 3]);
        assert_eq!(index.character_codepoints, vec![65, 66, 67]);
        assert_eq!(index.character_index, vec![0, 2, 1]);
        assert_eq!(index.glyph_ids, vec![36, 38, 5]);
    }

    #[test]
    fn indexes_an_empty_collection() {
        let index = CodepointIndex::build(Vec::new());
        assert!(index.fonts.is_empty());
        assert_eq!(index.offsets, vec![0]);
        assert!(index.character_codepoints.is_empty());
        assert!(index.character_index.is_empty());
        assert!(index.glyph_ids.is_empty());
    }

    #[test]
    fn indexes_each_face_glyph_once() {
        let index = GlyphIndex::build(vec![
            (PathBuf::from("a.ttf"), 0, vec![0, 1, 2]),
            (PathBuf::from("b.ttf"), 1, vec![0, 1]),
        ]);
        assert_eq!(
            index.fonts,
            vec![(PathBuf::from("a.ttf"), 0), (PathBuf::from("b.ttf"), 1)]
        );
        assert_eq!(index.offsets, vec![0, 3, 5]);
        assert_eq!(index.glyph_ids, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn indexes_an_empty_glyph_collection() {
        let index = GlyphIndex::build(Vec::new());
        assert!(index.fonts.is_empty());
        assert_eq!(index.offsets, vec![0]);
        assert!(index.glyph_ids.is_empty());
    }
}
