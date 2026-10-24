use std::path::PathBuf;

pub(crate) struct IndexedCodepointFace {
    pub(crate) path: PathBuf,
    pub(crate) face_index: u32,
    pub(crate) codepoints: Vec<u32>,
    pub(crate) glyph_ids: Vec<u32>,
    pub(crate) outline_lengths: Vec<u32>,
}

pub(crate) struct IndexedGlyphFace {
    pub(crate) path: PathBuf,
    pub(crate) face_index: u32,
    pub(crate) glyph_ids: Vec<u32>,
    pub(crate) outline_lengths: Vec<u32>,
}

pub(crate) struct CodepointIndex {
    pub(crate) fonts: Vec<(PathBuf, u32)>,
    pub(crate) offsets: Vec<i64>,
    pub(crate) character_codepoints: Vec<u32>,
    pub(crate) character_index: Vec<u32>,
    pub(crate) glyph_ids: Vec<u32>,
    pub(crate) outline_lengths: Vec<u32>,
}

pub(crate) struct GlyphIndex {
    pub(crate) fonts: Vec<(PathBuf, u32)>,
    pub(crate) offsets: Vec<i64>,
    pub(crate) glyph_ids: Vec<u32>,
    pub(crate) outline_lengths: Vec<u32>,
}

impl CodepointIndex {
    pub(crate) fn build(faces: Vec<IndexedCodepointFace>) -> Self {
        let mut fonts = Vec::with_capacity(faces.len());
        let mut offsets = Vec::with_capacity(faces.len() + 1);
        let sample_count = faces.iter().map(|face| face.codepoints.len()).sum();
        let mut codepoints = Vec::with_capacity(sample_count);
        let mut glyph_ids = Vec::with_capacity(sample_count);
        let mut outline_lengths = Vec::with_capacity(sample_count);
        offsets.push(0);
        for face in faces {
            debug_assert!(face.codepoints.is_sorted());
            debug_assert_eq!(face.codepoints.len(), face.glyph_ids.len());
            debug_assert_eq!(face.codepoints.len(), face.outline_lengths.len());
            fonts.push((face.path, face.face_index));
            codepoints.extend(face.codepoints);
            glyph_ids.extend(face.glyph_ids);
            outline_lengths.extend(face.outline_lengths);
            offsets.push(i64::try_from(codepoints.len()).expect("Vec length fits in i64"));
        }
        let table_len = codepoints
            .iter()
            .copied()
            .max()
            .map_or(0, |highest| highest as usize + 1);
        let mut ranks = vec![u32::MAX; table_len];
        for &codepoint in &codepoints {
            ranks[codepoint as usize] = 0;
        }
        let mut character_codepoints = Vec::new();
        for (codepoint, rank) in ranks.iter_mut().enumerate() {
            if *rank != u32::MAX {
                *rank = u32::try_from(character_codepoints.len())
                    .expect("unique u32 codepoint index fits in u32");
                character_codepoints.push(codepoint as u32);
            }
        }
        let character_index: Vec<u32> = codepoints
            .iter()
            .map(|&codepoint| ranks[codepoint as usize])
            .collect();
        Self {
            fonts,
            offsets,
            character_codepoints,
            character_index,
            glyph_ids,
            outline_lengths,
        }
    }
}

impl GlyphIndex {
    pub(crate) fn build(faces: Vec<IndexedGlyphFace>) -> Self {
        let mut fonts = Vec::with_capacity(faces.len());
        let mut offsets = Vec::with_capacity(faces.len() + 1);
        let mut glyph_ids = Vec::with_capacity(faces.iter().map(|face| face.glyph_ids.len()).sum());
        let mut outline_lengths = Vec::with_capacity(glyph_ids.capacity());
        offsets.push(0);
        for face in faces {
            debug_assert!(face.glyph_ids.is_sorted());
            debug_assert_eq!(face.glyph_ids.len(), face.outline_lengths.len());
            fonts.push((face.path, face.face_index));
            glyph_ids.extend(face.glyph_ids);
            outline_lengths.extend(face.outline_lengths);
            offsets.push(i64::try_from(glyph_ids.len()).expect("Vec length fits in i64"));
        }
        Self {
            fonts,
            offsets,
            glyph_ids,
            outline_lengths,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{CodepointIndex, GlyphIndex, IndexedCodepointFace, IndexedGlyphFace};

    #[test]
    fn indexes_each_face_codepoint_once() {
        let index = CodepointIndex::build(vec![
            IndexedCodepointFace {
                path: PathBuf::from("a.ttf"),
                face_index: 0,
                codepoints: vec![65, 67],
                glyph_ids: vec![36, 38],
                outline_lengths: vec![9, 7],
            },
            IndexedCodepointFace {
                path: PathBuf::from("b.ttf"),
                face_index: 1,
                codepoints: vec![66],
                glyph_ids: vec![5],
                outline_lengths: vec![4],
            },
        ]);
        assert_eq!(
            index.fonts,
            vec![(PathBuf::from("a.ttf"), 0), (PathBuf::from("b.ttf"), 1)]
        );
        assert_eq!(index.offsets, vec![0, 2, 3]);
        assert_eq!(index.character_codepoints, vec![65, 66, 67]);
        assert_eq!(index.character_index, vec![0, 2, 1]);
        assert_eq!(index.glyph_ids, vec![36, 38, 5]);
        assert_eq!(index.outline_lengths, vec![9, 7, 4]);
    }

    #[test]
    fn indexes_an_empty_collection() {
        let index = CodepointIndex::build(Vec::new());
        assert!(index.fonts.is_empty());
        assert_eq!(index.offsets, vec![0]);
        assert!(index.character_codepoints.is_empty());
        assert!(index.character_index.is_empty());
        assert!(index.glyph_ids.is_empty());
        assert!(index.outline_lengths.is_empty());
    }

    #[test]
    fn indexes_each_face_glyph_once() {
        let index = GlyphIndex::build(vec![
            IndexedGlyphFace {
                path: PathBuf::from("a.ttf"),
                face_index: 0,
                glyph_ids: vec![0, 1, 2],
                outline_lengths: vec![2, 4, 6],
            },
            IndexedGlyphFace {
                path: PathBuf::from("b.ttf"),
                face_index: 1,
                glyph_ids: vec![0, 1],
                outline_lengths: vec![3, 5],
            },
        ]);
        assert_eq!(
            index.fonts,
            vec![(PathBuf::from("a.ttf"), 0), (PathBuf::from("b.ttf"), 1)]
        );
        assert_eq!(index.offsets, vec![0, 3, 5]);
        assert_eq!(index.glyph_ids, vec![0, 1, 2, 0, 1]);
        assert_eq!(index.outline_lengths, vec![2, 4, 6, 3, 5]);
    }

    #[test]
    fn indexes_an_empty_glyph_collection() {
        let index = GlyphIndex::build(Vec::new());
        assert!(index.fonts.is_empty());
        assert_eq!(index.offsets, vec![0]);
        assert!(index.glyph_ids.is_empty());
        assert!(index.outline_lengths.is_empty());
    }
}
