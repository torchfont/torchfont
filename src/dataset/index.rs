use std::path::PathBuf;

/// Flat sample index over discovered font faces.
///
/// Samples are laid out face by face: face `i` owns the half-open sample range
/// `offsets[i]..offsets[i + 1]`, so `offsets` has one more element than `fonts`
/// and its last element is the total sample count. The codepoint of sample `s`
/// is `character_codepoints[character_index[s]]`.
pub(crate) struct FontIndex {
    pub(crate) fonts: Vec<(PathBuf, u32)>,
    pub(crate) offsets: Vec<i64>,
    pub(crate) character_codepoints: Vec<u32>,
    pub(crate) character_index: Vec<u32>,
}

impl FontIndex {
    /// Builds an index from faces whose codepoints are sorted in ascending order.
    pub(crate) fn build(faces: Vec<(PathBuf, u32, Vec<u32>)>) -> Self {
        let mut fonts = Vec::with_capacity(faces.len());
        let mut offsets = Vec::with_capacity(faces.len() + 1);
        let mut codepoints = Vec::with_capacity(faces.iter().map(|face| face.2.len()).sum());
        offsets.push(0);
        for (path, ttc_index, face_codepoints) in faces {
            debug_assert!(face_codepoints.is_sorted());
            fonts.push((path, ttc_index));
            codepoints.extend(face_codepoints);
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
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::FontIndex;

    #[test]
    fn indexes_each_face_codepoint_once() {
        let index = FontIndex::build(vec![
            (PathBuf::from("a.ttf"), 0, vec![65, 67]),
            (PathBuf::from("b.ttf"), 1, vec![66]),
        ]);
        assert_eq!(
            index.fonts,
            vec![(PathBuf::from("a.ttf"), 0), (PathBuf::from("b.ttf"), 1)]
        );
        assert_eq!(index.offsets, vec![0, 2, 3]);
        assert_eq!(index.character_codepoints, vec![65, 66, 67]);
        assert_eq!(index.character_index, vec![0, 2, 1]);
    }

    #[test]
    fn indexes_an_empty_collection() {
        let index = FontIndex::build(Vec::new());
        assert!(index.fonts.is_empty());
        assert_eq!(index.offsets, vec![0]);
        assert!(index.character_codepoints.is_empty());
        assert!(index.character_index.is_empty());
    }
}
