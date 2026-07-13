pub(super) fn expand_character_targets<T>(
    fonts: &[T],
    sample_count: usize,
    codepoints: impl Fn(&T) -> &[u32],
    repetitions: impl Fn(&T) -> usize,
    character_index: impl Fn(u32) -> usize,
) -> Vec<i64> {
    let mut out = Vec::with_capacity(sample_count);
    for font in fonts {
        let indices: Vec<_> = codepoints(font)
            .iter()
            .map(|&codepoint| character_index(codepoint) as i64)
            .collect();
        for _ in 0..repetitions(font) {
            out.extend_from_slice(&indices);
        }
    }
    out
}
