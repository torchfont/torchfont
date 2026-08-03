use std::collections::BTreeSet;

pub(super) fn character_index<T>(fonts: &[T], codepoints: impl Fn(&T) -> &[u32]) -> Vec<u32> {
    fonts
        .iter()
        .flat_map(|font| codepoints(font).iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
