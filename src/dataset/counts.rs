use super::IndexOverflow;

pub(super) fn checked_add_samples(
    total: usize,
    item_count: usize,
    repetition_count: usize,
) -> Result<usize, IndexOverflow> {
    let count = item_count
        .checked_mul(repetition_count)
        .ok_or(IndexOverflow::SampleCount)?;
    total.checked_add(count).ok_or(IndexOverflow::SampleCount)
}

pub(super) fn checked_add_styles(total: usize, style_count: usize) -> Result<usize, IndexOverflow> {
    total
        .checked_add(style_count)
        .ok_or(IndexOverflow::StyleCount)
}

#[cfg(test)]
mod tests {
    use super::{checked_add_samples, checked_add_styles};
    use crate::dataset::IndexOverflow;

    #[test]
    fn rejects_sample_count_multiplication_overflow() {
        assert_eq!(
            checked_add_samples(0, usize::MAX, 2),
            Err(IndexOverflow::SampleCount),
        );
    }

    #[test]
    fn rejects_count_addition_overflow() {
        assert_eq!(
            checked_add_samples(usize::MAX, 1, 1),
            Err(IndexOverflow::SampleCount),
        );
        assert_eq!(
            checked_add_styles(usize::MAX, 1),
            Err(IndexOverflow::StyleCount),
        );
    }
}
