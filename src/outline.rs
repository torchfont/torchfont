use crate::pen::Command;

pub(crate) fn quad_to_cubic(types: &mut [i64], coords: &mut [f32], seq_len: usize) {
    debug_assert_eq!(types.len() * 6, coords.len());
    if seq_len == 0 {
        return;
    }
    let quad = Command::QuadTo as i64;
    let cubic = Command::CurveTo as i64;

    for (t_seq, c_seq) in types
        .chunks_mut(seq_len)
        .zip(coords.chunks_mut(seq_len * 6))
    {
        let mut prev_x = 0.0f32;
        let mut prev_y = 0.0f32;
        for i in 0..t_seq.len() {
            let base = i * 6;
            let end_x = c_seq[base + 4];
            let end_y = c_seq[base + 5];
            if t_seq[i] == quad {
                let cx0 = c_seq[base];
                let cy0 = c_seq[base + 1];
                c_seq[base] = prev_x + (2.0 / 3.0) * (cx0 - prev_x);
                c_seq[base + 1] = prev_y + (2.0 / 3.0) * (cy0 - prev_y);
                c_seq[base + 2] = end_x + (2.0 / 3.0) * (cx0 - end_x);
                c_seq[base + 3] = end_y + (2.0 / 3.0) * (cy0 - end_y);
                t_seq[i] = cubic;
            }
            prev_x = end_x;
            prev_y = end_y;
        }
    }
}
