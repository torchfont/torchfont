use super::merge_curves;
use crate::outline::Command;

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
        for (i, t) in t_seq.iter_mut().enumerate() {
            let base = i * 6;
            let end_x = c_seq[base + 4];
            let end_y = c_seq[base + 5];
            if *t == quad {
                let cx0 = c_seq[base];
                let cy0 = c_seq[base + 1];
                c_seq[base] = prev_x + (2.0 / 3.0) * (cx0 - prev_x);
                c_seq[base + 1] = prev_y + (2.0 / 3.0) * (cy0 - prev_y);
                c_seq[base + 2] = end_x + (2.0 / 3.0) * (cx0 - end_x);
                c_seq[base + 3] = end_y + (2.0 / 3.0) * (cy0 - end_y);
                *t = cubic;
            }
            prev_x = end_x;
            prev_y = end_y;
        }
    }
}

pub(crate) fn quad_to_cubic_and_merge(types: &[i64], coords: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let mut out_types = types.to_vec();
    let mut out_coords = coords.to_vec();
    quad_to_cubic(&mut out_types, &mut out_coords, types.len());
    merge_curves::merge_curves(&out_types, &out_coords)
}
