use super::merge_curves;
use crate::outline::{ElementType, Outline, Point};

pub(crate) fn quad_to_cubic(types: &mut [i64], coords: &mut [f32], seq_len: usize) {
    debug_assert_eq!(types.len() * 6, coords.len());
    if seq_len == 0 {
        return;
    }
    let quad = ElementType::QuadTo as i64;
    let cubic = ElementType::CurveTo as i64;

    for (t_seq, c_seq) in types
        .chunks_mut(seq_len)
        .zip(coords.chunks_mut(seq_len * 6))
    {
        let mut prev = Point::default();
        for (i, t) in t_seq.iter_mut().enumerate() {
            let base = i * 6;
            let end = Point::new(c_seq[base + 4], c_seq[base + 5]);
            if *t == quad {
                let ctrl = Point::new(c_seq[base], c_seq[base + 1]);
                let c1 = prev.lerp(ctrl, 2.0 / 3.0);
                let c2 = end.lerp(ctrl, 2.0 / 3.0);
                c_seq[base] = c1.x;
                c_seq[base + 1] = c1.y;
                c_seq[base + 2] = c2.x;
                c_seq[base + 3] = c2.y;
                *t = cubic;
            }
            prev = end;
        }
    }
}

pub(crate) fn quad_to_cubic_and_merge(types: &[i64], coords: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let mut out_types = types.to_vec();
    let mut out_coords = coords.to_vec();
    quad_to_cubic(&mut out_types, &mut out_coords, types.len());
    let outline = Outline::decode(&out_types, &out_coords);
    merge_curves::merge_curves(&outline).encode()
}
