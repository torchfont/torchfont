use crate::outline::ElementType;

pub(crate) fn normalize_subpath_start_points(
    types: &[i64],
    coords: &[f32],
) -> (Vec<i64>, Vec<f32>) {
    transform_start_points(types, coords, |move_idx, close_idx| {
        (move_idx..close_idx)
            .min_by(|&a, &b| compare_endpoints(endpoint(coords, a), endpoint(coords, b)))
            .unwrap_or(move_idx)
    })
}

pub(crate) fn randomize_subpath_start_points(
    types: &[i64],
    coords: &[f32],
    random_values: &[f32],
) -> (Vec<i64>, Vec<f32>) {
    transform_start_points(types, coords, |move_idx, close_idx| {
        let node_count = close_idx - move_idx;
        let value = random_values[move_idx].clamp(0.0, 1.0 - f32::EPSILON);
        move_idx + (value * node_count as f32) as usize
    })
}

fn transform_start_points(
    types: &[i64],
    coords: &[f32],
    choose_start: impl Fn(usize, usize) -> usize,
) -> (Vec<i64>, Vec<f32>) {
    let mut out_types = Vec::with_capacity(types.len());
    let mut out_coords = Vec::with_capacity(coords.len());
    let mut cursor = 0;
    let mut i = 0;

    while i < types.len() {
        if types[i] != ElementType::MoveTo as i64 {
            if types[i] == ElementType::End as i64 {
                break;
            }
            i += 1;
            continue;
        }
        let move_idx = i;
        i += 1;
        while i < types.len()
            && types[i] != ElementType::Close as i64
            && types[i] != ElementType::MoveTo as i64
            && types[i] != ElementType::End as i64
        {
            i += 1;
        }
        if i >= types.len() || types[i] != ElementType::Close as i64 {
            continue;
        }
        let close_idx = i;
        append_range(
            &mut out_types,
            &mut out_coords,
            types,
            coords,
            cursor,
            move_idx,
        );
        append_rotated_subpath(
            &mut out_types,
            &mut out_coords,
            types,
            coords,
            move_idx,
            close_idx,
            choose_start(move_idx, close_idx),
        );
        cursor = close_idx + 1;
        i += 1;
    }

    append_range(
        &mut out_types,
        &mut out_coords,
        types,
        coords,
        cursor,
        types.len(),
    );
    (out_types, out_coords)
}

#[allow(clippy::too_many_arguments)]
fn append_rotated_subpath(
    out_types: &mut Vec<i64>,
    out_coords: &mut Vec<f32>,
    types: &[i64],
    coords: &[f32],
    move_idx: usize,
    close_idx: usize,
    start_idx: usize,
) {
    if start_idx == move_idx || close_idx - move_idx < 2 {
        append_range(
            out_types,
            out_coords,
            types,
            coords,
            move_idx,
            close_idx + 1,
        );
        return;
    }

    push(
        out_types,
        out_coords,
        ElementType::MoveTo as i64,
        endpoint_row(coords, start_idx),
    );
    append_range(
        out_types,
        out_coords,
        types,
        coords,
        start_idx + 1,
        close_idx,
    );
    if endpoint(coords, close_idx - 1) != endpoint(coords, move_idx) {
        push(
            out_types,
            out_coords,
            ElementType::LineTo as i64,
            endpoint_row(coords, move_idx),
        );
    }
    append_range(
        out_types,
        out_coords,
        types,
        coords,
        move_idx + 1,
        start_idx + 1,
    );
    append_range(
        out_types,
        out_coords,
        types,
        coords,
        close_idx,
        close_idx + 1,
    );
}

fn compare_endpoints(a: (f32, f32), b: (f32, f32)) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1))
}

fn endpoint(coords: &[f32], idx: usize) -> (f32, f32) {
    (coords[idx * 6 + 4], coords[idx * 6 + 5])
}

fn endpoint_row(coords: &[f32], idx: usize) -> [f32; 6] {
    let (x, y) = endpoint(coords, idx);
    [0.0, 0.0, 0.0, 0.0, x, y]
}

fn append_range(
    out_types: &mut Vec<i64>,
    out_coords: &mut Vec<f32>,
    types: &[i64],
    coords: &[f32],
    start: usize,
    end: usize,
) {
    out_types.extend_from_slice(&types[start..end]);
    out_coords.extend_from_slice(&coords[start * 6..end * 6]);
}

fn push(out_types: &mut Vec<i64>, out_coords: &mut Vec<f32>, element_type: i64, values: [f32; 6]) {
    out_types.push(element_type);
    out_coords.extend_from_slice(&values);
}
