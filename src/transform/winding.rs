use super::overlap::{overlap_groups, subpath_area};
use super::skia::{SKIA_SCALE, build_skia_path};
use crate::outline::{BezPath, PathEl};
use kurbo::Shape;
use skia_safe::PathFillType;

pub(crate) fn normalize_winding(
    outline: &BezPath,
    clockwise: &[bool],
) -> Result<BezPath, &'static str> {
    let paths: Vec<_> = outline.subpaths().collect();
    let areas: Vec<_> = paths.iter().map(|path| subpath_area(path)).collect();
    let groups = overlap_groups(&paths, &areas);
    if clockwise.len() != 1 && clockwise.len() < groups.len() {
        return Err("clockwise must contain at least one value per winding group");
    }
    let mut result: Vec<_> = paths
        .iter()
        .map(|path| BezPath::from_vec(path.to_vec()))
        .collect();
    for (index, group) in groups.iter().enumerate() {
        let clockwise = clockwise[if clockwise.len() == 1 { 0 } else { index }];
        let forward: BezPath = group
            .iter()
            .flat_map(|&index| paths[index].iter().copied())
            .collect();
        let reversed = forward.reverse_subpaths();
        let area: f64 = group.iter().map(|&index| areas[index]).sum();
        let reversed_area: f64 = reversed.subpaths().map(subpath_area).sum();
        if (clockwise && area > reversed_area) || (!clockwise && area < reversed_area) {
            for (&index, path) in group.iter().zip(reversed.subpaths()) {
                result[index] = BezPath::from_vec(path.to_vec());
            }
        }
    }
    Ok(result.into_iter().flatten().collect())
}

pub(crate) fn winding_from_even_odd(outline: &BezPath) -> BezPath {
    let mut contours: Vec<_> = outline
        .subpaths()
        .map(|subpath| (subpath_area(subpath), subpath))
        .collect();
    contours.sort_by(|a, b| {
        b.0.abs()
            .partial_cmp(&a.0.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let bounding_boxes: Vec<_> = contours
        .iter()
        .map(|(_, subpath)| subpath.bounding_box())
        .collect();
    let mut skia_paths = vec![None; contours.len()];

    let mut nesting = vec![0usize; contours.len()];
    for inner in 0..contours.len() {
        for outer in 0..inner {
            let is_inside = bounding_boxes[outer].contains_rect(bounding_boxes[inner])
                && skia_paths[outer]
                    .get_or_insert_with(|| {
                        build_skia_path(contours[outer].1)
                            .try_make_scale((SKIA_SCALE, SKIA_SCALE))
                            .map(|mut path| {
                                path.set_fill_type(PathFillType::EvenOdd);
                                path
                            })
                    })
                    .as_ref()
                    .is_some_and(|path| {
                        contours[inner]
                            .1
                            .iter()
                            .filter_map(PathEl::end_point)
                            .all(|point| {
                                path.contains((
                                    point.x as f32 * SKIA_SCALE,
                                    point.y as f32 * SKIA_SCALE,
                                ))
                            })
                    });
            if is_inside {
                nesting[inner] += 1;
            }
        }
    }

    let mut result = BezPath::new();
    for ((area, subpath), depth) in contours.into_iter().zip(nesting) {
        let reversed = BezPath::from_vec(subpath.to_vec()).reverse_subpaths();
        let reversed_area = subpath_area(reversed.elements());
        let clockwise = !depth.is_multiple_of(2);
        if (clockwise && area > reversed_area) || (!clockwise && area < reversed_area) {
            result.extend(reversed);
        } else {
            result.extend(subpath.iter().copied());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use kurbo::{BezPath, Rect, Shape};

    use super::winding_from_even_odd;

    #[test]
    fn equal_area_directions_preserve_contours() {
        let mut contour = BezPath::new();
        contour.move_to((1.0, 1.0));
        contour.curve_to((2.0, 1.0), (3.0, 1.0), (4.0, 1.0));
        contour.close_path();
        for contour in [contour.clone(), contour.reverse_subpaths()] {
            assert_eq!(winding_from_even_odd(&contour), contour);
            let mut nested = Rect::new(0.0, 0.0, 5.0, 5.0).to_path(0.1);
            nested.extend(contour.iter());
            assert_eq!(winding_from_even_odd(&nested), nested);
        }
    }
}
