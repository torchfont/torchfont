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
    let mut reverse = vec![false; paths.len()];
    for (index, group) in groups.iter().enumerate() {
        let clockwise = clockwise[if clockwise.len() == 1 { 0 } else { index }];
        let area = group_area(group.iter().map(|&index| areas[index]));
        let selected = area != 0.0 && (area < 0.0) != clockwise;
        if selected {
            for &index in group {
                reverse[index] = true;
            }
        }
    }
    Ok(paths
        .into_iter()
        .zip(reverse)
        .flat_map(|(path, reverse)| {
            let path = BezPath::from_vec(path.to_vec());
            if reverse {
                path.reverse_subpaths()
            } else {
                path
            }
        })
        .collect())
}

fn group_area(areas: impl Iterator<Item = (f64, f64)>) -> f64 {
    let mut total = 0.0;
    let mut error = 0.0;
    let mut magnitude = 0.0;
    let mut count = 0;
    for (area, area_error) in areas {
        total += area;
        magnitude += area.abs();
        error += area_error;
        count += 1;
    }
    if total.abs() > error + f64::EPSILON * magnitude * f64::from(count) {
        total
    } else {
        0.0
    }
}

pub(crate) fn winding_from_even_odd(outline: &BezPath) -> BezPath {
    let mut contours: Vec<_> = outline
        .subpaths()
        .map(|subpath| {
            let (area, error) = subpath_area(subpath);
            (if area.abs() <= error { 0.0 } else { area }, subpath)
        })
        .collect();
    if let [(area, subpath)] = contours.as_slice() {
        return if *area < 0.0 {
            BezPath::from_vec(subpath.to_vec()).reverse_subpaths()
        } else {
            outline.clone()
        };
    }
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
                        build_skia_path(contours[outer].1).map(|mut path| {
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
        let is_clockwise = area < 0.0;
        let is_outer = depth.is_multiple_of(2);
        if is_clockwise == is_outer {
            result.extend(
                BezPath::from_vec(subpath.to_vec())
                    .reverse_subpaths()
                    .elements()
                    .iter()
                    .copied(),
            );
        } else {
            result.extend(subpath.iter().copied());
        }
    }
    result
}
