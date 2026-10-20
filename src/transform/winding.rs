use super::overlap::{overlap_groups, subpath_area};
use crate::outline::BezPath;

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
