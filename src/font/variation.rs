use std::{collections::HashMap, path::Path};

use skrifa::MetadataProvider;

use crate::error::Error;

#[derive(Clone)]
pub(crate) struct AxisInfo {
    pub(crate) tag: String,
    pub(crate) min: f32,
    pub(crate) default: f32,
    pub(crate) max: f32,
}

pub(crate) fn axis_info(font: &skrifa::FontRef<'_>) -> Vec<AxisInfo> {
    font.axes()
        .iter()
        .map(|axis| AxisInfo {
            tag: axis.tag().to_string(),
            min: axis.min_value(),
            default: axis.default_value(),
            max: axis.max_value(),
        })
        .collect()
}

pub(crate) fn default_location(font: &skrifa::FontRef<'_>) -> Vec<(String, f32)> {
    axis_info(font)
        .into_iter()
        .map(|axis| (axis.tag, axis.default))
        .collect()
}

pub(crate) fn named_locations(
    font: &skrifa::FontRef<'_>,
    path: &Path,
    ttc_index: u32,
) -> Result<Vec<Vec<(String, f32)>>, Error> {
    let axes = axis_info(font);
    let axis_tags: Vec<String> = axes.iter().map(|axis| axis.tag.clone()).collect();
    let Some(coords) = named_instance_coords(font, &axis_tags, path, ttc_index)? else {
        return Ok(Vec::new());
    };
    Ok(coords
        .into_iter()
        .map(|values| axis_tags.iter().cloned().zip(values).collect())
        .collect())
}

pub(crate) fn grid_locations(
    font: &skrifa::FontRef<'_>,
    axes: &HashMap<String, i64>,
) -> Result<Vec<Vec<(String, f32)>>, Error> {
    validate_grid_axes(axes)?;

    let axis_info = axis_info(font);
    if axis_info.is_empty() {
        return Ok(vec![Vec::new()]);
    }

    let axis_tags: Vec<String> = axis_info.iter().map(|axis| axis.tag.clone()).collect();
    let axis_points: Vec<Vec<f32>> = axis_info
        .iter()
        .map(|axis| axis_grid_points(axis, axis_sample_count(axis, axes)))
        .collect();
    let total = axis_points.iter().map(Vec::len).product();
    let mut locations = Vec::with_capacity(total);
    for index in 0..total {
        let mut radix: usize = total;
        let mut remainder = index;
        let mut coords = Vec::with_capacity(axis_tags.len());
        for (tag, points) in axis_tags.iter().zip(&axis_points) {
            radix /= points.len();
            let digit = remainder / radix;
            remainder %= radix;
            coords.push((tag.clone(), points[digit]));
        }
        locations.push(coords);
    }
    Ok(locations)
}

pub(crate) fn grid_location_count(
    font: &skrifa::FontRef<'_>,
    axes: &HashMap<String, i64>,
) -> Result<usize, Error> {
    validate_grid_axes(axes)?;
    axis_info(font)
        .iter()
        .map(|axis| usize::try_from(axis_sample_count(axis, axes)).expect("count is positive"))
        .try_fold(1usize, |total, count| {
            total.checked_mul(count).ok_or_else(|| {
                Error::Parse("grid_instances location count overflowed usize".to_string())
            })
        })
}

pub(crate) fn canonicalize_location(
    font: &skrifa::FontRef<'_>,
    path: &Path,
    ttc_index: u32,
    location: Option<&HashMap<String, f32>>,
) -> Result<Vec<(String, f32)>, Error> {
    let axis_info = axis_info(font);
    let Some(location) = location else {
        return Ok(axis_info
            .into_iter()
            .map(|axis| (axis.tag, axis.default))
            .collect());
    };

    for (tag, value) in location {
        let Some(axis) = axis_info.iter().find(|axis| axis.tag == *tag) else {
            return Err(Error::Parse(format!(
                "font '{}' (ttc_index {ttc_index}) has no variation axis '{tag}'",
                path.display(),
            )));
        };
        if !value.is_finite() {
            return Err(Error::Parse(format!(
                "variation axis '{tag}' value must be finite"
            )));
        }
        if *value < axis.min || *value > axis.max {
            return Err(Error::Parse(format!(
                "variation axis '{tag}' value {value} is outside [{}, {}]",
                axis.min, axis.max,
            )));
        }
    }

    Ok(axis_info
        .into_iter()
        .map(|axis| {
            let value = location.get(&axis.tag).copied().unwrap_or(axis.default);
            (axis.tag, value)
        })
        .collect())
}

fn validate_grid_axes(axes: &HashMap<String, i64>) -> Result<(), Error> {
    if axes.is_empty() {
        return Err(Error::Parse(
            "grid_instances requires at least one axis; use default_instance for defaults"
                .to_string(),
        ));
    }
    if let Some((tag, count)) = axes.iter().find(|&(_, &count)| count <= 0) {
        return Err(Error::Parse(format!(
            "grid_instances axis density for '{tag}' must be greater than zero, got {count}"
        )));
    }
    Ok(())
}

fn named_instance_coords(
    font: &skrifa::FontRef<'_>,
    axis_tags: &[String],
    path: &Path,
    ttc_index: u32,
) -> Result<Option<Vec<Vec<f32>>>, Error> {
    let named_instances = font.named_instances();
    if named_instances.is_empty() {
        return Ok(None);
    }
    let instances = named_instances
        .iter()
        .map(|inst| {
            let coords: Vec<f32> = inst.user_coords().collect();
            if coords.len() != axis_tags.len() {
                return Err(Error::Parse(format!(
                    "font '{}' (ttc_index {ttc_index}) reported mismatched axis metadata",
                    path.display(),
                )));
            }
            Ok(coords)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(Some(dedup_coords(instances)))
}

fn dedup_coords(coords: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut unique = Vec::new();
    for coord in coords {
        if !unique.contains(&coord) {
            unique.push(coord);
        }
    }
    unique
}

/// Number of grid points for an axis: the explicitly requested count, or `1`
/// (pinned to the axis default) for axes the caller did not list.
fn axis_sample_count(axis: &AxisInfo, axes: &HashMap<String, i64>) -> i64 {
    axes.get(&axis.tag).copied().unwrap_or(1)
}

fn axis_grid_points(axis: &AxisInfo, count: i64) -> Vec<f32> {
    match count {
        1 => vec![axis.default],
        n => (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                lerp(axis.min, axis.max, t)
            })
            .collect(),
    }
}

fn lerp(min: f32, max: f32, t: f32) -> f32 {
    min + (max - min) * t
}

#[cfg(test)]
mod tests {
    use super::dedup_coords;

    #[test]
    fn dedup_coords_removes_duplicates() {
        let coords = vec![
            vec![100.0, 400.0],
            vec![100.0, 700.0],
            vec![100.0, 400.0],
            vec![75.0, 400.0],
            vec![100.0, 700.0],
        ];

        assert_eq!(
            dedup_coords(coords),
            vec![vec![100.0, 400.0], vec![100.0, 700.0], vec![75.0, 400.0],],
        );
    }
}
