use std::collections::BTreeMap;

use crate::{
    error::Error,
    font::{AxisInfo, Location, axis_info},
};

pub(crate) fn grid_locations(
    font: &skrifa::FontRef<'_>,
    axes: &BTreeMap<String, i64>,
) -> Result<Vec<Location>, Error> {
    validate_grid_axes(axes)?;
    let info = axis_info(font);
    if info.is_empty() {
        return Ok(vec![Vec::new()]);
    }
    let total = checked_location_count(&info, axes)?;
    let tags: Vec<_> = info.iter().map(|axis| axis.tag.clone()).collect();
    let points: Vec<_> = info
        .iter()
        .map(|axis| axis_grid_points(axis, axis_sample_count(axis, axes)))
        .collect();
    let mut locations = Vec::with_capacity(total);
    for index in 0..total {
        let mut radix = total;
        let mut remainder = index;
        let mut location = Vec::with_capacity(tags.len());
        for (tag, values) in tags.iter().zip(&points) {
            radix /= values.len();
            let digit = remainder / radix;
            remainder %= radix;
            location.push((tag.clone(), values[digit]));
        }
        locations.push(location);
    }
    Ok(locations)
}

pub(crate) fn grid_location_count(
    font: &skrifa::FontRef<'_>,
    axes: &BTreeMap<String, i64>,
) -> Result<usize, Error> {
    validate_grid_axes(axes)?;
    checked_location_count(&axis_info(font), axes)
}

fn validate_grid_axes(axes: &BTreeMap<String, i64>) -> Result<(), Error> {
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

fn axis_sample_count(axis: &AxisInfo, axes: &BTreeMap<String, i64>) -> i64 {
    axes.get(&axis.tag).copied().unwrap_or(1)
}

fn checked_location_count(info: &[AxisInfo], axes: &BTreeMap<String, i64>) -> Result<usize, Error> {
    info.iter().try_fold(1usize, |total, axis| {
        let count = usize::try_from(axis_sample_count(axis, axes)).map_err(|_| {
            Error::Parse("grid_instances axis density does not fit usize".to_string())
        })?;
        total.checked_mul(count).ok_or_else(|| {
            Error::Parse("grid_instances location count overflowed usize".to_string())
        })
    })
}

fn axis_grid_points(axis: &AxisInfo, count: i64) -> Vec<f32> {
    match count {
        1 => vec![axis.default],
        n => (0..n)
            .map(|index| {
                let t = index as f32 / (n - 1) as f32;
                axis.min + (axis.max - axis.min) * t
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{AxisInfo, checked_location_count};

    #[test]
    fn rejects_location_count_overflow() {
        let info = [axis("wght"), axis("wdth")];
        let axes = BTreeMap::from([
            ("wdth".to_string(), i64::MAX),
            ("wght".to_string(), i64::MAX),
        ]);
        assert!(checked_location_count(&info, &axes).is_err());
    }

    fn axis(tag: &str) -> AxisInfo {
        AxisInfo {
            tag: tag.to_string(),
            min: 0.0,
            default: 0.5,
            max: 1.0,
        }
    }
}
