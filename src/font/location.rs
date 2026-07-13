use skrifa::MetadataProvider;

use std::{collections::BTreeMap, path::Path};

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

pub(crate) type Location = Vec<(String, f32)>;

pub(crate) fn default_location(font: &skrifa::FontRef<'_>) -> Location {
    axis_info(font)
        .into_iter()
        .map(|axis| (axis.tag, axis.default))
        .collect()
}

pub(crate) fn canonicalize_location(
    font: &skrifa::FontRef<'_>,
    path: &Path,
    ttc_index: u32,
    location: Option<&BTreeMap<String, f32>>,
) -> Result<Location, Error> {
    let Some(location) = location else {
        return Ok(default_location(font));
    };
    let axes = axis_info(font);
    for (tag, value) in location {
        let Some(axis) = axes.iter().find(|axis| axis.tag == *tag) else {
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
    Ok(axes
        .into_iter()
        .map(|axis| {
            let value = location.get(&axis.tag).copied().unwrap_or(axis.default);
            (axis.tag, value)
        })
        .collect())
}
