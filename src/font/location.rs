use skrifa::MetadataProvider;

use std::{collections::BTreeMap, path::Path};

use crate::error::Error;

#[derive(Clone)]
pub(crate) struct AxisInfo {
    pub(crate) tag: String,
    pub(crate) min_value: f32,
    pub(crate) default_value: f32,
    pub(crate) max_value: f32,
}

pub(crate) fn axis_info(font: &skrifa::FontRef<'_>) -> Vec<AxisInfo> {
    font.axes()
        .iter()
        .map(|axis| AxisInfo {
            tag: axis.tag().to_string(),
            min_value: axis.min_value(),
            default_value: axis.default_value(),
            max_value: axis.max_value(),
        })
        .collect()
}

pub(crate) type Location = Vec<(String, f32)>;

pub(crate) fn default_location(font: &skrifa::FontRef<'_>) -> Location {
    axis_info(font)
        .into_iter()
        .map(|axis| (axis.tag, axis.default_value))
        .collect()
}

pub(crate) fn canonicalize_location(
    font: &skrifa::FontRef<'_>,
    path: &Path,
    face_index: u32,
    location: Option<&BTreeMap<String, f32>>,
) -> Result<Location, Error> {
    let Some(location) = location else {
        return Ok(default_location(font));
    };
    let axes = axis_info(font);
    for (tag, value) in location {
        let Some(axis) = axes.iter().find(|axis| axis.tag == *tag) else {
            return Err(Error::Parse(format!(
                "font '{}' (face_index {face_index}) has no variation axis '{tag}'",
                path.display(),
            )));
        };
        if !value.is_finite() {
            return Err(Error::Parse(format!(
                "variation axis '{tag}' value must be finite"
            )));
        }
        if *value < axis.min_value || *value > axis.max_value {
            return Err(Error::Parse(format!(
                "variation axis '{tag}' value {value} is outside [{}, {}]",
                axis.min_value, axis.max_value,
            )));
        }
    }
    Ok(axes
        .into_iter()
        .map(|axis| {
            let value = location
                .get(&axis.tag)
                .copied()
                .unwrap_or(axis.default_value);
            (axis.tag, value)
        })
        .collect())
}
