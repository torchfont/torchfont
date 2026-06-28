use std::{collections::HashMap, path::Path};

use skrifa::MetadataProvider;

use crate::error::Error;

#[derive(Clone)]
pub(crate) enum VariationInstantiation {
    Default,
    Named,
    Grid { axes: HashMap<String, u32> },
}

/// Lazily enumerable set of variable-font instances.
///
/// Rather than materializing every instance up front, this keeps only the
/// per-axis candidate values (for Cartesian policies such as `Default`/`Grid`)
/// or the explicit named-instance coordinate list, and derives the i-th
/// instance on demand. This keeps memory proportional to the axis description
/// instead of the combinatorial instance count.
pub(crate) struct StyleSpace {
    axis_tags: Vec<String>,
    kind: StyleSpaceKind,
}

enum StyleSpaceKind {
    /// Cartesian product of per-axis candidate values.
    Cartesian { axis_points: Vec<Vec<f32>> },
    /// Explicit list of full user-coordinate tuples (named instances).
    Explicit { instances: Vec<Vec<f32>> },
}

impl StyleSpace {
    fn static_single() -> Self {
        Self {
            axis_tags: Vec::new(),
            kind: StyleSpaceKind::Cartesian {
                axis_points: Vec::new(),
            },
        }
    }

    pub(crate) fn is_variable(&self) -> bool {
        !self.axis_tags.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        match &self.kind {
            StyleSpaceKind::Cartesian { axis_points } => axis_points.iter().map(Vec::len).product(),
            StyleSpaceKind::Explicit { instances } => instances.len(),
        }
    }

    /// User-space axis coordinates of the `index`-th instance.
    ///
    /// For the Cartesian kind the instance index is decomposed mixed-radix
    /// across the axes (first axis most significant), matching the order in
    /// which a fully materialized grid would have been enumerated.
    pub(crate) fn user_coords(&self, index: usize) -> Vec<(String, f32)> {
        match &self.kind {
            StyleSpaceKind::Cartesian { axis_points } => {
                let mut radix: usize = axis_points.iter().map(Vec::len).product();
                let mut remainder = index;
                let mut coords = Vec::with_capacity(self.axis_tags.len());
                for (tag, points) in self.axis_tags.iter().zip(axis_points) {
                    radix /= points.len();
                    let digit = remainder / radix;
                    remainder %= radix;
                    coords.push((tag.clone(), points[digit]));
                }
                coords
            }
            StyleSpaceKind::Explicit { instances } => self
                .axis_tags
                .iter()
                .cloned()
                .zip(instances[index].iter().copied())
                .collect(),
        }
    }
}

impl VariationInstantiation {
    pub(crate) fn instantiate(
        &self,
        font: &skrifa::FontRef<'_>,
        path: &Path,
        face_index: u32,
    ) -> Result<StyleSpace, Error> {
        let axis_info: Vec<AxisInfo> = font
            .axes()
            .iter()
            .map(|axis| AxisInfo {
                tag: axis.tag().to_string(),
                min: axis.min_value(),
                default: axis.default_value(),
                max: axis.max_value(),
            })
            .collect();

        if axis_info.is_empty() {
            return Ok(StyleSpace::static_single());
        }

        let axis_tags: Vec<String> = axis_info.iter().map(|axis| axis.tag.clone()).collect();
        let kind = match self {
            Self::Default => StyleSpaceKind::Cartesian {
                axis_points: default_points(&axis_info),
            },
            Self::Named => match named_instance_coords(font, &axis_tags, path, face_index)? {
                Some(instances) => StyleSpaceKind::Explicit { instances },
                None => StyleSpaceKind::Cartesian {
                    axis_points: default_points(&axis_info),
                },
            },
            Self::Grid { axes } => StyleSpaceKind::Cartesian {
                axis_points: axis_info
                    .iter()
                    .map(|axis| axis_grid_points(axis, axis_sample_count(axis, axes)))
                    .collect(),
            },
        };

        Ok(StyleSpace { axis_tags, kind })
    }
}

struct AxisInfo {
    tag: String,
    min: f32,
    default: f32,
    max: f32,
}

fn default_points(axes: &[AxisInfo]) -> Vec<Vec<f32>> {
    axes.iter().map(|axis| vec![axis.default]).collect()
}

fn named_instance_coords(
    font: &skrifa::FontRef<'_>,
    axis_tags: &[String],
    path: &Path,
    face_index: u32,
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
                    "font '{}' (face {face_index}) reported mismatched axis metadata",
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
fn axis_sample_count(axis: &AxisInfo, axes: &HashMap<String, u32>) -> u32 {
    axes.get(&axis.tag).copied().unwrap_or(1)
}

fn axis_grid_points(axis: &AxisInfo, count: u32) -> Vec<f32> {
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
