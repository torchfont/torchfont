use std::path::Path;

use skrifa::MetadataProvider;

use crate::{
    error::Error,
    font::{Location, axis_info},
};

pub(crate) fn named_locations(
    font: &skrifa::FontRef<'_>,
    path: &Path,
    ttc_index: u32,
) -> Result<Vec<Location>, Error> {
    let axis_tags: Vec<_> = axis_info(font).into_iter().map(|axis| axis.tag).collect();
    let named_instances = font.named_instances();
    if named_instances.is_empty() {
        return Ok(Vec::new());
    }
    let coords = named_instances
        .iter()
        .map(|instance| {
            let coords: Vec<f32> = instance.user_coords().collect();
            if coords.len() != axis_tags.len() {
                return Err(Error::Parse(format!(
                    "font '{}' (ttc_index {ttc_index}) reported mismatched axis metadata",
                    path.display(),
                )));
            }
            Ok(coords)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(dedup_coords(coords)
        .into_iter()
        .map(|values| axis_tags.iter().cloned().zip(values).collect())
        .collect())
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

#[cfg(test)]
mod tests {
    use super::dedup_coords;

    #[test]
    fn removes_duplicate_coordinates() {
        assert_eq!(
            dedup_coords(vec![vec![100.0], vec![700.0], vec![100.0]]),
            vec![vec![100.0], vec![700.0]],
        );
    }
}
