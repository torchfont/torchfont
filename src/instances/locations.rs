use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
};

use crate::{
    error::Error,
    font::{Location, canonicalize_location},
};

pub(crate) fn canonicalize_locations(
    font: &skrifa::FontRef<'_>,
    path: &Path,
    ttc_index: u32,
    locations: &[BTreeMap<String, f32>],
) -> Result<Vec<Location>, Error> {
    let locations = locations
        .iter()
        .map(|location| canonicalize_location(font, path, ttc_index, Some(location)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut seen = HashSet::new();
    for location in &locations {
        let key = location_key(location);
        if !seen.insert(key) {
            return Err(Error::Parse(format!(
                "InstanceLocationsFn returned duplicate variation locations for '{}' \
                 (ttc_index={ttc_index}) after canonicalization",
                path.display()
            )));
        }
    }
    Ok(locations)
}

fn location_key(location: &Location) -> Vec<(String, u32)> {
    location
        .iter()
        .map(|(tag, value)| {
            let bits = if *value == 0.0 {
                0.0_f32.to_bits()
            } else {
                value.to_bits()
            };
            (tag.clone(), bits)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::location_key;

    #[test]
    fn location_key_treats_signed_zero_as_equal() {
        let positive = vec![("ital".to_string(), 0.0)];
        let negative = vec![("ital".to_string(), -0.0)];

        assert_eq!(location_key(&positive), location_key(&negative));
    }
}
