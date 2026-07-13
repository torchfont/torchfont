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
        let key: Vec<_> = location
            .iter()
            .map(|(tag, value)| (tag.clone(), value.to_bits()))
            .collect();
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
