use std::collections::BTreeSet;

use crate::font::Location;

pub(super) fn character_index<T>(fonts: &[T], codepoints: impl Fn(&T) -> &[u32]) -> Vec<u32> {
    fonts
        .iter()
        .flat_map(|font| codepoints(font).iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(super) fn style_name(family_name: &str, subfamily_name: &str, location: &Location) -> String {
    if !location.is_empty() {
        return format!("{family_name} {}", format_location(location));
    }
    if !subfamily_name.is_empty() {
        return format!("{family_name} {subfamily_name}");
    }
    family_name.to_string()
}

fn format_location(location: &Location) -> String {
    location
        .iter()
        .map(|(tag, value)| format!("{tag}={}", format_axis_value(*value)))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_axis_value(value: f32) -> String {
    let mut formatted = format!("{value:.6}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    formatted
}

#[cfg(test)]
mod tests {
    use super::style_name;

    #[test]
    fn style_name_formats_locations_deterministically() {
        let location = vec![("wght".to_string(), 400.0), ("wdth".to_string(), 87.5)];
        assert_eq!(
            style_name("Family", "Regular", &location),
            "Family wght=400,wdth=87.5"
        );
    }
}
