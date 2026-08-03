use skrifa::raw::{
    TableProvider,
    tables::{head::MacStyle, os2::SelectionFlags},
};

use super::Location;

#[derive(Clone, Copy)]
pub(crate) struct RegisteredAxisValues {
    pub(crate) weight: f32,
    pub(crate) width: f32,
    pub(crate) italic: f32,
    pub(crate) slant: f32,
    pub(crate) optical_size: f32,
}

impl RegisteredAxisValues {
    fn apply_location(mut self, location: &[(String, f32)]) -> Self {
        for (tag, value) in location {
            match tag.as_str() {
                "wght" => self.weight = *value,
                "wdth" => self.width = *value,
                "ital" => self.italic = *value,
                "slnt" => self.slant = *value,
                "opsz" => self.optical_size = *value,
                _ => {}
            }
        }
        self
    }
}

pub(crate) fn registered_axis_values(
    font: &skrifa::FontRef<'_>,
    location: &Location,
) -> RegisteredAxisValues {
    let mut values = RegisteredAxisValues {
        weight: f32::NAN,
        width: f32::NAN,
        italic: f32::NAN,
        slant: f32::NAN,
        optical_size: f32::NAN,
    }
    .apply_location(location);

    if (values.weight.is_nan() || values.width.is_nan() || values.italic.is_nan())
        && let Ok(os2) = font.os2()
    {
        if values.weight.is_nan() {
            values.weight = weight_value(os2.us_weight_class());
        }
        if values.width.is_nan() {
            values.width = width_percentage(os2.us_width_class());
        }
        if values.italic.is_nan() {
            values.italic = f32::from(os2.fs_selection().contains(SelectionFlags::ITALIC));
        }
    }

    if values.italic.is_nan()
        && let Ok(head) = font.head()
    {
        values.italic = f32::from(head.mac_style().contains(MacStyle::ITALIC));
    }
    if values.slant.is_nan()
        && let Ok(post) = font.post()
    {
        values.slant = post.italic_angle().to_f64() as f32;
    }
    values
}

fn weight_value(weight_class: u16) -> f32 {
    match weight_class {
        value @ 1..=1000 => value as f32,
        _ => f32::NAN,
    }
}

fn width_percentage(width_class: u16) -> f32 {
    match width_class {
        1 => 50.0,
        2 => 62.5,
        3 => 75.0,
        4 => 87.5,
        5 => 100.0,
        6 => 112.5,
        7 => 125.0,
        8 => 150.0,
        9 => 200.0,
        _ => f32::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::{RegisteredAxisValues, weight_value, width_percentage};

    #[test]
    fn location_values_override_fallbacks_by_registered_tag() {
        let values = RegisteredAxisValues {
            weight: 400.0,
            width: 100.0,
            italic: 0.0,
            slant: 0.0,
            optical_size: f32::NAN,
        }
        .apply_location(&[
            ("wght".to_string(), 700.0),
            ("ital".to_string(), 0.5),
            ("TEST".to_string(), 42.0),
        ]);

        assert_eq!(values.weight, 700.0);
        assert_eq!(values.width, 100.0);
        assert_eq!(values.italic, 0.5);
        assert_eq!(values.slant, 0.0);
        assert!(values.optical_size.is_nan());
    }

    #[test]
    fn maps_valid_os2_classes_to_registered_scales() {
        assert_eq!(weight_value(1), 1.0);
        assert_eq!(weight_value(400), 400.0);
        assert_eq!(weight_value(1000), 1000.0);
        assert_eq!(width_percentage(1), 50.0);
        assert_eq!(width_percentage(5), 100.0);
        assert_eq!(width_percentage(9), 200.0);
    }

    #[test]
    fn invalid_os2_classes_are_unavailable() {
        assert!(weight_value(0).is_nan());
        assert!(weight_value(1001).is_nan());
        assert!(width_percentage(0).is_nan());
        assert!(width_percentage(10).is_nan());
    }
}
