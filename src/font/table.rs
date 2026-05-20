pub(crate) struct Head {
    pub units_per_em: u16,
    pub flags: u16,
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub mac_style: u16,
    pub lowest_rec_ppem: u16,
}

pub(crate) struct Hhea {
    pub ascender: f32,
    pub descender: f32,
    pub line_gap: f32,
    pub advance_width_max: f32,
    pub min_left_side_bearing: f32,
    pub min_right_side_bearing: f32,
    pub x_max_extent: f32,
    pub caret_slope_rise: i16,
    pub caret_slope_run: i16,
    pub caret_offset: f32,
}

pub(crate) struct Os2 {
    pub weight_class: u16,
    pub width_class: u16,
    pub fs_type: u16,
    pub fs_selection: u16,
    pub typo_ascender: f32,
    pub typo_descender: f32,
    pub typo_line_gap: f32,
    pub win_ascent: f32,
    pub win_descent: f32,
    pub avg_char_width: f32,
    pub y_subscript_x_size: f32,
    pub y_subscript_y_size: f32,
    pub y_subscript_x_offset: f32,
    pub y_subscript_y_offset: f32,
    pub y_superscript_x_size: f32,
    pub y_superscript_y_size: f32,
    pub y_superscript_x_offset: f32,
    pub y_superscript_y_offset: f32,
    pub y_strikeout_size: f32,
    pub y_strikeout_position: f32,
    pub s_family_class: i16,
    pub panose: [u8; 10],
    /// 4 ASCII bytes, e.g. `ADBO`, `GOOG`.
    pub vend_id: [u8; 4],
    pub us_first_char_index: u16,
    pub us_last_char_index: u16,
    /// `None` on OS/2 version < 2.
    pub x_height: Option<f32>,
    /// `None` on OS/2 version < 2.
    pub cap_height: Option<f32>,
    /// `None` on OS/2 version < 2.
    pub us_default_char: Option<u16>,
    /// `None` on OS/2 version < 2.
    pub us_break_char: Option<u16>,
    /// `None` on OS/2 version < 2.
    pub us_max_context: Option<u16>,
}

pub(crate) struct Post {
    /// Counter-clockwise degrees from the vertical.
    pub italic_angle: f32,
    pub is_fixed_pitch: bool,
    pub underline_position: f32,
    pub underline_thickness: f32,
}

pub(crate) struct Maxp {
    pub num_glyphs: u16,
    pub max_points: Option<u16>,
    pub max_contours: Option<u16>,
    pub max_composite_points: Option<u16>,
    pub max_composite_contours: Option<u16>,
    pub max_zones: Option<u16>,
    pub max_twilight_points: Option<u16>,
    pub max_storage: Option<u16>,
    pub max_function_defs: Option<u16>,
    pub max_instruction_defs: Option<u16>,
    pub max_stack_elements: Option<u16>,
    pub max_size_of_instructions: Option<u16>,
    pub max_component_elements: Option<u16>,
    pub max_component_depth: Option<u16>,
}

/// Font-level strings from the `name` table (English or first available).
pub(crate) struct Name {
    pub copyright_notice: String,
    pub family_name: String,
    pub subfamily_name: String,
    pub unique_font_identifier: String,
    pub full_name: String,
    pub version_string: String,
    pub postscript_name: String,
    pub trademark: String,
    pub manufacturer_name: String,
    pub designer: String,
    pub description: String,
    pub vendor_url: String,
    pub designer_url: String,
    pub license_description: String,
    pub license_info_url: String,
    pub compatible_full_name: String,
    pub sample_text: String,
    pub postscript_cid_findfont_name: String,
    pub wws_family_name: String,
    pub wws_subfamily_name: String,
    pub light_background_palette: String,
    pub dark_background_palette: String,
    pub variations_postscript_name_prefix: String,
}
