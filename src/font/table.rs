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

/// Font-level strings from the `name` table (IDs 0–25, one field per ID).
pub(crate) struct Name {
    pub copyright_notice: String,                  // ID 0
    pub family_name: String,                       // ID 1
    pub subfamily_name: String,                    // ID 2
    pub unique_font_identifier: String,            // ID 3
    pub full_name: String,                         // ID 4
    pub version_string: String,                    // ID 5
    pub postscript_name: String,                   // ID 6
    pub trademark: String,                         // ID 7
    pub manufacturer_name: String,                 // ID 8
    pub designer: String,                          // ID 9
    pub description: String,                       // ID 10
    pub vendor_url: String,                        // ID 11
    pub designer_url: String,                      // ID 12
    pub license_description: String,               // ID 13
    pub license_info_url: String,                  // ID 14
    pub reserved: String,                          // ID 15 (reserved; typically empty)
    pub typographic_family_name: String,           // ID 16
    pub typographic_subfamily_name: String,        // ID 17
    pub compatible_full_name: String,              // ID 18
    pub sample_text: String,                       // ID 19
    pub postscript_cid_findfont_name: String,      // ID 20
    pub wws_family_name: String,                   // ID 21
    pub wws_subfamily_name: String,                // ID 22
    pub light_background_palette: String,          // ID 23
    pub dark_background_palette: String,           // ID 24
    pub variations_postscript_name_prefix: String, // ID 25
}
