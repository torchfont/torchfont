use std::{fs, sync::Arc};

use memmap2::Mmap;
use skrifa::raw::types::NameId;
use skrifa::raw::{FileRef, TableProvider};
use skrifa::{GlyphId, MetadataProvider, instance::Location};

use super::glyph::{Bounds, Hmtx};
use super::reader::GlyphReader;
use super::table::{Head, Hhea, Maxp, Name, Os2, Post};
use crate::error::Error;
use crate::geom::Outline;

pub(super) struct GlyphIndex {
    codepoints: Vec<u32>,
    glyph_ids: Vec<GlyphId>,
}

pub(crate) struct FontEntry {
    index: GlyphIndex,
    reader: GlyphReader,
    pub(crate) head: Head,
    pub(crate) hhea: Hhea,
    pub(crate) os2: Os2,
    pub(crate) post: Post,
    pub(crate) maxp: Maxp,
    pub(crate) name: Name,
    locations: Vec<Location>,
    style_axes: Vec<Vec<(String, f32)>>,
}

impl FontEntry {
    pub(crate) fn load_faces(path: &str, filter: Option<&[u32]>) -> Result<Vec<Self>, Error> {
        let mapped = Arc::new(map_font(path)?);
        let parsed = FileRef::new(&mapped[..])
            .map_err(|err| Error::Parse(format!("failed to parse '{path}': {err}")))?;

        let entries = parsed
            .fonts()
            .enumerate()
            .map(|(face_index, face)| {
                let font = face.map_err(|err| {
                    Error::Parse(format!(
                        "failed to parse '{path}' (face {face_index}): {err}"
                    ))
                })?;
                Self::from_face(path, face_index as u32, Arc::clone(&mapped), &font, filter)
            })
            .collect::<Result<Vec<_>, Error>>()?;

        if entries.is_empty() {
            return Err(Error::Parse(format!(
                "font file '{path}' does not contain any fonts"
            )));
        }
        Ok(entries)
    }

    pub(crate) fn glyph_complete(
        &self,
        codepoint: u32,
        instance_index: Option<usize>,
    ) -> Result<(Outline, Hmtx, Bounds, String), Error> {
        let glyph_idx = self.lookup_glyph_index(codepoint)?;
        let glyph_id = self.index.glyph_ids[glyph_idx];
        self.reader.draw_glyph(
            glyph_id,
            self.head.units_per_em,
            &self.locations,
            instance_index,
        )
    }

    pub(crate) fn instance_count(&self) -> usize {
        self.style_axes.len()
    }

    pub(crate) fn is_variable(&self) -> bool {
        !self.locations.is_empty()
    }

    pub(crate) fn path(&self) -> &str {
        self.reader.path()
    }

    pub(crate) fn face_index(&self) -> u32 {
        self.reader.face_index()
    }

    pub(crate) fn codepoints(&self) -> &[u32] {
        &self.index.codepoints
    }

    pub(crate) fn codepoint_count(&self) -> usize {
        self.index.codepoints.len()
    }

    pub(crate) fn named_instance_names(&self) -> Vec<Option<String>> {
        if !self.is_variable() {
            return vec![];
        }
        self.reader.named_instance_names()
    }

    pub(crate) fn style_axes(&self) -> &[Vec<(String, f32)>] {
        &self.style_axes
    }

    fn from_face(
        base_path: &str,
        face_index: u32,
        data: Arc<Mmap>,
        font: &skrifa::FontRef<'_>,
        filter: Option<&[u32]>,
    ) -> Result<Self, Error> {
        let (head, hhea, os2, post, maxp, name) = parse_font_tables(font, base_path, face_index)?;

        let outline_glyphs = font.outline_glyphs();
        let mut mappings: Vec<_> = font
            .charmap()
            .mappings()
            .filter(|(codepoint, _)| {
                filter.is_none_or(|values| values.binary_search(codepoint).is_ok())
            })
            .filter(|(_, glyph_id)| outline_glyphs.get(*glyph_id).is_some())
            .collect();
        mappings.sort_unstable_by_key(|entry| entry.0);
        let (codepoints, glyph_ids): (Vec<_>, Vec<_>) = mappings.into_iter().unzip();

        let axis_tags: Vec<String> = font
            .axes()
            .iter()
            .map(|axis| axis.tag().to_string())
            .collect();
        let named_instances = font.named_instances();
        let locations: Vec<Location> = named_instances.iter().map(|inst| inst.location()).collect();
        let style_axes = if locations.is_empty() {
            vec![vec![]]
        } else {
            named_instances
                .iter()
                .map(|inst| {
                    debug_assert_eq!(
                        axis_tags.len(),
                        inst.user_coords().count(),
                        "font '{base_path}' (face {face_index}) reported mismatched axis metadata",
                    );
                    axis_tags.iter().cloned().zip(inst.user_coords()).collect()
                })
                .collect()
        };

        Ok(Self {
            index: GlyphIndex {
                codepoints,
                glyph_ids,
            },
            reader: GlyphReader::new(base_path.to_string(), face_index, data),
            head,
            hhea,
            os2,
            post,
            maxp,
            name,
            locations,
            style_axes,
        })
    }

    fn lookup_glyph_index(&self, codepoint: u32) -> Result<usize, Error> {
        self.index
            .codepoints
            .binary_search(&codepoint)
            .map_err(|_| {
                Error::OutOfRange(format!(
                    "codepoint U+{codepoint:04X} missing from '{}'",
                    self.reader.path()
                ))
            })
    }
}

fn parse_font_tables(
    font: &skrifa::FontRef<'_>,
    path: &str,
    face_index: u32,
) -> Result<(Head, Hhea, Os2, Post, Maxp, Name), Error> {
    let missing = |table: &'static str| {
        move |_| {
            Error::Parse(format!(
                "font '{path}' (face {face_index}) is missing '{table}' table"
            ))
        }
    };

    let raw_head = font.head().map_err(missing("head"))?;
    let raw_hhea = font.hhea().map_err(missing("hhea"))?;
    let raw_os2 = font.os2().map_err(missing("OS/2"))?;
    let raw_post = font.post().map_err(missing("post"))?;
    let raw_maxp = font.maxp().map_err(missing("maxp"))?;

    let inv = (raw_head.units_per_em() as f32).recip();
    let norm = |v: i16| v as f32 * inv;
    let norm_u = |v: u16| v as f32 * inv;
    let opt_norm = |v: Option<i16>| v.map(|x| x as f32 * inv);

    let head = Head {
        units_per_em: raw_head.units_per_em(),
        flags: raw_head.flags().bits(),
        x_min: norm(raw_head.x_min()),
        y_min: norm(raw_head.y_min()),
        x_max: norm(raw_head.x_max()),
        y_max: norm(raw_head.y_max()),
        mac_style: raw_head.mac_style().bits(),
        lowest_rec_ppem: raw_head.lowest_rec_ppem(),
    };

    let hhea = Hhea {
        ascender: norm(raw_hhea.ascender().into()),
        descender: norm(raw_hhea.descender().into()),
        line_gap: norm(raw_hhea.line_gap().into()),
        advance_width_max: norm_u(raw_hhea.advance_width_max().into()),
        min_left_side_bearing: norm(raw_hhea.min_left_side_bearing().into()),
        min_right_side_bearing: norm(raw_hhea.min_right_side_bearing().into()),
        x_max_extent: norm(raw_hhea.x_max_extent().into()),
        caret_slope_rise: raw_hhea.caret_slope_rise(),
        caret_slope_run: raw_hhea.caret_slope_run(),
        caret_offset: norm(raw_hhea.caret_offset()),
    };

    let mut panose = [0u8; 10];
    panose.copy_from_slice(raw_os2.panose_10());

    let os2 = Os2 {
        weight_class: raw_os2.us_weight_class(),
        width_class: raw_os2.us_width_class(),
        fs_type: raw_os2.fs_type(),
        fs_selection: raw_os2.fs_selection().bits(),
        typo_ascender: norm(raw_os2.s_typo_ascender()),
        typo_descender: norm(raw_os2.s_typo_descender()),
        typo_line_gap: norm(raw_os2.s_typo_line_gap()),
        win_ascent: norm_u(raw_os2.us_win_ascent()),
        win_descent: norm_u(raw_os2.us_win_descent()),
        avg_char_width: norm(raw_os2.x_avg_char_width()),
        y_subscript_x_size: norm(raw_os2.y_subscript_x_size()),
        y_subscript_y_size: norm(raw_os2.y_subscript_y_size()),
        y_subscript_x_offset: norm(raw_os2.y_subscript_x_offset()),
        y_subscript_y_offset: norm(raw_os2.y_subscript_y_offset()),
        y_superscript_x_size: norm(raw_os2.y_superscript_x_size()),
        y_superscript_y_size: norm(raw_os2.y_superscript_y_size()),
        y_superscript_x_offset: norm(raw_os2.y_superscript_x_offset()),
        y_superscript_y_offset: norm(raw_os2.y_superscript_y_offset()),
        y_strikeout_size: norm(raw_os2.y_strikeout_size()),
        y_strikeout_position: norm(raw_os2.y_strikeout_position()),
        s_family_class: raw_os2.s_family_class(),
        panose,
        vend_id: raw_os2.ach_vend_id().into_bytes(),
        us_first_char_index: raw_os2.us_first_char_index(),
        us_last_char_index: raw_os2.us_last_char_index(),
        x_height: opt_norm(raw_os2.sx_height()),
        cap_height: opt_norm(raw_os2.s_cap_height()),
        us_default_char: raw_os2.us_default_char(),
        us_break_char: raw_os2.us_break_char(),
        us_max_context: raw_os2.us_max_context(),
    };

    let post = Post {
        italic_angle: raw_post.italic_angle().to_f64() as f32,
        is_fixed_pitch: raw_post.is_fixed_pitch() != 0,
        underline_position: norm(raw_post.underline_position().into()),
        underline_thickness: norm(raw_post.underline_thickness().into()),
    };

    let maxp = Maxp {
        num_glyphs: raw_maxp.num_glyphs(),
        max_points: raw_maxp.max_points(),
        max_contours: raw_maxp.max_contours(),
        max_composite_points: raw_maxp.max_composite_points(),
        max_composite_contours: raw_maxp.max_composite_contours(),
        max_zones: raw_maxp.max_zones(),
        max_twilight_points: raw_maxp.max_twilight_points(),
        max_storage: raw_maxp.max_storage(),
        max_function_defs: raw_maxp.max_function_defs(),
        max_instruction_defs: raw_maxp.max_instruction_defs(),
        max_stack_elements: raw_maxp.max_stack_elements(),
        max_size_of_instructions: raw_maxp.max_size_of_instructions(),
        max_component_elements: raw_maxp.max_component_elements(),
        max_component_depth: raw_maxp.max_component_depth(),
    };

    let localized = |ids: &[NameId]| -> String {
        ids.iter()
            .find_map(|&id| {
                font.localized_strings(id)
                    .english_or_first()
                    .map(|s| s.to_string())
            })
            .unwrap_or_default()
    };

    let one = |id: NameId| localized(&[id]);
    let name = Name {
        copyright_notice: one(NameId::COPYRIGHT_NOTICE),
        family_name: localized(&[NameId::TYPOGRAPHIC_FAMILY_NAME, NameId::FAMILY_NAME]),
        subfamily_name: localized(&[NameId::TYPOGRAPHIC_SUBFAMILY_NAME, NameId::SUBFAMILY_NAME]),
        unique_font_identifier: one(NameId::UNIQUE_ID),
        full_name: localized(&[NameId::FULL_NAME]),
        version_string: one(NameId::VERSION_STRING),
        postscript_name: one(NameId::POSTSCRIPT_NAME),
        trademark: one(NameId::TRADEMARK),
        manufacturer_name: one(NameId::MANUFACTURER),
        designer: one(NameId::DESIGNER),
        description: one(NameId::DESCRIPTION),
        vendor_url: one(NameId::VENDOR_URL),
        designer_url: one(NameId::DESIGNER_URL),
        license_description: one(NameId::LICENSE_DESCRIPTION),
        license_info_url: one(NameId::LICENSE_URL),
        compatible_full_name: one(NameId::COMPATIBLE_FULL_NAME),
        sample_text: one(NameId::SAMPLE_TEXT),
        postscript_cid_findfont_name: one(NameId::POSTSCRIPT_CID_NAME),
        wws_family_name: one(NameId::WWS_FAMILY_NAME),
        wws_subfamily_name: one(NameId::WWS_SUBFAMILY_NAME),
        light_background_palette: one(NameId::LIGHT_BACKGROUND_PALETTE),
        dark_background_palette: one(NameId::DARK_BACKGROUND_PALETTE),
        variations_postscript_name_prefix: one(NameId::VARIATIONS_POSTSCRIPT_NAME_PREFIX),
    };

    Ok((head, hhea, os2, post, maxp, name))
}

fn map_font(path: &str) -> Result<Mmap, Error> {
    let file =
        fs::File::open(path).map_err(|err| Error::Io(format!("failed to open '{path}': {err}")))?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|err| Error::Io(format!("failed to map '{path}': {err}")))?;
    Ok(mmap)
}
