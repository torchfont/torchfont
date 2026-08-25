use skrifa::outline::{DrawError, DrawSettings, OutlineGlyph, OutlinePen};

use crate::outline::{BezPath, Point};

/// Counts the elements one glyph draws, guarding drawing calls made with no
/// open subpath exactly as [`OutlineEncodingPen`] does so both pens agree.
struct ElementCountingPen {
    count: usize,
    has_open_subpath: bool,
}

impl ElementCountingPen {
    fn new() -> Self {
        Self {
            count: 0,
            has_open_subpath: false,
        }
    }
}

impl OutlinePen for ElementCountingPen {
    fn move_to(&mut self, _x: f32, _y: f32) {
        self.count += 1;
        self.has_open_subpath = true;
    }

    fn line_to(&mut self, _x: f32, _y: f32) {
        self.count += usize::from(self.has_open_subpath);
    }

    fn quad_to(&mut self, _cx0: f32, _cy0: f32, _x: f32, _y: f32) {
        self.count += usize::from(self.has_open_subpath);
    }

    fn curve_to(&mut self, _cx0: f32, _cy0: f32, _cx1: f32, _cy1: f32, _x: f32, _y: f32) {
        self.count += usize::from(self.has_open_subpath);
    }

    fn close(&mut self) {
        if self.has_open_subpath {
            self.count += 1;
            self.has_open_subpath = false;
        }
    }
}

struct OutlineEncodingPen {
    outline: BezPath,
    scale: f64,
    // skrifa's glyph draw contract is a well-formed sequence of subpaths, but
    // font files are external data; guard against a malformed/corrupt glyph
    // sending drawing calls with no open subpath instead of letting kurbo's
    // BezPath violate the "starts with MoveTo" invariant every downstream
    // helper relies on.
    has_open_subpath: bool,
}

impl OutlineEncodingPen {
    fn new(units_per_em: f32) -> Self {
        debug_assert!(units_per_em > 0.0, "units_per_em must be positive");
        Self {
            outline: BezPath::new(),
            scale: f64::from(units_per_em).recip(),
            has_open_subpath: false,
        }
    }

    fn point(&self, x: f32, y: f32) -> Point {
        Point::new(f64::from(x) * self.scale, f64::from(y) * self.scale)
    }
}

impl OutlinePen for OutlineEncodingPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.outline.move_to(self.point(x, y));
        self.has_open_subpath = true;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        if self.has_open_subpath {
            self.outline.line_to(self.point(x, y));
        }
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        if self.has_open_subpath {
            self.outline.quad_to(self.point(cx0, cy0), self.point(x, y));
        }
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        if self.has_open_subpath {
            self.outline
                .curve_to(self.point(cx0, cy0), self.point(cx1, cy1), self.point(x, y));
        }
    }

    fn close(&mut self) {
        if self.has_open_subpath {
            self.outline.close_path();
            self.has_open_subpath = false;
        }
    }
}

/// Counts the encoded sequence elements of one glyph, including the trailing
/// `End` marker, without building its outline.
pub(crate) fn count_glyph_elements<'a>(
    glyph: &OutlineGlyph<'a>,
    settings: DrawSettings<'a>,
) -> Result<usize, DrawError> {
    let mut pen = ElementCountingPen::new();
    glyph.draw(settings, &mut pen)?;
    Ok(pen.count + 1)
}

pub(crate) fn extract_glyph_outline<'a>(
    glyph: &OutlineGlyph<'a>,
    settings: DrawSettings<'a>,
    units_per_em: f32,
) -> Result<BezPath, DrawError> {
    let mut pen = OutlineEncodingPen::new(units_per_em);
    glyph.draw(settings, &mut pen)?;
    Ok(pen.outline)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use skrifa::{
        MetadataProvider as _,
        instance::{LocationRef, Size},
        outline::DrawSettings,
        raw::TableProvider as _,
    };

    use crate::font::{map_font_file, parse_font_ref};
    use crate::outline::encode;

    use super::{count_glyph_elements, extract_glyph_outline};

    #[test]
    fn counts_the_elements_the_encoder_emits() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fonts/source-sans/SourceSans3-Regular.ttf");
        let data = map_font_file(&path).unwrap();
        let font = parse_font_ref(&data[..], &path, 0).unwrap();
        let units_per_em = f32::from(font.head().unwrap().units_per_em());
        let settings = || DrawSettings::unhinted(Size::unscaled(), LocationRef::default());
        for (_, glyph) in font.outline_glyphs().iter() {
            let outline = extract_glyph_outline(&glyph, settings(), units_per_em).unwrap();
            assert_eq!(
                count_glyph_elements(&glyph, settings()).unwrap(),
                encode(&outline).0.len()
            );
        }
    }
}
