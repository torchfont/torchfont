use skrifa::outline::{DrawError, DrawSettings, OutlineGlyph, OutlinePen};

use crate::outline::{BezPath, Point};

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

pub(crate) fn extract_glyph_outline<'a>(
    glyph: &OutlineGlyph<'a>,
    settings: DrawSettings<'a>,
    units_per_em: f32,
) -> Result<BezPath, DrawError> {
    let mut pen = OutlineEncodingPen::new(units_per_em);
    glyph.draw(settings, &mut pen)?;
    Ok(pen.outline)
}
