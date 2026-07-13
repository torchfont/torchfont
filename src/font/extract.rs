use skrifa::outline::{DrawError, DrawSettings, OutlineGlyph, OutlinePen};

use crate::outline::{Outline, PathElement, Point, Subpath, SubpathBuilder};

struct OutlineEncodingPen {
    subpaths: Vec<Subpath>,
    current: Option<SubpathBuilder>,
    scale: f32,
}

impl OutlineEncodingPen {
    fn new(units_per_em: f32) -> Self {
        debug_assert!(units_per_em > 0.0, "units_per_em must be positive");
        Self {
            subpaths: Vec::new(),
            current: None,
            scale: units_per_em.recip(),
        }
    }

    fn finish(mut self) -> Outline {
        if let Some(builder) = self.current.take() {
            self.subpaths.push(builder.finish(false));
        }
        Outline::new(self.subpaths)
    }

    fn point(&self, x: f32, y: f32) -> Point {
        Point::new(x * self.scale, y * self.scale)
    }

    fn push_element(&mut self, element: PathElement) {
        if let Some(subpath) = &mut self.current {
            subpath.elements.push(element);
        }
    }
}

impl OutlinePen for OutlineEncodingPen {
    fn move_to(&mut self, x: f32, y: f32) {
        if let Some(builder) = self.current.take() {
            self.subpaths.push(builder.finish(false));
        }
        self.current = Some(SubpathBuilder::new(self.point(x, y)));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.push_element(PathElement::LineTo(self.point(x, y)));
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.push_element(PathElement::QuadTo {
            control: self.point(cx0, cy0),
            end: self.point(x, y),
        });
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.push_element(PathElement::CurveTo {
            control0: self.point(cx0, cy0),
            control1: self.point(cx1, cy1),
            end: self.point(x, y),
        });
    }

    fn close(&mut self) {
        if let Some(builder) = self.current.take() {
            self.subpaths.push(builder.finish(true));
        }
    }
}

pub(crate) fn extract_glyph_outline<'a>(
    glyph: &OutlineGlyph<'a>,
    settings: DrawSettings<'a>,
    units_per_em: f32,
) -> Result<Outline, DrawError> {
    let mut pen = OutlineEncodingPen::new(units_per_em);
    glyph.draw(settings, &mut pen)?;
    Ok(pen.finish())
}
