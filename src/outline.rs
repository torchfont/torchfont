use skrifa::outline::{DrawError, DrawSettings, OutlineGlyph, OutlinePen};

#[derive(Clone, Copy)]
#[repr(i32)]
pub(crate) enum ElementType {
    MoveTo = 1,
    LineTo = 2,
    QuadTo = 3,
    CurveTo = 4,
    Close = 5,
    End = 6,
}

struct OutlineEncodingPen {
    types: Vec<i64>,
    coords: Vec<f32>,
    scale: f32,
}

impl OutlineEncodingPen {
    fn new(units_per_em: f32) -> Self {
        debug_assert!(units_per_em > 0.0, "units_per_em must be positive");
        Self {
            types: Vec::new(),
            coords: Vec::new(),
            scale: units_per_em.recip(),
        }
    }

    fn finish(mut self) -> (Vec<i64>, Vec<f32>) {
        self.push(ElementType::End, [0.0; 6]);
        (self.types, self.coords)
    }

    fn push(&mut self, element_type: ElementType, values: [f32; 6]) {
        self.types.push(element_type as i64);
        let scaled = values.map(|v| v * self.scale);
        self.coords.extend_from_slice(&scaled);
    }

    fn push_endpoint(&mut self, element_type: ElementType, x: f32, y: f32) {
        self.push(element_type, [0.0, 0.0, 0.0, 0.0, x, y]);
    }
}

impl OutlinePen for OutlineEncodingPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.push_endpoint(ElementType::MoveTo, x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.push_endpoint(ElementType::LineTo, x, y);
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.push(ElementType::QuadTo, [cx0, cy0, 0.0, 0.0, x, y]);
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.push(ElementType::CurveTo, [cx0, cy0, cx1, cy1, x, y]);
    }

    fn close(&mut self) {
        self.push(ElementType::Close, [0.0; 6]);
    }
}

pub(crate) fn extract_glyph_outline<'a>(
    glyph: &OutlineGlyph<'a>,
    settings: DrawSettings<'a>,
    units_per_em: f32,
) -> Result<(Vec<i64>, Vec<f32>), DrawError> {
    let mut pen = OutlineEncodingPen::new(units_per_em);
    glyph.draw(settings, &mut pen)?;
    Ok(pen.finish())
}
