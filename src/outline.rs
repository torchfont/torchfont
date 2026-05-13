use skrifa::outline::{DrawError, DrawSettings, OutlineGlyph, OutlinePen};

#[derive(Clone, Copy)]
#[repr(i32)]
pub(crate) enum Command {
    MoveTo = 1,
    LineTo = 2,
    QuadTo = 3,
    CurveTo = 4,
    Close = 5,
    End = 6,
}

struct SegmentPen {
    commands: Vec<i32>,
    coords: Vec<f32>,
    scale: f32,
}

impl SegmentPen {
    fn new(units_per_em: f32) -> Self {
        debug_assert!(units_per_em > 0.0, "units_per_em must be positive");
        Self {
            commands: Vec::new(),
            coords: Vec::new(),
            scale: units_per_em.recip(),
        }
    }

    fn finish(mut self) -> (Vec<i32>, Vec<f32>) {
        self.push(Command::End, [0.0; 6]);
        (self.commands, self.coords)
    }

    fn push(&mut self, command: Command, values: [f32; 6]) {
        self.commands.push(command as i32);
        let scaled = values.map(|v| v * self.scale);
        self.coords.extend_from_slice(&scaled);
    }

    fn push_endpoint(&mut self, command: Command, x: f32, y: f32) {
        self.push(command, [0.0, 0.0, 0.0, 0.0, x, y]);
    }
}

impl OutlinePen for SegmentPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.push_endpoint(Command::MoveTo, x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.push_endpoint(Command::LineTo, x, y);
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.push(Command::QuadTo, [cx0, cy0, 0.0, 0.0, x, y]);
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.push(Command::CurveTo, [cx0, cy0, cx1, cy1, x, y]);
    }

    fn close(&mut self) {
        self.push(Command::Close, [0.0; 6]);
    }
}

pub(crate) fn extract_glyph_segments<'a>(
    glyph: &OutlineGlyph<'a>,
    settings: DrawSettings<'a>,
    units_per_em: f32,
) -> Result<(Vec<i32>, Vec<f32>), DrawError> {
    let mut pen = SegmentPen::new(units_per_em);
    glyph.draw(settings, &mut pen)?;
    Ok(pen.finish())
}

pub(crate) fn quad_to_cubic(types: &mut [i64], coords: &mut [f32], seq_len: usize) {
    debug_assert_eq!(types.len() * 6, coords.len());
    if seq_len == 0 {
        return;
    }
    let quad = Command::QuadTo as i64;
    let cubic = Command::CurveTo as i64;

    for (t_seq, c_seq) in types
        .chunks_mut(seq_len)
        .zip(coords.chunks_mut(seq_len * 6))
    {
        let mut prev_x = 0.0f32;
        let mut prev_y = 0.0f32;
        for (i, t) in t_seq.iter_mut().enumerate() {
            let base = i * 6;
            let end_x = c_seq[base + 4];
            let end_y = c_seq[base + 5];
            if *t == quad {
                let cx0 = c_seq[base];
                let cy0 = c_seq[base + 1];
                c_seq[base] = prev_x + (2.0 / 3.0) * (cx0 - prev_x);
                c_seq[base + 1] = prev_y + (2.0 / 3.0) * (cy0 - prev_y);
                c_seq[base + 2] = end_x + (2.0 / 3.0) * (cx0 - end_x);
                c_seq[base + 3] = end_y + (2.0 / 3.0) * (cy0 - end_y);
                *t = cubic;
            }
            prev_x = end_x;
            prev_y = end_y;
        }
    }
}
