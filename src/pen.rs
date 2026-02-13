use skrifa::outline::OutlinePen;

#[derive(Clone, Copy)]
#[repr(i32)]
enum Command {
    MoveTo = 1,
    LineTo = 2,
    QuadTo = 3,
    CurveTo = 4,
    Close = 5,
    End = 6,
}

pub struct SegmentPen {
    commands: Vec<i32>,
    coords: Vec<f32>,
    scale: f32,
}

impl SegmentPen {
    pub fn new(units_per_em: f32) -> Self {
        debug_assert!(units_per_em > 0.0, "units_per_em must be positive");
        let scale = units_per_em.recip();
        Self {
            commands: Vec::new(),
            coords: Vec::new(),
            scale,
        }
    }

    pub fn finish(mut self) -> (Vec<i32>, Vec<f32>) {
        self.push(Command::End, [0.0; 6]);
        (self.commands, self.coords)
    }

    fn push(&mut self, command: Command, values: [f32; 6]) {
        self.commands.push(command as i32);
        let scaled = values.map(|value| value * self.scale);
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
