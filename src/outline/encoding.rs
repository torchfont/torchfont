use super::model::SubpathBuilder;
use super::{Outline, PathElement, Point};

impl Outline {
    pub(crate) fn decode(types: &[i64], coords: &[f32]) -> Self {
        debug_assert_eq!(types.len() * 6, coords.len());
        let mut subpaths = Vec::new();
        let mut current: Option<SubpathBuilder> = None;

        let mut terminated = false;
        for (&ty, values) in types.iter().zip(coords.chunks_exact(6)) {
            match ty {
                v if v == ElementType::MoveTo as i64 => {
                    if let Some(builder) = current.take() {
                        subpaths.push(builder.finish(false));
                    }
                    current = Some(SubpathBuilder::new(Point::new(values[4], values[5])));
                }
                v if v == ElementType::LineTo as i64 => {
                    if let Some(builder) = &mut current {
                        builder
                            .elements
                            .push(PathElement::LineTo(Point::new(values[4], values[5])));
                    }
                }
                v if v == ElementType::QuadTo as i64 => {
                    if let Some(builder) = &mut current {
                        builder.elements.push(PathElement::QuadTo {
                            control: Point::new(values[0], values[1]),
                            end: Point::new(values[4], values[5]),
                        });
                    }
                }
                v if v == ElementType::CurveTo as i64 => {
                    if let Some(builder) = &mut current {
                        builder.elements.push(PathElement::CurveTo {
                            control0: Point::new(values[0], values[1]),
                            control1: Point::new(values[2], values[3]),
                            end: Point::new(values[4], values[5]),
                        });
                    }
                }
                v if v == ElementType::Close as i64 => {
                    if let Some(builder) = current.take() {
                        subpaths.push(builder.finish(true));
                    }
                }
                v if v == ElementType::End as i64 => {
                    terminated = true;
                    break;
                }
                _ => break,
            }
        }
        if let Some(builder) = current {
            subpaths.push(builder.finish(false));
        }
        Self {
            subpaths,
            terminated,
        }
    }

    pub(crate) fn encode(&self) -> (Vec<i64>, Vec<f32>) {
        let element_count: usize = self
            .subpaths
            .iter()
            .map(|subpath| 1 + subpath.elements.len() + usize::from(subpath.closed))
            .sum();
        let mut types = Vec::with_capacity(element_count + 1);
        let mut coords = Vec::with_capacity((element_count + 1) * 6);
        for subpath in &self.subpaths {
            push_endpoint(&mut types, &mut coords, ElementType::MoveTo, subpath.start);
            for element in &subpath.elements {
                match *element {
                    PathElement::LineTo(point) => {
                        push_endpoint(&mut types, &mut coords, ElementType::LineTo, point)
                    }
                    PathElement::QuadTo { control, end } => push(
                        &mut types,
                        &mut coords,
                        ElementType::QuadTo,
                        [control.x, control.y, 0.0, 0.0, end.x, end.y],
                    ),
                    PathElement::CurveTo {
                        control0,
                        control1,
                        end,
                    } => push(
                        &mut types,
                        &mut coords,
                        ElementType::CurveTo,
                        [control0.x, control0.y, control1.x, control1.y, end.x, end.y],
                    ),
                }
            }
            if subpath.closed {
                push(&mut types, &mut coords, ElementType::Close, [0.0; 6]);
            }
        }
        if self.terminated {
            push(&mut types, &mut coords, ElementType::End, [0.0; 6]);
        }
        (types, coords)
    }
}

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

fn push_endpoint(types: &mut Vec<i64>, coords: &mut Vec<f32>, ty: ElementType, point: Point) {
    push(types, coords, ty, [0.0, 0.0, 0.0, 0.0, point.x, point.y]);
}

fn push(types: &mut Vec<i64>, coords: &mut Vec<f32>, ty: ElementType, values: [f32; 6]) {
    types.push(ty as i64);
    coords.extend_from_slice(&values);
}
