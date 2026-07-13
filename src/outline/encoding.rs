use super::path::SubpathBuilder;
use super::{Outline, PathElement, Point};

#[derive(Debug)]
pub(crate) enum DecodeError {
    CoordsLen,
    InvalidElementType { index: usize, value: i64 },
    ElementOutsideSubpath { index: usize, value: i64 },
    NonPaddingAfterEnd { index: usize, value: i64 },
}

impl<'a> TryFrom<(&'a [i64], &'a [f32])> for Outline {
    type Error = DecodeError;
    fn try_from((types, coords): (&'a [i64], &'a [f32])) -> Result<Self, Self::Error> {
        if coords.len() != types.len() * 6 {
            return Err(DecodeError::CoordsLen);
        }

        let outline_len = types
            .iter()
            .position(|&value| value == ElementType::End as i64)
            .map_or(types.len(), |index| index + 1);
        if let Some((index, value)) = types[..outline_len]
            .iter()
            .copied()
            .enumerate()
            .find(|&(_, v)| !(1..=ElementType::End as i64).contains(&v))
        {
            return Err(DecodeError::InvalidElementType { index, value });
        }
        let mut in_subpath = false;
        for (index, &value) in types[..outline_len].iter().enumerate() {
            match ElementType::try_from(value) {
                Ok(ElementType::MoveTo) => in_subpath = true,
                Ok(ElementType::LineTo | ElementType::QuadTo | ElementType::CurveTo)
                    if !in_subpath =>
                {
                    return Err(DecodeError::ElementOutsideSubpath { index, value });
                }
                Ok(ElementType::Close) => {
                    if !in_subpath {
                        return Err(DecodeError::ElementOutsideSubpath { index, value });
                    }
                    in_subpath = false;
                }
                Ok(ElementType::End) => break,
                Ok(_) | Err(_) => {}
            }
        }
        if let Some((offset, value)) = types[outline_len..]
            .iter()
            .copied()
            .enumerate()
            .find(|&(_, value)| value != 0)
        {
            return Err(DecodeError::NonPaddingAfterEnd {
                index: outline_len + offset,
                value,
            });
        }

        Ok(Self::decode(
            &types[..outline_len],
            &coords[..outline_len * 6],
        ))
    }
}

impl Outline {
    fn decode(types: &[i64], coords: &[f32]) -> Self {
        debug_assert_eq!(types.len() * 6, coords.len());
        let mut subpaths = Vec::new();
        let mut current: Option<SubpathBuilder> = None;

        for (&ty, values) in types.iter().zip(coords.chunks_exact(6)) {
            match ElementType::try_from(ty) {
                Ok(ElementType::MoveTo) => {
                    if let Some(builder) = current.take() {
                        subpaths.push(builder.finish(false));
                    }
                    current = Some(SubpathBuilder::new(Point::new(values[4], values[5])));
                }
                Ok(ElementType::LineTo) => {
                    if let Some(builder) = &mut current {
                        builder
                            .elements
                            .push(PathElement::LineTo(Point::new(values[4], values[5])));
                    }
                }
                Ok(ElementType::QuadTo) => {
                    if let Some(builder) = &mut current {
                        builder.elements.push(PathElement::QuadTo {
                            control: Point::new(values[0], values[1]),
                            end: Point::new(values[4], values[5]),
                        });
                    }
                }
                Ok(ElementType::CurveTo) => {
                    if let Some(builder) = &mut current {
                        builder.elements.push(PathElement::CurveTo {
                            control0: Point::new(values[0], values[1]),
                            control1: Point::new(values[2], values[3]),
                            end: Point::new(values[4], values[5]),
                        });
                    }
                }
                Ok(ElementType::Close) => {
                    if let Some(builder) = current.take() {
                        subpaths.push(builder.finish(true));
                    }
                }
                Ok(ElementType::End) | Err(_) => break,
            }
        }
        if let Some(builder) = current {
            subpaths.push(builder.finish(false));
        }
        Self { subpaths }
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
        push(&mut types, &mut coords, ElementType::End, [0.0; 6]);
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

impl TryFrom<i64> for ElementType {
    type Error = ();
    fn try_from(v: i64) -> Result<Self, Self::Error> {
        Ok(match v {
            1 => Self::MoveTo,
            2 => Self::LineTo,
            3 => Self::QuadTo,
            4 => Self::CurveTo,
            5 => Self::Close,
            6 => Self::End,
            _ => return Err(()),
        })
    }
}

fn push_endpoint(types: &mut Vec<i64>, coords: &mut Vec<f32>, ty: ElementType, point: Point) {
    push(types, coords, ty, [0.0, 0.0, 0.0, 0.0, point.x, point.y]);
}

fn push(types: &mut Vec<i64>, coords: &mut Vec<f32>, ty: ElementType, values: [f32; 6]) {
    types.push(ty as i64);
    coords.extend_from_slice(&values);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_from_strips_padding_after_end() {
        let types = [ElementType::MoveTo as i64, ElementType::End as i64, 0, 0];
        let coords = [0.0; 24];

        let outline = Outline::try_from((types.as_slice(), coords.as_slice())).unwrap();

        assert_eq!(
            outline.encode().0,
            [ElementType::MoveTo as i64, ElementType::End as i64]
        );
    }

    #[test]
    fn try_from_rejects_padding_before_end() {
        let types = [ElementType::MoveTo as i64, 0, ElementType::End as i64];
        let coords = [0.0; 18];

        assert!(matches!(
            Outline::try_from((types.as_slice(), coords.as_slice())),
            Err(DecodeError::InvalidElementType { index: 1, value: 0 })
        ));
    }

    #[test]
    fn try_from_rejects_non_padding_after_end() {
        let types = [
            ElementType::MoveTo as i64,
            ElementType::End as i64,
            ElementType::LineTo as i64,
        ];
        let coords = [0.0; 18];

        assert!(matches!(
            Outline::try_from((types.as_slice(), coords.as_slice())),
            Err(DecodeError::NonPaddingAfterEnd { index: 2, value: 2 })
        ));
    }

    #[test]
    fn try_from_rejects_drawing_element_before_move() {
        let types = [ElementType::LineTo as i64, ElementType::End as i64];
        let coords = [0.0; 12];

        assert!(matches!(
            Outline::try_from((types.as_slice(), coords.as_slice())),
            Err(DecodeError::ElementOutsideSubpath { index: 0, value: 2 })
        ));
    }

    #[test]
    fn try_from_rejects_drawing_element_after_close() {
        let types = [
            ElementType::MoveTo as i64,
            ElementType::Close as i64,
            ElementType::LineTo as i64,
            ElementType::End as i64,
        ];
        let coords = [0.0; 24];

        assert!(matches!(
            Outline::try_from((types.as_slice(), coords.as_slice())),
            Err(DecodeError::ElementOutsideSubpath { index: 2, value: 2 })
        ));
    }
}
