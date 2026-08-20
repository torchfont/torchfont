use super::BezPath;
use kurbo::{PathEl, Point};

#[derive(Debug)]
pub(crate) enum DecodeError {
    CoordsLen,
    InvalidElementType { index: usize, value: i64 },
    ElementOutsideSubpath { index: usize, value: i64 },
    NonPaddingAfterEnd { index: usize, value: i64 },
}

pub(crate) fn decode(types: &[i64], coords: &[f32]) -> Result<BezPath, DecodeError> {
    if coords.len() != types.len() * 6 {
        return Err(DecodeError::CoordsLen);
    }
    let len = types
        .iter()
        .position(|&v| v == ElementType::End as i64)
        .map_or(types.len(), |i| i + 1);
    if let Some((index, value)) = types[..len]
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, v)| !(1..=ElementType::End as i64).contains(&v))
    {
        return Err(DecodeError::InvalidElementType { index, value });
    }
    let mut inside = false;
    for (index, &value) in types[..len].iter().enumerate() {
        match ElementType::try_from(value) {
            Ok(ElementType::MoveTo) => inside = true,
            Ok(ElementType::LineTo | ElementType::QuadTo | ElementType::CurveTo) if !inside => {
                return Err(DecodeError::ElementOutsideSubpath { index, value });
            }
            Ok(ElementType::Close) => {
                if !inside {
                    return Err(DecodeError::ElementOutsideSubpath { index, value });
                }
                inside = false;
            }
            Ok(ElementType::End) => break,
            Ok(_) | Err(_) => {}
        }
    }
    if let Some((offset, value)) = types[len..]
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, v)| v != 0)
    {
        return Err(DecodeError::NonPaddingAfterEnd {
            index: len + offset,
            value,
        });
    }
    let mut path = BezPath::new();
    for (&ty, v) in types[..len].iter().zip(coords[..len * 6].chunks_exact(6)) {
        match ElementType::try_from(ty) {
            Ok(ElementType::MoveTo) => path.move_to(point(v[4], v[5])),
            Ok(ElementType::LineTo) => path.line_to(point(v[4], v[5])),
            Ok(ElementType::QuadTo) => path.quad_to(point(v[0], v[1]), point(v[4], v[5])),
            Ok(ElementType::CurveTo) => {
                path.curve_to(point(v[0], v[1]), point(v[2], v[3]), point(v[4], v[5]))
            }
            Ok(ElementType::Close) => path.close_path(),
            Ok(ElementType::End) | Err(_) => break,
        }
    }
    Ok(path)
}

pub(crate) fn encode(path: &BezPath) -> (Vec<i64>, Vec<f32>) {
    let mut types = Vec::with_capacity(path.elements().len() + 1);
    let mut coords = Vec::with_capacity((path.elements().len() + 1) * 6);
    for element in path.elements() {
        match *element {
            PathEl::MoveTo(p) => push_endpoint(&mut types, &mut coords, ElementType::MoveTo, p),
            PathEl::LineTo(p) => push_endpoint(&mut types, &mut coords, ElementType::LineTo, p),
            PathEl::QuadTo(c, p) => push(
                &mut types,
                &mut coords,
                ElementType::QuadTo,
                [c.x as f32, c.y as f32, 0.0, 0.0, p.x as f32, p.y as f32],
            ),
            PathEl::CurveTo(c0, c1, p) => push(
                &mut types,
                &mut coords,
                ElementType::CurveTo,
                [
                    c0.x as f32,
                    c0.y as f32,
                    c1.x as f32,
                    c1.y as f32,
                    p.x as f32,
                    p.y as f32,
                ],
            ),
            PathEl::ClosePath => push(&mut types, &mut coords, ElementType::Close, [0.0; 6]),
        }
    }
    push(&mut types, &mut coords, ElementType::End, [0.0; 6]);
    (types, coords)
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
fn point(x: f32, y: f32) -> Point {
    Point::new(x.into(), y.into())
}
fn push_endpoint(types: &mut Vec<i64>, coords: &mut Vec<f32>, ty: ElementType, p: Point) {
    push(
        types,
        coords,
        ty,
        [0.0, 0.0, 0.0, 0.0, p.x as f32, p.y as f32],
    );
}
fn push(types: &mut Vec<i64>, coords: &mut Vec<f32>, ty: ElementType, values: [f32; 6]) {
    types.push(ty as i64);
    coords.extend_from_slice(&values);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strips_padding_after_end() {
        let outline = decode(&[1, 6, 0], &[0.0; 18]).unwrap();
        assert_eq!(encode(&outline).0, [1, 6]);
    }

    #[test]
    fn rejects_drawing_before_move() {
        assert!(matches!(
            decode(&[2, 6], &[0.0; 12]),
            Err(DecodeError::ElementOutsideSubpath { .. })
        ));
    }

    #[test]
    fn rejects_padding_before_end() {
        assert!(matches!(
            decode(&[1, 0, 6], &[0.0; 18]),
            Err(DecodeError::InvalidElementType { .. })
        ));
    }

    #[test]
    fn rejects_nonpadding_after_end() {
        assert!(matches!(
            decode(&[1, 6, 2], &[0.0; 18]),
            Err(DecodeError::NonPaddingAfterEnd { .. })
        ));
    }

    #[test]
    fn rejects_drawing_after_close() {
        assert!(matches!(
            decode(&[1, 5, 2, 6], &[0.0; 24]),
            Err(DecodeError::ElementOutsideSubpath { .. })
        ));
    }
}
