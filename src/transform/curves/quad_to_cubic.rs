use crate::outline::{Outline, PathElement, Subpath};

pub(crate) fn quad_to_cubic(outline: &Outline) -> Outline {
    let subpaths = outline
        .subpaths()
        .iter()
        .map(|subpath| {
            let mut prev = subpath.start();
            let elements = subpath
                .elements()
                .iter()
                .map(|&el| {
                    let converted = if let PathElement::QuadTo { control, end } = el {
                        let c1 = prev.lerp(control, 2.0 / 3.0);
                        let c2 = end.lerp(control, 2.0 / 3.0);
                        PathElement::CurveTo {
                            control0: c1,
                            control1: c2,
                            end,
                        }
                    } else {
                        el
                    };
                    prev = converted.end();
                    converted
                })
                .collect();
            Subpath::new(subpath.start(), elements, subpath.is_closed())
        })
        .collect();
    Outline::new(subpaths)
}
