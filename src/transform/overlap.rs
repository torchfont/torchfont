use kurbo::Shape;
use skia_safe::{PathFillType, PathOp};

use super::skia::{PATHOPS_SCALE, build_skia_path_builder};
use crate::outline::{BezPath, PathEl, subpath_start};

pub(super) fn overlap_groups(
    paths: &[&[PathEl]],
    areas: &[(f64, f64)],
) -> Result<Vec<Vec<usize>>, &'static str> {
    if paths.len() < 2 {
        return Ok((0..paths.len()).map(|index| vec![index]).collect());
    }
    let mut bounds = paths[0].bounding_box();
    for path in &paths[1..] {
        bounds = bounds.union(path.bounding_box());
    }
    let scale_x = if bounds.width() > 0.0 {
        f64::from(PATHOPS_SCALE) / bounds.width()
    } else {
        1.0
    };
    let scale_y = if bounds.height() > 0.0 {
        f64::from(PATHOPS_SCALE) / bounds.height()
    } else {
        1.0
    };
    let transform = kurbo::Affine::scale_non_uniform(scale_x, scale_y)
        * kurbo::Affine::translate((-bounds.x0, -bounds.y0));
    let filled: Vec<_> = paths
        .iter()
        .zip(areas)
        .map(|(path, &(area, _))| {
            let path = BezPath::from_vec(path.to_vec());
            let path = if area < 0.0 {
                path.reverse_subpaths()
            } else {
                path
            };
            let elements: Vec<_> = path.iter().map(|element| transform * element).collect();
            let mut builder = build_skia_path_builder(&elements, PathFillType::Winding);
            builder.close();
            builder.detach()
        })
        .collect();
    let mut roots: Vec<_> = (0..paths.len()).collect();
    for left in 0..paths.len() {
        for right in left + 1..paths.len() {
            let left_root = root(&mut roots, left);
            let right_root = root(&mut roots, right);
            if left_root == right_root {
                continue;
            }
            let intersection = filled[left]
                .op(&filled[right], PathOp::Intersect)
                .or_else(|| {
                    filled[left].try_make_scale((1.0, -1.0))?.op(
                        &filled[right].try_make_scale((1.0, -1.0))?,
                        PathOp::Intersect,
                    )
                })
                .ok_or("could not intersect contour fills for overlap groups")?;
            if !intersection.is_empty() {
                roots[left_root.max(right_root)] = left_root.min(right_root);
            }
        }
    }
    let mut members = vec![Vec::new(); paths.len()];
    for index in 0..paths.len() {
        let group = root(&mut roots, index);
        members[group].push(index);
    }
    Ok(members
        .into_iter()
        .filter(|group| !group.is_empty())
        .collect())
}

fn root(roots: &mut [usize], mut index: usize) -> usize {
    while roots[index] != index {
        roots[index] = roots[roots[index]];
        index = roots[index];
    }
    index
}

pub(super) fn subpath_area(subpath: &[PathEl]) -> (f64, f64) {
    let mut path = BezPath::from_vec(subpath.to_vec());
    path.apply_affine(kurbo::Affine::translate(-subpath_start(subpath).to_vec2()));
    let bounds = path.control_box();
    let extent_x = bounds.x0.abs().max(bounds.x1.abs());
    let extent_y = bounds.y0.abs().max(bounds.y1.abs());
    let area = path.area();
    let error = 32.0 * f64::EPSILON * extent_x * extent_y * subpath.len() as f64;
    if area.abs() <= error {
        (0.0, error)
    } else {
        (area, error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outline::{Point, subpath_from_elements};
    use kurbo::Rect;

    fn pt(x: f32, y: f32) -> Point {
        Point::new(x.into(), y.into())
    }

    fn line(x: f32, y: f32) -> PathEl {
        PathEl::LineTo(pt(x, y))
    }

    fn closed(start: Point, elements: Vec<PathEl>) -> BezPath {
        subpath_from_elements(start, elements, true)
    }

    fn rectangle(rect: Rect) -> BezPath {
        rect.to_path(0.1)
    }

    #[test]
    fn groups_fills_with_containment_and_transitive_overlaps() {
        let paths = [
            rectangle(Rect::new(0.0, 0.0, 10.0, 10.0)),
            rectangle(Rect::new(2.0, 2.0, 8.0, 8.0)),
            rectangle(Rect::new(10.0, 0.0, 12.0, 10.0)),
            rectangle(Rect::new(9.0, 4.0, 11.0, 6.0)),
            rectangle(Rect::new(20.0, 0.0, 21.0, 1.0)),
        ];
        let paths: Vec<_> = paths.iter().map(|p| p.elements()).collect();
        let areas: Vec<_> = paths.iter().map(|path| subpath_area(path)).collect();
        assert_eq!(
            overlap_groups(&paths, &areas).unwrap(),
            vec![vec![0, 1, 2, 3], vec![4]]
        );
    }

    #[test]
    fn edge_contacts_and_zero_width_bounds_remain_independent() {
        let paths = [
            rectangle(Rect::new(0.0, 0.0, 1.0, 1.0)),
            rectangle(Rect::new(1.0, 0.0, 2.0, 1.0)),
            closed(pt(0.5, 0.2), vec![line(0.5, 0.8)]),
        ];
        let paths: Vec<_> = paths.iter().map(|p| p.elements()).collect();
        let areas: Vec<_> = paths.iter().map(|path| subpath_area(path)).collect();
        assert_eq!(
            overlap_groups(&paths, &areas).unwrap(),
            vec![vec![0], vec![1], vec![2]]
        );
    }

    #[test]
    fn subpath_area_treats_open_subpath_as_implicitly_closed() {
        let mut open = BezPath::new();
        open.move_to((1.0, 1.0));
        open.line_to((5.0, 1.0));
        open.line_to((1.0, 5.0));

        let raw_area = open.elements().area();
        let (closed_area, _) = super::subpath_area(open.elements());

        assert_ne!(raw_area, closed_area);
        assert!((closed_area.abs() - 8.0).abs() < 1e-9);
    }
}
