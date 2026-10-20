use kurbo::Shape;
use petgraph::unionfind::UnionFind;
use skia_safe::PathOp;

use super::skia::{SKIA_SCALE, build_skia_path};
use crate::outline::{BezPath, PathEl, subpath_start};

pub(super) fn overlap_groups(paths: &[&[PathEl]], areas: &[(f64, f64)]) -> Vec<Vec<usize>> {
    if paths.len() < 2 {
        return (0..paths.len()).map(|index| vec![index]).collect();
    }
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
            build_skia_path(path.elements()).try_make_scale((SKIA_SCALE, SKIA_SCALE))
        })
        .collect();
    bounds_groups(paths, false, |left, right| {
        let (Some(left), Some(right)) = (&filled[left], &filled[right]) else {
            return true;
        };
        left.op(right, PathOp::Intersect)
            .is_none_or(|intersection| !intersection.is_empty())
    })
}

pub(super) fn bounds_groups(
    paths: &[&[PathEl]],
    touching: bool,
    mut overlap: impl FnMut(usize, usize) -> bool,
) -> Vec<Vec<usize>> {
    let bounds: Vec<_> = paths.iter().map(|path| path.bounding_box()).collect();
    let mut roots = UnionFind::new(bounds.len());
    let mut ordered: Vec<_> = (0..bounds.len()).collect();
    ordered.sort_unstable_by(|&left, &right| bounds[left].x0.total_cmp(&bounds[right].x0));
    for (position, &left) in ordered.iter().enumerate() {
        for &right in &ordered[position + 1..] {
            if bounds[right].x0 > bounds[left].x1 {
                break;
            }
            let candidate = if touching {
                bounds[left].overlaps(bounds[right])
            } else {
                !bounds[left].intersect(bounds[right]).is_zero_area()
            };
            if candidate && !roots.equiv(left, right) && overlap(left, right) {
                roots.union(left, right);
            }
        }
    }
    let labels = roots.into_labeling();
    let mut members = vec![Vec::new(); labels.len()];
    for (index, group) in labels.into_iter().enumerate() {
        members[group].push(index);
    }
    let mut groups: Vec<_> = members
        .into_iter()
        .filter(|group| !group.is_empty())
        .collect();
    groups.sort_unstable_by_key(|group| group[0]);
    groups
}

pub(super) fn subpath_area(subpath: &[PathEl]) -> (f64, f64) {
    let mut path = BezPath::from_vec(subpath.to_vec());
    path.apply_affine(kurbo::Affine::translate(-subpath_start(subpath).to_vec2()));
    let bounds = path.control_box();
    let extent_x = bounds.x0.abs().max(bounds.x1.abs());
    let extent_y = bounds.y0.abs().max(bounds.y1.abs());
    let area = path.area();
    let error = 32.0 * f64::EPSILON * extent_x * extent_y * subpath.len() as f64;
    (area, error)
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

    fn curve(x0: f32, y0: f32, x1: f32, y1: f32, x: f32, y: f32) -> PathEl {
        PathEl::CurveTo(pt(x0, y0), pt(x1, y1), pt(x, y))
    }

    fn closed(start: Point, elements: Vec<PathEl>) -> BezPath {
        subpath_from_elements(start, elements, true)
    }

    fn rectangle(rect: Rect) -> BezPath {
        rect.to_path(0.1)
    }

    #[test]
    fn contours_that_pathops_cannot_intersect_join_one_group() {
        let paths = [
            closed(
                pt(0.742_541_2, 0.668_623),
                vec![
                    line(0.769_798_46, 0.630_166_05),
                    curve(
                        0.737_004_76,
                        0.594_638_2,
                        0.686_106_15,
                        0.552_403,
                        0.648_498_5,
                        0.522_434_3,
                    ),
                    line(0.597_005_3, 0.583_727_54),
                    curve(
                        0.637_774_94,
                        0.600_433_5,
                        0.651_300_8,
                        0.614_271_6,
                        0.683_370_05,
                        0.634_912_3,
                    ),
                    curve(
                        0.702_241_7,
                        0.650_161_6,
                        0.709_872_25,
                        0.671_301_4,
                        0.728_635_8,
                        0.688_607_16,
                    ),
                    line(0.735_493_54, 0.655_443_1),
                ],
            ),
            closed(
                pt(0.575_259_4, 0.770_773_35),
                vec![
                    curve(
                        0.591_229_26,
                        0.757_481_5,
                        0.616_278,
                        0.725_683_3,
                        0.714_425_7,
                        0.633_487_46,
                    ),
                    line(0.676_852_4, 0.576_843_5),
                    line(0.545_650_36, 0.685_919),
                    line(0.529_213_5, 0.717_811_9),
                    line(0.552_695_3, 0.779_385_7),
                    curve(
                        0.560_307_6,
                        0.786_794,
                        0.561_504_66,
                        0.771_635_65,
                        0.562_448_44,
                        0.772_919_24,
                    ),
                    curve(
                        0.567_667_96,
                        0.776_633,
                        0.548_577_1,
                        0.778_467_1,
                        0.576_196_4,
                        0.763_719_4,
                    ),
                ],
            ),
        ];
        let paths: Vec<_> = paths.iter().map(|p| p.elements()).collect();
        let areas: Vec<_> = paths.iter().map(|path| subpath_area(path)).collect();
        let fills: Vec<_> = paths
            .iter()
            .zip(&areas)
            .map(|(path, &(area, _))| {
                let path = BezPath::from_vec(path.to_vec());
                let path = if area < 0.0 {
                    path.reverse_subpaths()
                } else {
                    path
                };
                build_skia_path(path.elements())
                    .try_make_scale((SKIA_SCALE, SKIA_SCALE))
                    .unwrap()
            })
            .collect();

        assert!(fills[0].op(&fills[1], PathOp::Intersect).is_none());
        assert_eq!(overlap_groups(&paths, &areas), vec![vec![0, 1]]);
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
            overlap_groups(&paths, &areas),
            vec![vec![0, 1, 2, 3], vec![4]]
        );
    }

    #[test]
    fn edge_contacts_and_zero_width_bounds_remain_independent() {
        let paths = [
            rectangle(Rect::new(0.0, 0.0, 1.0, 1.0)),
            rectangle(Rect::new(1.0, 0.0, 2.0, 1.0)),
            closed(pt(0.5, 0.2), vec![line(0.5, 0.8)]),
            rectangle(Rect::new(0.0, 2.0, 1.0, 3.0)),
            closed(
                pt(-2.0, 0.0),
                vec![
                    line(-1.0, 0.0),
                    line(-1.0, 1.0),
                    line(-2.0, 1.0),
                    line(-2.0, 0.0),
                    line(0.5, 0.5),
                    line(-2.0, 0.0),
                ],
            ),
        ];
        let paths: Vec<_> = paths.iter().map(|p| p.elements()).collect();
        let areas: Vec<_> = paths.iter().map(|path| subpath_area(path)).collect();
        assert_eq!(
            overlap_groups(&paths, &areas),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
        );
    }

    #[test]
    fn subpath_area_retains_residuals_within_roundoff() {
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        for point in [
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 1e-14),
            (0.0, 1e-14),
        ] {
            path.line_to(point);
        }
        path.close_path();

        let (area, error) = subpath_area(path.elements());

        assert!(area > 0.0);
        assert!(area < error);
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
