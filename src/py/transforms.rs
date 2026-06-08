use numpy::{IntoPyArray as _, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::{Bound, prelude::*, types::PyModule};
use tiny_skia::FillRule;

use crate::geom::{ElementType, Outline};
use crate::transform::render_bitmap::RenderMode;
use crate::{curves, skia, transform};

#[pyfunction]
pub(crate) fn quad_to_cubic<'py>(
    mut types: PyReadwriteArray1<'py, i64>,
    mut coords: PyReadwriteArray1<'py, f32>,
    seq_len: usize,
) -> PyResult<()> {
    let t = types.as_slice_mut()?;
    let c = coords.as_slice_mut()?;
    curves::quad_to_cubic::quad_to_cubic(t, c, seq_len);
    Ok(())
}

#[pyfunction]
pub(crate) fn quad_to_cubic_and_merge(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    Ok(curves::quad_to_cubic::quad_to_cubic_and_merge(t, c))
}

#[pyfunction]
pub(crate) fn cubic_to_quad(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = Outline::decode(t, c);
    curves::cubic_to_quad::cubic_to_quad(&outline)
        .map(|outline| outline.encode())
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "cubic_to_quad could not approximate a curve within MAX_N segments",
            )
        })
}

#[pyfunction]
pub(crate) fn merge_curves(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = Outline::decode(t, c);
    Ok(curves::merge_curves::merge_curves(&outline).encode())
}

#[pyfunction]
pub(crate) fn normalize_subpath_start_points(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = Outline::decode(t, c);
    Ok(transform::subpath::normalize_subpath_start_points(&outline).encode())
}

#[pyfunction]
pub(crate) fn randomize_subpath_start_points(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
    random_values: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    let r = random_values.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    if r.len() != t.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "random_values length must equal types length",
        ));
    }
    let outline = Outline::decode(t, c);
    let subpath_random_values: Vec<f32> = t
        .iter()
        .zip(r)
        .filter_map(|(&ty, &rv)| (ty == ElementType::MoveTo as i64).then_some(rv))
        .collect();
    Ok(
        transform::subpath::randomize_subpath_start_points(&outline, &subpath_random_values)
            .encode(),
    )
}

#[pyfunction]
pub(crate) fn reverse_closed_subpaths(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = Outline::decode(t, c);
    Ok(transform::subpath::reverse_closed_subpaths(&outline).encode())
}

#[pyfunction]
pub(crate) fn remove_overlaps(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = Outline::decode(t, c);
    Ok(skia::remove_overlaps::remove_overlaps(&outline).encode())
}

#[pyfunction]
pub(crate) fn tight_bbox(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<Option<(f32, f32, f32, f32)>> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = Outline::decode(t, c);
    Ok(crate::geom::bounds_from_outline(&outline).map(|b| (b.x_min, b.y_min, b.x_max, b.y_max)))
}

#[pyfunction]
pub(crate) fn render_bitmap(
    py: Python<'_>,
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
    size: u32,
    mode: &str,
    fill_rule: &str,
) -> PyResult<(Py<PyArray1<u8>>, u32, u32)> {
    if size == 0 || size > 4096 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "size must be between 1 and 4096",
        ));
    }
    let mode = match mode {
        "fixed" => RenderMode::Fixed,
        "bbox" => RenderMode::Bbox,
        "bbox_square" => RenderMode::BboxSquare,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mode must be one of 'fixed', 'bbox', or 'bbox_square'",
            ));
        }
    };
    let fill_type = match fill_rule {
        "winding" => FillRule::Winding,
        "even_odd" => FillRule::EvenOdd,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fill_rule must be 'winding' or 'even_odd'",
            ));
        }
    };
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = Outline::decode(t, c);
    let rendered =
        crate::transform::render_bitmap::render_bitmap(&outline, size, mode, fill_type).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "bbox output dimensions must be between 1 and 4096",
            )
        })?;
    Ok((
        rendered.data.into_pyarray(py).unbind(),
        rendered.width,
        rendered.height,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quad_to_cubic, m)?)?;
    m.add_function(wrap_pyfunction!(quad_to_cubic_and_merge, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_to_quad, m)?)?;
    m.add_function(wrap_pyfunction!(merge_curves, m)?)?;
    m.add_function(wrap_pyfunction!(remove_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_subpath_start_points, m)?)?;
    m.add_function(wrap_pyfunction!(randomize_subpath_start_points, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_closed_subpaths, m)?)?;
    m.add_function(wrap_pyfunction!(tight_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(render_bitmap, m)?)?;
    Ok(())
}

fn ensure_flat_coords_len(types_len: usize, coords_len: usize) -> PyResult<()> {
    if coords_len == types_len * 6 {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "coords length must equal types length times 6",
        ))
    }
}
