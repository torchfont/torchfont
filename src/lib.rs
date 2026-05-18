mod bounds;
mod dataset;
mod error;
mod outline;
mod transform;

use dataset::{GlyphDataset, GlyphItem};
use numpy::{IntoPyArray as _, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::{Bound, prelude::*, types::PyModule};

#[pyfunction]
fn quad_to_cubic<'py>(
    mut types: PyReadwriteArray1<'py, i64>,
    mut coords: PyReadwriteArray1<'py, f32>,
    seq_len: usize,
) -> PyResult<()> {
    let t = types.as_slice_mut()?;
    let c = coords.as_slice_mut()?;
    transform::quad_to_cubic::quad_to_cubic(t, c, seq_len);
    Ok(())
}

#[pyfunction]
fn quad_to_cubic_and_merge(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    Ok(transform::quad_to_cubic::quad_to_cubic_and_merge(t, c))
}

#[pyfunction]
fn cubic_to_quad(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = outline::Outline::decode(t, c);
    transform::cubic_to_quad::cubic_to_quad(&outline)
        .map(|outline| outline.encode())
        .map_err(|err| match err {
            transform::cubic_to_quad::CubicToQuadError::ApproximationFailed => {
                pyo3::exceptions::PyValueError::new_err(
                    "cubic_to_quad could not approximate a curve within MAX_N segments",
                )
            }
        })
}

#[pyfunction]
fn merge_curves(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = outline::Outline::decode(t, c);
    Ok(transform::merge_curves::merge_curves(&outline).encode())
}

#[pyfunction]
fn normalize_subpath_start_points(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = outline::Outline::decode(t, c);
    Ok(transform::subpath::normalize_subpath_start_points(&outline).encode())
}

#[pyfunction]
fn randomize_subpath_start_points(
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
    let outline = outline::Outline::decode(t, c);
    let subpath_random_values = t
        .iter()
        .enumerate()
        .filter_map(|(idx, &ty)| (ty == outline::ElementType::MoveTo as i64).then_some(r[idx]))
        .collect::<Vec<_>>();
    Ok(
        transform::subpath::randomize_subpath_start_points(&outline, &subpath_random_values)
            .encode(),
    )
}

#[pyfunction]
fn reverse_closed_subpaths(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = outline::Outline::decode(t, c);
    Ok(transform::subpath::reverse_closed_subpaths(&outline).encode())
}

#[pyfunction]
fn remove_overlaps(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = outline::Outline::decode(t, c);
    Ok(transform::remove_overlaps::remove_overlaps(&outline).encode())
}

#[pyfunction]
fn tight_bbox(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<Option<(f32, f32, f32, f32)>> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = outline::Outline::decode(t, c);
    Ok(bounds::bounds_from_outline(&outline).map(|b| (b.x_min, b.y_min, b.x_max, b.y_max)))
}

#[pyfunction]
fn render_bitmap(
    py: Python<'_>,
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
    size: u32,
    mode: &str,
) -> PyResult<(Py<PyArray1<u8>>, u32, u32)> {
    if size == 0 || size > 4096 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "size must be between 1 and 4096",
        ));
    }
    let mode = match mode {
        "fixed" => transform::render_bitmap::RenderMode::Fixed,
        "bbox" => transform::render_bitmap::RenderMode::Bbox,
        "bbox_square" => transform::render_bitmap::RenderMode::BboxSquare,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mode must be one of 'fixed', 'bbox', or 'bbox_square'",
            ));
        }
    };
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    let outline = outline::Outline::decode(t, c);
    let rendered =
        transform::render_bitmap::render_bitmap(&outline, size, mode).map_err(|err| match err {
            transform::render_bitmap::RenderBitmapError::BboxTooLarge => {
                pyo3::exceptions::PyValueError::new_err(
                    "bbox output dimensions must be between 1 and 4096",
                )
            }
        })?;
    Ok((
        rendered.data.into_pyarray(py).unbind(),
        rendered.width,
        rendered.height,
    ))
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

#[pymodule]
fn _torchfont(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GlyphDataset>()?;
    m.add_class::<GlyphItem>()?;
    m.add_function(wrap_pyfunction!(quad_to_cubic, &m)?)?;
    m.add_function(wrap_pyfunction!(quad_to_cubic_and_merge, &m)?)?;
    m.add_function(wrap_pyfunction!(cubic_to_quad, &m)?)?;
    m.add_function(wrap_pyfunction!(merge_curves, &m)?)?;
    m.add_function(wrap_pyfunction!(remove_overlaps, &m)?)?;
    m.add_function(wrap_pyfunction!(normalize_subpath_start_points, &m)?)?;
    m.add_function(wrap_pyfunction!(randomize_subpath_start_points, &m)?)?;
    m.add_function(wrap_pyfunction!(reverse_closed_subpaths, &m)?)?;
    m.add_function(wrap_pyfunction!(tight_bbox, &m)?)?;
    m.add_function(wrap_pyfunction!(render_bitmap, &m)?)?;
    Ok(())
}
