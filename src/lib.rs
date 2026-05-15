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
    transform::cubic_to_quad::cubic_to_quad(t, c).map_err(|err| match err {
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
    Ok(transform::merge_curves::merge_curves(t, c))
}

#[pyfunction]
fn remove_overlaps(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    Ok(transform::remove_overlaps::remove_overlaps(t, c))
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
    let rendered =
        transform::render_bitmap::render_bitmap(types.as_slice()?, coords.as_slice()?, size, mode)
            .map_err(|err| match err {
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
    m.add_function(wrap_pyfunction!(render_bitmap, &m)?)?;
    Ok(())
}
