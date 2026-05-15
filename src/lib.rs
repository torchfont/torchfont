mod bitmap;
mod bounds;
mod cubic_to_quad;
mod dataset;
mod error;
mod merge_curves;
mod outline;
mod pathops;
mod quad_to_cubic;

use dataset::{GlyphDataset, GlyphItem};
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::{Bound, prelude::*, types::PyModule};

#[pyfunction]
fn render_bitmap(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
    size: u32,
    mode: &str,
) -> PyResult<(Vec<u8>, u32, u32)> {
    if size == 0 || size > 4096 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "size must be between 1 and 4096",
        ));
    }
    let mode = match mode {
        "fixed" => bitmap::RenderMode::Fixed,
        "bbox" => bitmap::RenderMode::Bbox,
        "bbox_square" => bitmap::RenderMode::BboxSquare,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mode must be one of 'fixed', 'bbox', or 'bbox_square'",
            ));
        }
    };
    let rendered = bitmap::render_bitmap(types.as_slice()?, coords.as_slice()?, size, mode)
        .map_err(|err| match err {
            bitmap::RenderBitmapError::BboxTooLarge => pyo3::exceptions::PyValueError::new_err(
                "bbox output dimensions must be between 1 and 4096",
            ),
        })?;
    Ok((rendered.data, rendered.width, rendered.height))
}

#[pyfunction(name = "merge_curves")]
fn py_merge_curves(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    Ok(merge_curves::merge_curves(t, c))
}

#[pyfunction]
fn remove_overlaps(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    Ok(pathops::remove_overlaps(
        types.as_slice()?,
        coords.as_slice()?,
    ))
}

#[pyfunction(name = "quad_to_cubic")]
fn py_quad_to_cubic<'py>(
    mut types: PyReadwriteArray1<'py, i64>,
    mut coords: PyReadwriteArray1<'py, f32>,
    seq_len: usize,
) -> PyResult<()> {
    let t = types.as_slice_mut()?;
    let c = coords.as_slice_mut()?;
    quad_to_cubic::quad_to_cubic(t, c, seq_len);
    Ok(())
}

#[pyfunction(name = "quad_to_cubic_and_merge")]
fn py_quad_to_cubic_and_merge(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    Ok(quad_to_cubic::quad_to_cubic_and_merge(t, c))
}

#[pyfunction(name = "cubic_to_quad")]
fn py_cubic_to_quad(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let t = types.as_slice()?;
    let c = coords.as_slice()?;
    ensure_flat_coords_len(t.len(), c.len())?;
    cubic_to_quad::cubic_to_quad(t, c).map_err(|err| match err {
        cubic_to_quad::CubicToQuadError::ApproximationFailed => {
            pyo3::exceptions::PyValueError::new_err(
                "cubic_to_quad could not approximate a curve within MAX_N segments",
            )
        }
    })
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
    m.add_function(wrap_pyfunction!(py_cubic_to_quad, &m)?)?;
    m.add_function(wrap_pyfunction!(py_merge_curves, &m)?)?;
    m.add_function(wrap_pyfunction!(render_bitmap, &m)?)?;
    m.add_function(wrap_pyfunction!(remove_overlaps, &m)?)?;
    m.add_function(wrap_pyfunction!(py_quad_to_cubic, &m)?)?;
    m.add_function(wrap_pyfunction!(py_quad_to_cubic_and_merge, &m)?)?;
    Ok(())
}
