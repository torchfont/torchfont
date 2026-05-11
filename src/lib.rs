mod bitmap;
mod dataset;
mod error;
mod outline;
mod pen;

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

#[pyfunction]
fn quad_to_cubic_inplace<'py>(
    mut types: PyReadwriteArray1<'py, i64>,
    mut coords: PyReadwriteArray1<'py, f32>,
    seq_len: usize,
) -> PyResult<()> {
    let t = types.as_slice_mut()?;
    let c = coords.as_slice_mut()?;
    outline::quad_to_cubic(t, c, seq_len);
    Ok(())
}

#[pymodule]
fn _torchfont(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GlyphDataset>()?;
    m.add_class::<GlyphItem>()?;
    m.add_function(wrap_pyfunction!(render_bitmap, &m)?)?;
    m.add_function(wrap_pyfunction!(quad_to_cubic_inplace, &m)?)?;
    Ok(())
}
