use numpy::{IntoPyArray as _, PyArray1, PyReadonlyArray1};
use pyo3::{Bound, prelude::*, types::PyModule};
use tiny_skia::FillRule;

use crate::outline::{DecodeError, Outline};
use crate::transform::render_bitmap::RenderMode;
use crate::transform::{curves, subpath};

mod load;

type OutlineArrays<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>);

fn decode(types: &[i64], coords: &[f32]) -> PyResult<Outline> {
    Outline::try_from((types, coords)).map_err(|e| match e {
        DecodeError::CoordsLen => {
            pyo3::exceptions::PyValueError::new_err("coords length must equal types length times 6")
        }
        DecodeError::InvalidElementType { index, value } => {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid element type {value} at index {index}"
            ))
        }
        DecodeError::ElementOutsideSubpath { index, value } => {
            pyo3::exceptions::PyValueError::new_err(format!(
                "element type {value} at index {index} requires a preceding MOVE_TO"
            ))
        }
        DecodeError::NonPaddingAfterEnd { index, value } => {
            pyo3::exceptions::PyValueError::new_err(format!(
                "only PAD elements may follow END; found {value} at index {index}"
            ))
        }
    })
}

fn encode<'py>(py: Python<'py>, outline: &Outline) -> OutlineArrays<'py> {
    let (types, coords) = outline.encode();
    (types.into_pyarray(py), coords.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn quad_to_cubic<'py>(
    py: Python<'py>,
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
    merge_curves: bool,
) -> PyResult<OutlineArrays<'py>> {
    let outline = decode(types.as_slice()?, coords.as_slice()?)?;
    let result = curves::quad_to_cubic::quad_to_cubic(&outline);
    let result = if merge_curves {
        crate::transform::curves::merge_curves::merge_curves(&result)
    } else {
        result
    };
    Ok(encode(py, &result))
}

#[pyfunction]
pub(crate) fn cubic_to_quad<'py>(
    py: Python<'py>,
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<OutlineArrays<'py>> {
    let outline = decode(types.as_slice()?, coords.as_slice()?)?;
    curves::cubic_to_quad::cubic_to_quad(&outline)
        .map(|outline| encode(py, &outline))
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "cubic_to_quad could not approximate a curve within MAX_N segments",
            )
        })
}

#[pyfunction]
pub(crate) fn merge_curves<'py>(
    py: Python<'py>,
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<OutlineArrays<'py>> {
    let outline = decode(types.as_slice()?, coords.as_slice()?)?;
    Ok(encode(py, &curves::merge_curves::merge_curves(&outline)))
}

#[pyfunction]
pub(crate) fn normalize_subpath_start_points<'py>(
    py: Python<'py>,
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<OutlineArrays<'py>> {
    let outline = decode(types.as_slice()?, coords.as_slice()?)?;
    Ok(encode(
        py,
        &subpath::normalize_subpath_start_points(&outline),
    ))
}

#[pyfunction]
pub(crate) fn randomize_subpath_start_points<'py>(
    py: Python<'py>,
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
    random_values: PyReadonlyArray1<'_, f32>,
) -> PyResult<OutlineArrays<'py>> {
    use crate::outline::ElementType;
    let t = types.as_slice()?;
    let r = random_values.as_slice()?;
    let outline = decode(t, coords.as_slice()?)?;
    if r.len() < t.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "random_values length must be at least types length",
        ));
    }
    let subpath_random_values: Vec<f32> = t
        .iter()
        .zip(r)
        .filter_map(|(&ty, &rv)| (ty == ElementType::MoveTo as i64).then_some(rv))
        .collect();
    Ok(encode(
        py,
        &subpath::randomize_subpath_start_points(&outline, &subpath_random_values),
    ))
}

#[pyfunction]
pub(crate) fn reverse_closed_subpaths<'py>(
    py: Python<'py>,
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<OutlineArrays<'py>> {
    let outline = decode(types.as_slice()?, coords.as_slice()?)?;
    Ok(encode(py, &subpath::reverse_closed_subpaths(&outline)))
}

#[pyfunction]
pub(crate) fn remove_overlaps<'py>(
    py: Python<'py>,
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<OutlineArrays<'py>> {
    let outline = decode(types.as_slice()?, coords.as_slice()?)?;
    Ok(encode(
        py,
        &crate::transform::remove_overlaps::remove_overlaps(&outline),
    ))
}

#[pyfunction]
pub(crate) fn tight_bbox(
    types: PyReadonlyArray1<'_, i64>,
    coords: PyReadonlyArray1<'_, f32>,
) -> PyResult<Option<(f32, f32, f32, f32)>> {
    let outline = decode(types.as_slice()?, coords.as_slice()?)?;
    Ok(crate::outline::bounds_from_outline(&outline).map(|b| (b.x_min, b.y_min, b.x_max, b.y_max)))
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
    let fill_rule = match fill_rule {
        "winding" => FillRule::Winding,
        "even_odd" => FillRule::EvenOdd,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fill_rule must be 'winding' or 'even_odd'",
            ));
        }
    };
    let outline = decode(types.as_slice()?, coords.as_slice()?)?;
    let rendered = crate::transform::render_bitmap::render_bitmap(&outline, size, mode, fill_rule)
        .map_err(|_| {
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
    m.add_function(wrap_pyfunction!(load::load_glyph, m)?)?;
    m.add_function(wrap_pyfunction!(load::variation_axes, m)?)?;
    m.add_function(wrap_pyfunction!(quad_to_cubic, m)?)?;
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
