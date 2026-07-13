use std::{collections::BTreeMap, path::Path};

use pyo3::prelude::*;
use pyo3::types::PyModule;

pub(crate) struct InstanceFunctionsBridge<'py> {
    font_ref_class: Bound<'py, PyAny>,
    py_float: Bound<'py, PyAny>,
    py_fspath: Bound<'py, PyAny>,
    py_index: Bound<'py, PyAny>,
}

impl<'py> InstanceFunctionsBridge<'py> {
    pub(crate) fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            font_ref_class: PyModule::import(py, "torchfont.datasets")?.getattr("FontRef")?,
            py_float: PyModule::import(py, "builtins")?.getattr("float")?,
            py_fspath: PyModule::import(py, "os")?.getattr("fspath")?,
            py_index: PyModule::import(py, "operator")?.getattr("index")?,
        })
    }

    pub(crate) fn call_instance_locations(
        &self,
        callable: &Bound<'py, PyAny>,
        path: &Path,
        ttc_index: u32,
    ) -> PyResult<Vec<BTreeMap<String, f32>>> {
        let font_ref = self.font_ref(path, ttc_index)?;
        let raw_locations = callable.call1((font_ref,))?;
        let mut locations = Vec::new();
        for raw_location in raw_locations.try_iter()? {
            locations.push(self.location_mapping(raw_location?.as_borrowed())?);
        }
        Ok(locations)
    }

    pub(crate) fn call_instance_count(
        &self,
        callable: &Bound<'py, PyAny>,
        path: &Path,
        ttc_index: u32,
    ) -> PyResult<usize> {
        let font_ref = self.font_ref(path, ttc_index)?;
        let raw_count = callable.call1((font_ref,))?;
        let count = self.py_index.call1((raw_count,))?.extract::<isize>()?;
        if count < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "InstanceCountFn must return a non-negative integer",
            ));
        }
        Ok(count as usize)
    }

    fn font_ref(&self, path: &Path, ttc_index: u32) -> PyResult<Py<PyAny>> {
        let path = self.py_fspath.call1((path,))?;
        Ok(self.font_ref_class.call1((path, ttc_index))?.unbind())
    }

    fn location_mapping(
        &self,
        location: Borrowed<'_, 'py, PyAny>,
    ) -> PyResult<BTreeMap<String, f32>> {
        let mut out = BTreeMap::new();
        for item in location.call_method0("items")?.try_iter()? {
            let item = item?;
            let tag = item.get_item(0)?.str()?.to_string_lossy().into_owned();
            let value = self
                .py_float
                .call1((item.get_item(1)?,))?
                .extract::<f32>()?;
            out.insert(tag, value);
        }
        Ok(out)
    }
}
