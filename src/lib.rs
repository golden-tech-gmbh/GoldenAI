mod req_structs;
mod request;
mod res_structs;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use request::get_response;

/// Send prepared AnthropicRequest
#[pyfunction]
fn send(request_body: req_structs::AnthropicRequest) -> PyResult<res_structs::AnthropicResponse> {
    match get_response(request_body) {
        Ok(response) => Ok(response),
        Err(e) => Err(PyException::new_err(e.to_string())),
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn goldenai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<req_structs::AnthropicRequest>()?;
    m.add_class::<req_structs::Message>()?;
    m.add_class::<req_structs::TextContent>()?;
    m.add_class::<req_structs::DocumentContent>()?;
    m.add_class::<req_structs::DocumentSourceContent>()?;
    m.add_class::<req_structs::Content>()?;
    m.add_function(wrap_pyfunction!(send, m)?)?;
    Ok(())
}
