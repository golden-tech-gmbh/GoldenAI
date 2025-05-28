mod req_structs;
mod request;
mod res_structs;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use request::{get_response_anthropic, get_response_openai};

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Clone, Debug)]
enum LLM {
    Anthropic,
    OpenAI,
}

/// Send prepared AnthropicRequest
#[pyfunction]
fn send<'p>(
    llm: Bound<'p, PyAny>,
    request_body: Bound<'p, PyAny>,
) -> PyResult<res_structs::LLMResponse> {
    let llm = llm.extract::<LLM>()?;

    match llm {
        LLM::Anthropic => {
            let request_body = request_body.extract::<req_structs::AnthropicRequest>()?;
            match get_response_anthropic(request_body) {
                Ok(response) => Ok(response),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        }
        LLM::OpenAI => {
            let request_body = request_body.extract::<req_structs::OpenAIRequest>()?;
            match get_response_openai(request_body) {
                Ok(response) => Ok(response),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        }
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
    m.add_class::<req_structs::OpenAIRequest>()?;
    m.add_function(wrap_pyfunction!(send, m)?)?;
    Ok(())
}
