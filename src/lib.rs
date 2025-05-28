mod req_structs;
mod request;
mod res_structs;

use anyhow::{Result, anyhow};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use request::{get_response_anthropic, get_response_openai};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Clone, Debug)]
enum LLM {
    Anthropic,
    OpenAI,
}

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Clone, Debug)]
enum SupportedModels {
    GPT41Nano20250414,
    Claude35HaikuLatest,
}

impl Serialize for SupportedModels {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string())
    }
}

impl<'de> Deserialize<'de> for SupportedModels {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_string(&s).map_err(serde::de::Error::custom)
    }
}

impl SupportedModels {
    fn to_string(&self) -> &'static str {
        match self {
            SupportedModels::GPT41Nano20250414 => "gpt-4.1-nano-2025-04-14",
            SupportedModels::Claude35HaikuLatest => "claude-3-5-haiku-latest",
        }
    }

    fn from_string(model: &str) -> Result<SupportedModels> {
        match model {
            "gpt-4.1-nano-2025-04-14" => Ok(SupportedModels::GPT41Nano20250414),
            "claude-3-5-haiku-latest" => Ok(SupportedModels::Claude35HaikuLatest),
            "claude-3-5-haiku-20241022" => Ok(SupportedModels::Claude35HaikuLatest),
            _ => Err(anyhow!("Unsupported model: {}", model)),
        }
    }
}

/// Send prepared AnthropicRequest
#[pyfunction]
#[pyo3(signature = (request_body,llm=LLM::Anthropic))]
fn send<'p>(request_body: Bound<'p, PyAny>, llm: LLM) -> PyResult<res_structs::LLMResponse> {
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
    m.add_class::<LLM>()?;
    m.add_class::<res_structs::LLMResponse>()?;
    m.add_function(wrap_pyfunction!(send, m)?)?;
    Ok(())
}
