pub mod message;
pub mod response;

pub mod openai {
    pub mod openai;
    pub mod structs;
}

pub mod anthropic {
    pub mod anthropic;
    pub mod structs;
}

pub mod ollama {
    pub mod ollama;
    pub mod structs;
}

use anyhow::{Result, anyhow};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Clone, Debug)]
pub enum SupportedModels {
    GPT41Nano,
    GPT41,
    Claude35HaikuLatest,
    Qwen25VL,
}

impl Serialize for SupportedModels {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

impl<'de> Deserialize<'de> for SupportedModels {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_str(&s).map_err(serde::de::Error::custom)
    }
}

impl SupportedModels {
    fn to_str(&self) -> &'static str {
        match self {
            SupportedModels::GPT41Nano => "gpt-4.1-nano",
            SupportedModels::GPT41 => "gpt-4.1",
            SupportedModels::Claude35HaikuLatest => "claude-3-5-haiku-latest",
            SupportedModels::Qwen25VL => "qwen2.5vl:latest",
        }
    }

    fn from_str(model: &str) -> Result<SupportedModels> {
        match model {
            "gpt-4.1-nano-2025-04-14" => Ok(SupportedModels::GPT41Nano),
            "gpt-4.1-nano" => Ok(SupportedModels::GPT41Nano),
            "gpt-4.1-2025-04-14" => Ok(SupportedModels::GPT41),
            "gpt-4.1" => Ok(SupportedModels::GPT41),
            "claude-3-5-haiku-latest" => Ok(SupportedModels::Claude35HaikuLatest),
            "claude-3-5-haiku-20241022" => Ok(SupportedModels::Claude35HaikuLatest),
            "qwen2.5vl:latest" => Ok(SupportedModels::Qwen25VL),
            "qwen2.5vl:7b" => Ok(SupportedModels::Qwen25VL),
            "qwen2.5vl" => Ok(SupportedModels::Qwen25VL),
            _ => Err(anyhow!("Unsupported model: {}", model)),
        }
    }
}

#[pyfunction]
fn send<'p>(request_body: Bound<'p, PyAny>) -> PyResult<response::LLMResponse> {
    if let Ok(anthropic_req) = request_body.extract::<anthropic::structs::AnthropicRequest>() {
        match anthropic::anthropic::get_response_anthropic(anthropic_req) {
            Ok(response) => Ok(response),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    } else if let Ok(openai_req) = request_body.extract::<openai::structs::OpenAIRequest>() {
        match openai::openai::get_response_openai(openai_req) {
            Ok(response) => Ok(response),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    } else if let Ok(ollama_req) = request_body.extract::<ollama::structs::OllamaRequest>() {
        match ollama::ollama::get_response_ollama(ollama_req, false) {
            // NOTE! in send mode, chat mode is disabled
            Ok(response) => Ok(response),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    } else {
        Err(PyException::new_err("Invalid request body"))
    }
}

#[pyfunction]
fn chat(request_body: Bound<PyAny>) -> PyResult<response::LLMResponse> {
    if let Ok(ollama_req) = request_body.extract::<ollama::structs::OllamaRequest>() {
        match ollama::ollama::get_response_ollama(ollama_req, true) {
            // NOTE! in send mode, chat mode is disabled
            Ok(response) => Ok(response),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    } else {
        Err(PyException::new_err(
            "Only Ollama is supported with chat mode",
        ))
    }
}

#[pyfunction]
fn count_tokens<'p>(request_body: Bound<'p, PyAny>) -> PyResult<u32> {
    if let Ok(anthropic_req) = request_body.extract::<anthropic::structs::AnthropicRequest>() {
        // TODO! still using match instead of map,
        // TODO! to avoid RustRover IDE error hint due to Async type missmatch
        // TODO! but chain with map_err() is working without async (it actually have to do without it)
        match anthropic::anthropic::get_count_tokens_anthropic(anthropic_req) {
            Ok(tokens) => Ok(tokens),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    } else if let Ok(openai_req) = request_body.extract::<openai::structs::OpenAIRequest>() {
        openai::openai::get_count_tokens_openai(openai_req)
            .map_err(|e| PyException::new_err(e.to_string()))
    } else if let Ok(_ollama_req) = request_body.extract::<ollama::structs::OllamaRequest>() {
        Ok(0u32) // TODO! Ollama is not necessary to count tokens for now
    } else {
        Err(PyException::new_err("Invalid request body"))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn goldenai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<anthropic::structs::AnthropicRequest>()?;
    m.add_class::<openai::structs::OpenAIRequest>()?;
    m.add_class::<ollama::structs::OllamaRequest>()?;

    m.add_class::<message::Message>()?;
    m.add_class::<message::TextContent>()?;
    m.add_class::<message::DocumentContent>()?;
    m.add_class::<message::DocumentSourceContent>()?;
    m.add_class::<message::Content>()?;

    m.add_class::<response::LLMResponse>()?;

    m.add_function(wrap_pyfunction!(send, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(chat, m)?)?;
    Ok(())
}
