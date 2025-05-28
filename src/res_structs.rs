use crate::SupportedModels;
use anyhow::{Result, anyhow};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, set_all)]
struct ResponseMsgOpenAI {
    role: String,
    content: String,
}

#[pymethods]
impl ResponseMsgOpenAI {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ResponseMsgOpenAI<role={:?},content={:?}>",
            self.role, self.content
        ))
    }
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, set_all)]
struct ResponseChoiceOpenAI {
    index: u32,
    message: ResponseMsgOpenAI,
    finish_reason: Option<String>,
}

#[pymethods]
impl ResponseChoiceOpenAI {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ResponseChoiceOpenAI<index={:?},message={:?},finish_reason={:?}>",
            self.index, self.message, self.finish_reason
        ))
    }
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, set_all)]
struct ResponseContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[pymethods]
impl ResponseContent {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("ResponseContent<text={:?}>", self.text))
    }
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, set_all)]
struct Usage {
    #[serde(alias = "prompt_tokens")]
    input_tokens: u32,
    #[serde(alias = "completion_tokens")]
    output_tokens: u32,
}

#[pymethods]
impl Usage {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Usage<input_tokens={:?},output_tokens={:?}>",
            self.input_tokens, self.output_tokens
        ))
    }
}

#[derive(Deserialize, Debug)]
#[pyclass(dict, get_all, frozen)]
pub(crate) struct LLMResponse {
    id: String,
    model: SupportedModels,
    #[serde(rename = "type", alias = "object")]
    response_type: String,
    usage: Usage,

    role: Option<String>,                  // Anthropic
    content: Option<Vec<ResponseContent>>, // Anthropic
    stop_reason: Option<String>,           // Anthropic

    choices: Option<Vec<ResponseChoiceOpenAI>>, // OpenAI
}

#[pymethods]
impl LLMResponse {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "AnthropicResponse<id={:?}, model={:?}>",
            self.id, self.model
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        let content: String = match self.role {
            Some(..) => {
                // Anthropic response
                let mut content = String::new();
                for c in self.content.as_ref().unwrap() {
                    content.push_str(&c.text);
                }
                content
            }
            None => {
                // OpenAI response
                let mut content = String::new();
                for choice in self.choices.as_ref().unwrap() {
                    content.push_str(&choice.message.content);
                }
                content
            }
        };
        Ok(format!("{}", content))
    }

    fn cost(&self) -> PyResult<f64> {
        let input: f64;
        let output: f64;
        if self.model.to_string() == "claude-3-5-haiku-latest" {
            input = 0.8;
            output = 4.0;
        } else if self.model.to_string() == "gpt-4.1-nano-2025-04-14" {
            input = 0.1;
            output = 0.4;
        } else {
            return Err(PyTypeError::new_err("Unsupported model"));
        }

        let input_tokens = self.usage.input_tokens as f64;
        let output_tokens = self.usage.output_tokens as f64;

        let input_cost = input * input_tokens;
        let output_cost = output * output_tokens;

        Ok((input_cost + output_cost) / 1_000_000.0)
    }
}

impl std::fmt::Display for LLMResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content: String = match self.role {
            Some(..) => {
                // Anthropic response
                let mut content = String::new();
                for c in self.content.as_ref().unwrap() {
                    content.push_str(&c.text);
                }
                content
            }
            None => {
                // OpenAI response
                let mut content = String::new();
                for choice in self.choices.as_ref().unwrap() {
                    content.push_str(&choice.message.content);
                }
                content
            }
        };
        write!(f, "{}", content)
    }
}

/// for test purposes
impl LLMResponse {
    #[allow(dead_code)]
    pub(crate) fn cost_test(&self) -> Result<f64> {
        let input: f64;
        let output: f64;
        if self.model.to_string() == "claude-3-5-haiku-latest" {
            input = 0.8;
            output = 4.0;
        } else if self.model.to_string() == "gpt-4.1-nano-2025-04-14" {
            input = 0.1;
            output = 0.4;
        } else {
            return Err(anyhow!("Unsupported model"));
        }

        let input_tokens = self.usage.input_tokens as f64;
        let output_tokens = self.usage.output_tokens as f64;

        let input_cost = input * input_tokens;
        let output_cost = output * output_tokens;

        Ok((input_cost + output_cost) / 1_000_000.0)
    }
}
