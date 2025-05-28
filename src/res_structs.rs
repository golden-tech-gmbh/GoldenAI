use crate::SupportedModels;
use anyhow::{Result, anyhow};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass;
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;

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

#[derive(Deserialize, Debug)]
#[pyclass(dict, get_all, frozen)]
pub(crate) struct LLMResponse {
    id: String,
    model: SupportedModels,
    #[serde(rename = "type", alias = "object")]
    response_type: String,

    role: Option<String>,                  // Anthropic
    content: Option<Vec<ResponseContent>>, // Anthropic
    stop_reason: Option<String>,           // Anthropic

    choices: Option<Vec<ResponseChoiceOpenAI>>, // OpenAI

    #[serde(deserialize_with = "deserialize_filtered_map")]
    usage: HashMap<String, u32>,
}

fn deserialize_filtered_map<'de, D>(deserializer: D) -> Result<HashMap<String, u32>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde_json::Value;

    let value: HashMap<String, Value> = HashMap::deserialize(deserializer)?;
    let mut result = HashMap::new();

    for (key, val) in value {
        match val.as_u64() {
            Some(n) => {
                let num = n.try_into().ok();
                if let Some(num) = num {
                    result.insert(key, num);
                }
            }
            None => {}
        }
    }

    Ok(result)
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
