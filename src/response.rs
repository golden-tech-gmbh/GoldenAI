use pyo3::exceptions::PyTypeError;
use pyo3::{PyResult, pyclass, pymethods};
use serde::Deserialize;

use crate::SupportedModels;
use crate::anthropic::structs::ResponseAnthropic;
use crate::openai::structs::{
    OpenAIReasoning, OpenAIResError, ResponseChoiceOpenAI, deserialize_message_only,
};

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, set_all)]
pub struct Usage {
    #[serde(alias = "prompt_tokens")]
    pub input_tokens: u32,
    #[serde(alias = "completion_tokens")]
    pub output_tokens: u32,
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

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, frozen)]
pub struct LLMResponse {
    pub id: String,
    pub model: SupportedModels,
    #[serde(rename = "type", alias = "object")]
    pub response_type: String,
    pub usage: Usage,

    pub role: Option<String>,                    // Anthropic
    pub content: Option<Vec<ResponseAnthropic>>, // Anthropic
    pub stop_reason: Option<String>,             // Anthropic

    #[serde(default, deserialize_with = "deserialize_message_only")]
    pub output: Option<Vec<ResponseChoiceOpenAI>>, // OpenAI
    pub reasoning: Option<OpenAIReasoning>, // OpenAI
    pub instructions: Option<String>,       // OpenAI
    pub error: Option<OpenAIResError>,      // OpenAI
    pub status: Option<String>,             // OpenAI
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
                for choice in self.output.as_ref().unwrap() {
                    content.push_str(&choice.content[0].text); // TODO! Risky
                }
                content
            }
        };
        Ok(format!("{}", content))
    }

    pub fn cost(&self) -> PyResult<f64> {
        let input: f64;
        let output: f64;
        if self.model.to_str() == "claude-3-5-haiku-latest" {
            input = 0.8;
            output = 4.0;
        } else if self.model.to_str() == "gpt-4.1-nano" {
            input = 0.1;
            output = 0.4;
        } else if self.model.to_str() == "gpt-4.1" {
            input = 1.7;
            output = 6.84;
        } else if self.model.to_str() == "gpt-5" {
            input = 1.25;
            output = 10.0;
        } else if self.model.to_str() == "gpt-5-nano" {
            input = 0.05;
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
                // Anthropic or Ollama response
                let mut content = String::new();
                for c in self.content.as_ref().unwrap() {
                    content.push_str(&c.text);
                }
                content
            }
            None => {
                // OpenAI response
                let mut content = String::new();
                for choice in self.output.as_ref().unwrap() {
                    content.push_str(&choice.content[0].text); // TODO! Risky
                }
                content
            }
        };
        write!(f, "{}", content)
    }
}
