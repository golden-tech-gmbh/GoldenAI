use pyo3::prelude::*;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};

use crate::SupportedModels;
use crate::anthropic::structs::ResponseAnthropic;
use crate::message::{ContentTypeInner, Message};
use crate::response::{LLMResponse, Usage};

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct OllamaRequest {
    pub(crate) url: String,
    pub(crate) model: SupportedModels,
    pub(crate) system: Option<String>,
    pub(crate) messages: Vec<Message>,
}

#[pymethods]
impl OllamaRequest {
    #[new]
    #[pyo3(signature = (url, model, messages,prompt=None))]
    fn new(url: &str, model: &str, messages: Vec<Message>, prompt: Option<&str>) -> PyResult<Self> {
        Ok(Self {
            url: url.to_string(),
            model: SupportedModels::from_str(model).unwrap(),
            system: prompt.map(|s| s.to_string()),
            messages,
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}

#[derive(Serialize)]
pub struct ConvertedOllamaRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
}

impl ConvertedOllamaRequest {
    pub fn from_ollama_request(request_body: &OllamaRequest) -> Self {
        Self {
            model: request_body.model.to_str().to_string(),
            prompt: match &request_body.system {
                Some(system) => {
                    format!(
                        "{}\n\nThe following is my message:\n\n{}",
                        system,
                        request_body.messages[0]
                            .content
                            .iter()
                            .map(|each_content| match &each_content.ctx {
                                ContentTypeInner::Text(text) => text.text.clone(),
                                _ => panic!("Invalid content type"),
                            })
                            .collect::<Vec<String>>()
                            .join("\n\n")
                    )
                }
                None => {
                    format!(
                        "{}",
                        request_body.messages[0]
                            .content
                            .iter()
                            .map(|each_content| match &each_content.ctx {
                                ContentTypeInner::Text(text) => text.text.clone(),
                                _ => panic!("Invalid content type"),
                            })
                            .collect::<Vec<String>>()
                            .join("\n\n")
                    )
                }
            },
            stream: false, // TODO! for now is disabled
        }
    }
}

#[derive(Deserialize)]
pub struct OllamaResponse {
    pub model: SupportedModels,
    pub response: String,
    pub done_reason: String,
    pub eval_count: u32,
    pub prompt_eval_count: u32,
}

impl OllamaResponse {
    pub fn to_llm_response(self) -> LLMResponse {
        LLMResponse {
            id: "".to_string(),
            model: self.model,
            response_type: "text".to_string(),
            usage: Usage {
                input_tokens: self.eval_count,
                output_tokens: self.prompt_eval_count,
            },
            role: Some("ollama".to_string()), // to make sure the ResponseContent is used in fmt,
            content: Some(vec![ResponseAnthropic {
                content_type: "text".to_string(),
                text: self.response,
            }]),
            stop_reason: Some(self.done_reason),
            choices: None,
        }
    }
}
