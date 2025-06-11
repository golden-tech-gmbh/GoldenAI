use base64::Engine;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::SupportedModels;
use crate::anthropic::structs::ResponseAnthropic;
use crate::message::{Content, ContentTypeInner, Message, TextContent};
use crate::response::{LLMResponse, Usage};

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct OllamaRequest {
    pub(crate) url: String,
    pub(crate) model: SupportedModels,
    pub(crate) system: Option<String>,
    pub(crate) messages: Vec<Message>,
    pub(crate) image: Option<String>, // base64 encoded images
}

#[pymethods]
impl OllamaRequest {
    #[new]
    #[pyo3(signature = (url, model, messages, prompt=None, image=None))]
    pub fn new(
        url: &str,
        model: &str,
        messages: Vec<Message>,
        prompt: Option<&str>,
        image: Option<&str>, // image path
    ) -> PyResult<Self> {
        let data = match image {
            Some(image) => {
                let path = PathBuf::from(image);
                if !path.exists() {
                    return Err(PyException::new_err("Image file does not exist"));
                }
                let data = std::fs::read(&path)
                    .map_err(|e| PyException::new_err(format!("Failed to read file: {}", e)))?;
                Some(base64::engine::general_purpose::STANDARD.encode(&data))
            }
            None => None,
        };

        Ok(Self {
            url: url.to_string(),
            model: SupportedModels::from_str(model).unwrap(),
            system: prompt.map(|s| s.to_string()),
            messages,
            image: data,
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }

    pub fn add_response(&mut self, response: LLMResponse) {
        self.messages.push(Message {
            role: "assistant".to_string(),
            content: vec![Content {
                ctx: ContentTypeInner::Text(TextContent {
                    content_type: "text".to_string(),
                    text: response.content.unwrap()[0].text.clone(),
                }),
            }],
        });
    }

    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }
}

#[derive(Serialize)]
pub struct ConvertedOllamaRequest {
    pub url: String,
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    pub images: Option<Vec<String>>, // base64 encoded images
}

impl ConvertedOllamaRequest {
    pub fn from_ollama_request(request_body: OllamaRequest, stream: bool) -> Self {
        Self {
            url: request_body.url,
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
            stream,
            images: match request_body.image {
                Some(image) => Some(vec![image]),
                None => None,
            },
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

#[derive(Deserialize)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
    pub images: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct OllamaChatResponse {
    pub model: SupportedModels,
    pub created_at: String,
    pub message: OllamaChatMessage,
    pub done: bool,
    pub done_reason: String,
}

impl OllamaChatResponse {
    pub fn to_llm_response(self) -> LLMResponse {
        LLMResponse {
            id: "".to_string(),
            model: self.model,
            response_type: "text".to_string(),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
            role: Some(self.message.role),
            content: Some(vec![ResponseAnthropic {
                content_type: "text".to_string(),
                text: self.message.content,
            }]),
            stop_reason: Some(self.done_reason),
            choices: None,
        }
    }
}
