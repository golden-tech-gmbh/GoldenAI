use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::SupportedModels;
use crate::message::{Content, ContentTypeInner, Message, TextContent};
use crate::response::LLMResponse;

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct OpenAIRequest {
    pub(crate) model: SupportedModels,
    #[serde(skip)]
    pub(crate) system: Option<String>,
    pub(crate) max_tokens: u32,
    pub(crate) messages: Vec<Message>,
}

#[pymethods]
impl OpenAIRequest {
    #[new]
    #[pyo3(signature = (model, max_tokens,messages,prompt=None))]
    pub fn new(model: &str, max_tokens: u32, messages: Vec<Message>, prompt: Option<&str>) -> Self {
        Self {
            model: SupportedModels::from_str(model).unwrap(),
            max_tokens,
            messages: match prompt {
                Some(p) => {
                    let mut new_messages = Vec::new();
                    new_messages.push(Message {
                        role: "developer".to_string(),
                        content: vec![Content {
                            ctx: ContentTypeInner::Text(TextContent {
                                content_type: "text".to_string(),
                                text: p.to_string(),
                            }),
                        }],
                    });
                    new_messages.extend(messages);
                    new_messages
                }
                None => messages,
            },
            system: None,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }

    pub fn add_response(&mut self, response: LLMResponse) -> PyResult<()> {
        let resp = match response.choices.is_some() {
            true => response.choices.unwrap(),
            false => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "No choices in response",
                ));
            }
        };

        if resp.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "More than one choice in response",
            ));
        }

        self.messages.push(Message {
            role: resp[0].message.role.clone(),
            content: vec![Content {
                ctx: ContentTypeInner::Text(TextContent {
                    content_type: "text".to_string(),
                    text: resp[0].message.content.clone(),
                }),
            }],
        });

        Ok(())
    }

    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, set_all)]
pub struct ResponseMsgOpenAI {
    pub role: String,
    pub content: String,
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
pub struct ResponseChoiceOpenAI {
    pub index: u32,
    pub message: ResponseMsgOpenAI,
    pub finish_reason: Option<String>,
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
