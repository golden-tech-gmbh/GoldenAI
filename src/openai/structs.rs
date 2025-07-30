use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::SupportedModels;
use crate::message::{Content, ContentTypeInner, Message, TextContent};
use crate::response::LLMResponse;

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct OpenAIRequest {
    pub(crate) model: SupportedModels,
    pub(crate) input: Vec<Message>,
    #[serde(skip)]
    pub(crate) system: Option<String>,
    #[serde(skip)]
    pub(crate) endpoint: Option<String>,
}

#[pymethods]
impl OpenAIRequest {
    #[new]
    #[pyo3(signature = (model,messages,prompt=None,endpoint=None))]
    pub fn new(
        model: &str,
        messages: Vec<Message>,
        prompt: Option<&str>,
        endpoint: Option<&str>,
    ) -> Self {
        let modified_messages = messages
            .into_iter()
            .map(|mut msg| {
                for content in &mut msg.content {
                    if let ContentTypeInner::Text(ref mut text_content) = content.ctx {
                        text_content.content_type = "input_text".to_string();
                    }
                }
                msg
            })
            .collect::<Vec<Message>>();

        Self {
            model: SupportedModels::from_str(model).unwrap(),
            input: match prompt {
                Some(p) => {
                    let mut new_messages = Vec::new();
                    new_messages.push(Message {
                        role: "developer".to_string(),
                        content: vec![Content {
                            ctx: ContentTypeInner::Text(TextContent {
                                content_type: "input_text".to_string(),
                                text: p.to_string(),
                            }),
                        }],
                    });
                    new_messages.extend(modified_messages);
                    new_messages
                }
                None => modified_messages,
            },
            system: None,
            endpoint: endpoint.map(|s| s.to_string()),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }

    pub fn add_response(&mut self, response: LLMResponse) -> PyResult<()> {
        let resp = match response.output.is_some() {
            true => response.output.unwrap(),
            false => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "No choices in response",
                ));
            }
        };

        if resp.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "More than one output in response",
            ));
        }

        self.input.push(Message {
            role: resp[0].role.clone(), // TODO! Risky
            content: vec![Content {
                ctx: ContentTypeInner::Text(TextContent {
                    content_type: "output_text".to_string(),
                    text: resp[0].content[0].text.clone(), // TODO! Risky
                }),
            }],
        });

        Ok(())
    }

    pub fn add_message(&mut self, mut message: Message) {
        for content in &mut message.content {
            if let ContentTypeInner::Text(ref mut text_content) = content.ctx {
                text_content.content_type = "input_text".to_string();
            }
        }
        self.input.push(message);
    }

    #[getter]
    fn model(&self) -> PyResult<String> {
        Ok(self.model.to_str().to_string())
    }
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, set_all)]
pub struct ResponseMsgOpenAI {
    #[serde(rename = "type")]
    pub response_type: String,
    pub text: String,
}

#[pymethods]
impl ResponseMsgOpenAI {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("ResponseMsgOpenAI<content={:?}>", self.text))
    }
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass(dict, get_all, set_all)]
pub struct ResponseChoiceOpenAI {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub status: String,
    pub role: String,
    pub content: Vec<ResponseMsgOpenAI>,
}

#[pymethods]
impl ResponseChoiceOpenAI {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ResponseChoiceOpenAI<id={:?},message={:?},status={:?}>",
            self.id, self.content, self.status
        ))
    }
}
