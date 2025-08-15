use pyo3::prelude::*;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct OpenAIReasoning {
    pub(crate) effort: Option<String>,
    pub(crate) summary: Option<String>,
}

impl Default for OpenAIReasoning {
    fn default() -> Self {
        Self {
            effort: Some("low".to_string()),
            summary: None,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
#[pyclass(dict, get_all, set_all, subclass)]
pub struct OpenAIRequest {
    pub(crate) model: SupportedModels,
    pub(crate) input: Vec<Message>,
    #[serde(skip)]
    pub(crate) system: Option<String>,
    #[serde(skip)]
    pub(crate) endpoint: Option<String>,
    pub(crate) reasoning: OpenAIReasoning,
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
            reasoning: OpenAIReasoning::default(),
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

    pub fn add_response_from_str(&mut self, response: String) -> PyResult<()> {
        self.input.push(Message {
            role: "assistant".to_string(),
            content: vec![Content {
                ctx: ContentTypeInner::Text(TextContent {
                    content_type: "output_text".to_string(),
                    text: response,
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

// Custom deserializer for OpenAI response that only extracts messages response and not reasoning
pub fn deserialize_message_only<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<ResponseChoiceOpenAI>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct MessageOnlyVisitor;

    impl<'de> Visitor<'de> for MessageOnlyVisitor {
        type Value = Option<Vec<ResponseChoiceOpenAI>>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence of response objects")
        }

        fn visit_seq<V>(self, mut seq: V) -> Result<Option<Vec<ResponseChoiceOpenAI>>, V::Error>
        where
            V: SeqAccess<'de>,
        {
            let mut responses = Vec::new();

            while let Some(value) = seq.next_element::<serde_json::Value>()? {
                if let Some(type_field) = value.get("type") {
                    if type_field == "message" {
                        let response: ResponseChoiceOpenAI =
                            serde_json::from_value(value).map_err(de::Error::custom)?;
                        responses.push(response);
                    }
                }
            }

            if responses.is_empty() {
                Ok(None)
            } else {
                Ok(Some(responses))
            }
        }
    }

    deserializer.deserialize_seq(MessageOnlyVisitor)
}
