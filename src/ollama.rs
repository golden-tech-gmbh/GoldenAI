use crate::SupportedModels;
use crate::req_structs::{ContentTypeInner, Message};
use crate::res_structs::{LLMResponse, ResponseContent, Usage};
use anyhow::{Result, anyhow};
use pyo3::prelude::*;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};

#[tokio::main]
pub async fn get_response_ollama(request_body: OllamaRequest) -> Result<LLMResponse> {
    request_ollama(request_body).await
}

pub async fn request_ollama(request_body: OllamaRequest) -> Result<LLMResponse> {
    #[derive(Serialize)]
    struct ConvertedOllamaRequest {
        model: String,
        prompt: String,
        stream: bool,
    }
    let converted_request = ConvertedOllamaRequest {
        model: request_body.model.to_str().to_string(),
        prompt: match request_body.system {
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
    };

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/api/generate", request_body.url))
        .json(&converted_request)
        .send()
        .await?;

    if response.status().is_success() {
        #[derive(Deserialize)]
        struct OllamaResponse {
            model: SupportedModels,
            response: String,
            done_reason: String,
            eval_count: u32,
            prompt_eval_count: u32,
        }

        // let response_text = response.text().await?;
        // println!("Raw response: {}", response_text);
        // let response: OllamaResponse = serde_json::from_str(&response_text)?;
        let response: OllamaResponse = response.json().await?;

        let response = LLMResponse {
            id: "".to_string(),
            model: response.model,
            response_type: "text".to_string(),
            usage: Usage {
                input_tokens: response.eval_count,
                output_tokens: response.prompt_eval_count,
            },
            role: Some("ollama".to_string()), // to make sure the ResponseContent is used in fmt,
            content: Some(vec![ResponseContent {
                content_type: "text".to_string(),
                text: response.response,
            }]),
            stop_reason: Some(response.done_reason),
            choices: None,
        };
        Ok(response)
    } else {
        let err_status = response.status();
        let error_text = response.text().await?;
        Err(anyhow!(
            "Error: HTTP {}, Response: {}",
            err_status,
            error_text
        ))
    }
}

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
    pub fn new(
        url: &str,
        model: &str,
        messages: Vec<Message>,
        prompt: Option<&str>,
    ) -> PyResult<Self> {
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

#[tokio::test]
async fn test_ollama_request() {
    use std::env;
    let url = env::var("OLLAMA_URL").unwrap_or("".to_string());
    if url.is_empty() {
        panic!("OLLAMA_URL environment variable must be set");
    }

    let request_body = OllamaRequest {
        url,
        model: SupportedModels::Qwen25VL,
        system: None,
        messages: vec![Message {
            role: "user".to_string(),
            content: vec![crate::req_structs::Content {
                ctx: ContentTypeInner::Text(crate::req_structs::TextContent {
                    content_type: "text".to_string(),
                    text: "Hello, Claude!".to_string(),
                }),
            }],
        }],
    };
    let res = request_ollama(request_body).await.unwrap();
    println!("{:?}", res);
}
