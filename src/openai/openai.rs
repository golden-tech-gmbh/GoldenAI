use anyhow::{Result, anyhow};
use reqwest;
use std::env;
use tiktoken_rs::{ChatCompletionRequestMessage, num_tokens_from_messages};

use crate::message::ContentTypeInner;
use crate::openai::structs::OpenAIRequest;
use crate::response::LLMResponse;

#[tokio::main]
pub async fn get_response_openai(request_body: OpenAIRequest) -> Result<LLMResponse> {
    request_openai(request_body).await
}

pub fn get_count_tokens_openai(request_body: OpenAIRequest) -> Result<u32> {
    count_tokens_openai(request_body)
}

async fn request_openai(request_body: OpenAIRequest) -> Result<LLMResponse> {
    let endpoint = match &request_body.endpoint {
        Some(endpoint) => endpoint.clone(),
        None => "https://api.openai.com/v1/responses".to_string(),
    };

    let api_key = {
        if endpoint.contains("azure") {
            env::var("AZURE_OPENAI_API_KEY").unwrap_or("".to_string())
        } else {
            env::var("OPENAI_API_KEY").unwrap_or("".to_string())
        }
    };

    if api_key.is_empty() {
        return Err(anyhow!("OpenAI API key must be set"));
    }

    let client = reqwest::Client::new();
    let response = client
        .post(endpoint)
        .header("content-type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request_body)
        .send()
        .await?;

    if response.status().is_success() {
        // let response_text = response.text().await?;
        // println!("Raw response: {}", response_text);
        // let response: LLMResponse = serde_json::from_str(&response_text)?;
        let response: LLMResponse = response.json().await?;
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

fn count_tokens_openai(request_body: OpenAIRequest) -> Result<u32> {
    let api_key = env::var("OPENAI_API_KEY").unwrap_or("".to_string());
    if api_key.is_empty() {
        return Err(anyhow!("OPENAI_API_KEY environment variable must be set"));
    }

    let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();

    for each_message in request_body.input.iter() {
        let message = ChatCompletionRequestMessage {
            role: each_message.role.clone(),
            content: each_message
                .content
                .iter()
                .map(|content| match &content.ctx {
                    ContentTypeInner::Text(text) => Some(text.text.clone()),
                    ContentTypeInner::Document(document) => {
                        Some(document.clone().file_data.unwrap())
                    }
                })
                .collect(),
            name: None,
            function_call: None,
        };
        messages.push(message);
    }

    let max_tokens = num_tokens_from_messages(request_body.model.to_str(), &messages)?;

    Ok(max_tokens as u32)
}

#[tokio::test]
async fn test_request_openai() {
    use crate::SupportedModels;
    use crate::message::{Content, ContentTypeInner, DocumentContent, Message, TextContent};

    // for testing OpenAI, please always construct a request with new()
    // this will ensure the prompt is built correctly with messages
    // in Anthropic, the prompt is built alongside the messages, so this is not necessary
    let mut request_body = OpenAIRequest::new(
        "gpt-4.1-nano-2025-04-14",
        vec![Message {
            role: "user".to_string(),
            content: vec![
                Content {
                    ctx: ContentTypeInner::Text(TextContent {
                        content_type: "input_text".to_string(),
                        text: "What does the document say?".to_string(),
                    }),
                },
                Content {
                    ctx: ContentTypeInner::Document(
                        DocumentContent::new(
                            "examples/python/test.pdf",
                            Some(SupportedModels::GPT41Nano),
                        )
                        .unwrap(),
                    ),
                },
            ],
        }],
        Some("Please answer in Chinese"),
        Some(
            "https://goldenai.openai.azure.com/openai/deployments/gpt-4.1-nano/chat/completions?api-version=2025-01-01-preview",
        ),
    );

    println!(
        "Input tokens: {}",
        count_tokens_openai(request_body.clone()).unwrap()
    );

    let response = request_openai(request_body.clone()).await;
    match response {
        Ok(res) => {
            println!("{}", res);
            println!("{:?}", res.cost().unwrap());
            request_body.add_response(res).expect("TODO: panic message");
            request_body.add_message(Message {
                role: "user".to_string(),
                content: vec![Content {
                    ctx: ContentTypeInner::Text(TextContent {
                        content_type: "input_text".to_string(),
                        text: "Please answer the same question but in English again".to_string(),
                    }),
                }],
            });
            println!(
                "Input tokens 2: {}",
                count_tokens_openai(request_body.clone()).unwrap()
            );
            let new_response = request_openai(request_body).await;
            match new_response {
                Ok(new_res) => {
                    println!("{}", new_res);
                    println!("{:?}", new_res.cost().unwrap());
                }
                Err(e) => println!("Error: {}", e),
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
