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
    let api_key = env::var("OPENAI_API_KEY").unwrap_or("".to_string());
    if api_key.is_empty() {
        return Err(anyhow!("OPENAI_API_KEY environment variable must be set"));
    }

    let client = reqwest::Client::new();
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
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

    for each_message in request_body.messages.iter() {
        let message = ChatCompletionRequestMessage {
            role: each_message.role.clone(),
            content: each_message
                .content
                .iter()
                .map(|content| match &content.ctx {
                    ContentTypeInner::Text(text) => Some(text.text.clone()),
                    _ => panic!("Invalid content type"),
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
    use crate::message::{Content, ContentTypeInner, Message, TextContent};

    // for testing OpenAI, please always construct a request with new()
    // this will ensure the prompt is built correctly with messages
    // in Anthropic, the prompt is built alongside the messages, so this is not necessary
    let request_body = OpenAIRequest::new(
        "gpt-4.1-nano-2025-04-14",
        1024,
        vec![Message {
            role: "user".to_string(),
            content: vec![Content {
                ctx: ContentTypeInner::Text(TextContent {
                    content_type: "text".to_string(),
                    text: "Hello, Claude!".to_string(),
                }),
            }],
        }],
        Some("Please answer in Chinese"),
    );

    let input_tokens = count_tokens_openai(request_body.clone()).unwrap();
    println!("Input tokens: {}", input_tokens);

    let response = request_openai(request_body).await;
    match response {
        Ok(res) => {
            println!("{}", res);
            println!("{:?}", res.cost_test().unwrap());
        }
        Err(e) => println!("Error: {}", e),
    }
}
