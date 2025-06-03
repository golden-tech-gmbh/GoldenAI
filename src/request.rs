use crate::SupportedModels;
use crate::req_structs::{AnthropicRequest, ContentTypeInner, Message, OpenAIRequest};
use crate::res_structs::LLMResponse;
use anyhow::{Result, anyhow};
use reqwest;
use serde::{Deserialize, Serialize};
use std::env;
use tiktoken_rs::{ChatCompletionRequestMessage, num_tokens_from_messages};

#[tokio::main]
pub async fn get_response_anthropic(request_body: AnthropicRequest) -> Result<LLMResponse> {
    request_anthropic(request_body).await
}

#[tokio::main]
pub async fn get_response_openai(request_body: OpenAIRequest) -> Result<LLMResponse> {
    request_openai(request_body).await
}

#[tokio::main]
pub async fn get_count_tokens_anthropic(request_body: AnthropicRequest) -> Result<u32> {
    count_tokens_anthropic(request_body).await
}

pub fn get_count_tokens_openai(request_body: OpenAIRequest) -> Result<u32> {
    count_tokens_openai(request_body)
}

async fn request_anthropic(request_body: AnthropicRequest) -> Result<LLMResponse> {
    let api_key = env::var("ANTHROPIC_API_KEY").unwrap_or("".to_string());
    if api_key.is_empty() {
        return Err(anyhow!(
            "ANTHROPIC_API_KEY environment variable must be set"
        ));
    }

    let client = reqwest::Client::new();
    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
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

async fn count_tokens_anthropic(request_body: AnthropicRequest) -> Result<u32> {
    let api_key = env::var("ANTHROPIC_API_KEY").unwrap_or("".to_string());
    if api_key.is_empty() {
        return Err(anyhow!(
            "ANTHROPIC_API_KEY environment variable must be set"
        ));
    }

    #[derive(Serialize, Debug)]
    struct CountTokensRequest {
        model: SupportedModels,
        messages: Vec<Message>,
    }

    let client = reqwest::Client::new();
    let response = client
        .post("https://api.anthropic.com/v1/messages/count_tokens")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&CountTokensRequest {
            model: request_body.model,
            messages: request_body.messages,
        })
        .send()
        .await?;

    if response.status().is_success() {
        #[derive(Deserialize, Debug)]
        struct CountTokensResponse {
            input_tokens: u32,
        }
        // let response_text = response.text().await?;
        // println!("Raw response: {}", response_text);
        // let response: LLMResponse = serde_json::from_str(&response_text)?;
        let response: CountTokensResponse = response.json().await?;
        Ok(response.input_tokens)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SupportedModels;

    #[tokio::test]
    async fn test_request_anthropic() {
        let request_body = AnthropicRequest {
            model: SupportedModels::from_str("claude-3-5-haiku-latest").unwrap(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: vec![crate::req_structs::Content {
                    ctx: ContentTypeInner::Text(crate::req_structs::TextContent {
                        content_type: "text".to_string(),
                        text: "Hello, Claude!".to_string(),
                    }),
                }],
            }],
            system: Some("Please answer in Chinese".to_string()),
        };
        let input_tokens = count_tokens_anthropic(request_body.clone()).await.unwrap();
        println!("Input tokens: {}", input_tokens);

        let response = request_anthropic(request_body).await;
        match response {
            Ok(res) => {
                println!("{}", res);
                println!("{:?}", res.cost_test().unwrap());
            }
            Err(e) => println!("Error: {}", e),
        }
    }

    #[tokio::test]
    async fn test_request_openai() {
        // for testing OpenAI, please always construct a request with new()
        // this will ensure the prompt is built correctly with messages
        // in Anthropic, the prompt is built alongside the messages, so this is not necessary
        let request_body = OpenAIRequest::new(
            "gpt-4.1-nano-2025-04-14",
            1024,
            vec![Message {
                role: "user".to_string(),
                content: vec![crate::req_structs::Content {
                    ctx: ContentTypeInner::Text(crate::req_structs::TextContent {
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
}
