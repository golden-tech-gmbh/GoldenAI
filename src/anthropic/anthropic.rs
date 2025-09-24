use anyhow::{Result, anyhow};
use reqwest;
use serde::{Deserialize, Serialize};
use std::env;

use crate::ANTHROPIC_API_URL;
use crate::SupportedModels;
use crate::anthropic::structs::AnthropicRequest;
use crate::message::Message;
use crate::response::LLMResponse;

#[tokio::main]
pub async fn get_response_anthropic(request_body: AnthropicRequest) -> Result<LLMResponse> {
    request_anthropic(request_body).await
}

#[tokio::main]
pub async fn get_count_tokens_anthropic(request_body: AnthropicRequest) -> Result<u32> {
    count_tokens_anthropic(request_body).await
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
        .post(ANTHROPIC_API_URL)
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
        .post(format!("{}/count_tokens", ANTHROPIC_API_URL))
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

#[tokio::test]
async fn test_request_anthropic() {
    use crate::message::{Content, ContentTypeInner, DocumentContent, TextContent};

    let mut request_body = AnthropicRequest {
        model: SupportedModels::from_str("claude-3-5-haiku-latest").unwrap(),
        max_tokens: 1024,
        messages: vec![Message {
            role: "user".to_string(),
            content: vec![
                Content {
                    ctx: ContentTypeInner::Document(
                        DocumentContent::new("test.pdf", None).unwrap(),
                    ),
                },
                Content {
                    ctx: ContentTypeInner::Text(TextContent {
                        content_type: "text".to_string(),
                        text: "What is the recipient address from this invoice?".to_string(),
                    }),
                },
            ],
        }],
        system: Some("Please answer in Chinese".to_string()),
    };

    println!(
        "Input tokens: {}",
        count_tokens_anthropic(request_body.clone()).await.unwrap()
    );

    let response = request_anthropic(request_body.clone()).await;
    match response {
        Ok(res) => {
            println!("{}", res);
            println!("{:?}", res.cost().unwrap());
            request_body.add_response(res.clone());
            request_body.add_message(Message {
                role: "user".to_string(),
                content: vec![Content {
                    ctx: ContentTypeInner::Text(TextContent {
                        content_type: "text".to_string(),
                        text: "I want you to answer the same question again but in English"
                            .to_string(),
                    }),
                }],
            });
            println!(
                "Input tokens 2: {}",
                count_tokens_anthropic(request_body.clone()).await.unwrap()
            );
            let response = request_anthropic(request_body.clone()).await;
            match response {
                Ok(res) => {
                    println!("{}", res);
                    println!("{:?}", res.cost().unwrap());
                }
                Err(e) => println!("Error: {}", e),
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
