use anyhow::{Result, anyhow};
use reqwest;
use std::env;

use crate::req_structs::AnthropicRequest;
use crate::res_structs::AnthropicResponse;

#[tokio::main]
pub async fn get_response(request_body: AnthropicRequest) -> Result<AnthropicResponse> {
    request(request_body).await
}

async fn request(request_body: AnthropicRequest) -> Result<AnthropicResponse> {
    let api_key =
        env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY environment variable must be set");

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
        // let response: AnthropicResponse = serde_json::from_str(&response_text)?;
        let response: AnthropicResponse = response.json().await?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_request() {
        let request_body = AnthropicRequest {
            model: "claude-3-5-haiku-latest".to_string(),
            max_tokens: 1024,
            messages: vec![crate::req_structs::Message {
                role: "user".to_string(),
                content: vec![crate::req_structs::Content {
                    ctx: crate::req_structs::ContentTypeInner::Text(
                        crate::req_structs::TextContent {
                            content_type: "text".to_string(),
                            text: "Hello, Claude!".to_string(),
                        },
                    ),
                }],
            }],
        };
        let response = request(request_body).await;
        match response {
            Ok(res) => println!("{:?}", res),
            Err(e) => println!("Error: {}", e),
        }
    }
}
