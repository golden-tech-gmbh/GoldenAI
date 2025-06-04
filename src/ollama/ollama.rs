use anyhow::{Result, anyhow};
use std::time::Duration;

use crate::ollama::structs::{ConvertedOllamaRequest, OllamaRequest, OllamaResponse};
use crate::response::LLMResponse;

#[tokio::main]
pub async fn get_response_ollama(request_body: OllamaRequest) -> Result<LLMResponse> {
    request_ollama(request_body).await
}

async fn request_ollama(request_body: OllamaRequest) -> Result<LLMResponse> {
    // check if url is connectable
    let url = format!("{}/api/version", request_body.url);
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .timeout(Duration::from_secs(3))
        .send()
        .await?;

    if !response.status().is_success() {
        let err_status = response.status();
        let error_text = response.text().await?;
        return Err(anyhow!(
            "Error: HTTP {}, Response: {}",
            err_status,
            error_text
        ));
    }

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/api/generate", request_body.url))
        .json(&ConvertedOllamaRequest::from_ollama_request(request_body))
        .send()
        .await?;

    if response.status().is_success() {
        // let response_text = response.text().await?;
        // println!("Raw response: {}", response_text);
        // let response: OllamaResponse = serde_json::from_str(&response_text)?;
        let response: OllamaResponse = response.json().await?;
        Ok(response.to_llm_response())
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
async fn test_ollama_request() {
    use std::env;

    use crate::message::{Content, ContentTypeInner, Message, TextContent};

    let url = env::var("OLLAMA_URL").unwrap_or("".to_string());
    if url.is_empty() {
        panic!("OLLAMA_URL environment variable must be set");
    }

    let request_body = OllamaRequest::new(
        &url,
        "qwen2.5vl:latest",
        vec![Message {
            role: "user".to_string(),
            content: vec![Content {
                ctx: ContentTypeInner::Text(TextContent {
                    content_type: "text".to_string(),
                    text: "What's the color of this picture?".to_string(),
                }),
            }],
        }],
        Some("Please answer in Chinese"),
        Some("white.jpg"),
    )
    .unwrap();

    let res = request_ollama(request_body).await.unwrap_or_else(|e| {
        println!("Error: {}", e);
        panic!();
    });
    println!("{:?}", res);
}
