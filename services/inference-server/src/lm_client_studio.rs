use reqwest;
use serde_json;
use axum::{http::StatusCode, response::IntoResponse, response::Response, Json};
use tracing::{debug, error};

use crate::error::ErrorResponse;

#[derive(Debug)]
pub enum LMStudioError {
    RequestFailed(String),
    InvalidResponse(String),
    ServerError(StatusCode),
}

impl IntoResponse for LMStudioError {
    fn into_response(self) -> Response {
        let (status, error_code, message) = match self {
            LMStudioError::RequestFailed(msg) => (
                StatusCode::BAD_GATEWAY,
                "LM_STUDIO_REQUEST_FAILED",
                format!("Failed to connect to LM Studio: {}", msg),
            ),
            LMStudioError::InvalidResponse(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "LM_STUDIO_INVALID_RESPONSE",
                format!("Invalid response from LM Studio: {}", msg),
            ),
            LMStudioError::ServerError(code) => (
                code,
                "LM_STUDIO_SERVER_ERROR",
                format!("LM Studio returned error status: {}", code.as_u16()),
            ),
        };
        
        let body = Json(ErrorResponse {
            error: message,
            code: error_code.to_string(),
        });
        
        (status, body).into_response()
    }
}

pub struct LMStudioResponse {
    pub text: String,
    pub total_tokens: Option<u32>,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

pub async fn call_lm_studio(
    client: &reqwest::Client,
    base_url: &str,
    prompt: &str,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    model: &str,
) -> Result<LMStudioResponse, LMStudioError> {

    let request_body = serde_json::json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens.unwrap_or(100),
        "temperature": temperature.unwrap_or(0.7),
    });

    debug!("Sending request to LM Studio: {}", request_body);

    let response = client
        .post(format!("{}/chat/completions", base_url))
        .json(&request_body)
        .send()
        .await
        .map_err(|e| {
            error!("Failed to send request to LM Studio: {}", e);
            LMStudioError::RequestFailed(e.to_string())
        })?;

    if !response.status().is_success() {
        error!("LM Studio returned error status: {}", response.status());
        return Err(LMStudioError::ServerError(
            StatusCode::from_u16(response.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
        ));
    }

    let lm_response: serde_json::Value = response
        .json()
    .await
    .map_err(|e| {
        error!("Failed to parse LM Studio response: {}", e);
        LMStudioError::InvalidResponse(e.to_string())
    })?;

    debug!("LM Studio response: {}", lm_response);
    
    // Extract the response fields
    let text = lm_response["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| LMStudioError::InvalidResponse("Missing content field".to_string()))?
        .to_string();
    
    let total_tokens = lm_response["usage"]["total_tokens"]
        .as_u64()
        .map(|t| t as u32);
    
    let prompt_tokens = lm_response["usage"]["prompt_tokens"]
        .as_u64()
        .map(|t| t as u32);
        
    let completion_tokens = lm_response["usage"]["completion_tokens"]
        .as_u64()
        .map(|t| t as u32);
    
    Ok(LMStudioResponse {
        text,
        total_tokens,
        prompt_tokens,
        completion_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // For testing, we'll need to mock the HTTP client
    // This is more complex, so let's start with a simple example
    
    #[tokio::test]
    async fn test_error_handling() {
        // This test would require a mock server or test instance
        // For now, we can test the error types themselves
        let error = LMStudioError::RequestFailed("Connection refused".to_string());
        assert_eq!(error.into_response().status(), StatusCode::BAD_GATEWAY);
        
        let error = LMStudioError::InvalidResponse("Bad JSON".to_string());
        assert_eq!(error.into_response().status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}

