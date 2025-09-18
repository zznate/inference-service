use axum::{http::StatusCode, response::IntoResponse, response::Response, Json};
use crate::error::ErrorResponse;
use crate::models::CompletionRequest;
use std::collections::HashSet;

#[derive(Debug)]
pub enum ValidationError {
    EmptyMessages,
    NoContent,  // All messages have null content and no tool calls
    InvalidMaxTokens(u32),
    InvalidTemperature(f32),
    InvalidTopP(f32),
    InvalidFrequencyPenalty(f32),
    InvalidPresencePenalty(f32),
    InvalidTopLogprobs(u8),
    InvalidN(u32),
    ModelNotInAllowedList { model: String , allowed: Vec<String> },
    StreamingNotSupported,
    ToolsNotSupported,
}

impl IntoResponse for ValidationError {
    fn into_response(self) -> Response {

        let (status, error_code, message) = match self { 
            ValidationError::EmptyMessages => (
                StatusCode::BAD_REQUEST,
                "EMPTY_MESSAGES",
                "Messages array cannot be empty".to_string(),
            ),
            ValidationError::NoContent => (
                StatusCode::BAD_REQUEST,
                "NO_CONTENT",
                "At least one message must have content or tool calls".to_string(),
            ),
            ValidationError::InvalidMaxTokens(max_tokens) => (
                StatusCode::BAD_REQUEST,
                "INVALID_MAX_TOKENS",
                format!("Max tokens must be between 1 and 128000, got {}", max_tokens),
            ),
            ValidationError::InvalidTemperature(temperature) => (
                StatusCode::BAD_REQUEST,
                "INVALID_TEMPERATURE",
                format!("Temperature must be between 0.0 and 2.0, got {}", temperature),
            ),
            ValidationError::InvalidTopP(top_p) => (
                StatusCode::BAD_REQUEST,
                "INVALID_TOP_P",
                format!("Top-p must be between 0.0 and 1.0, got {}", top_p),
            ),
            ValidationError::InvalidFrequencyPenalty(penalty) => (
                StatusCode::BAD_REQUEST,
                "INVALID_FREQUENCY_PENALTY",
                format!("Frequency penalty must be between -2.0 and 2.0, got {}", penalty),
            ),
            ValidationError::InvalidPresencePenalty(penalty) => (
                StatusCode::BAD_REQUEST,
                "INVALID_PRESENCE_PENALTY",
                format!("Presence penalty must be between -2.0 and 2.0, got {}", penalty),
            ),
            ValidationError::InvalidTopLogprobs(n) => (
                StatusCode::BAD_REQUEST,
                "INVALID_TOP_LOGPROBS",
                format!("Top logprobs must be between 0 and 20, got {}", n),
            ),
            ValidationError::InvalidN(n) => (
                StatusCode::BAD_REQUEST,
                "INVALID_N",
                format!("N (number of choices) must be between 1 and 10, got {}", n),
            ),
            ValidationError::ModelNotInAllowedList { model, allowed } => (
                StatusCode::BAD_REQUEST,
                "MODEL_NOT_ALLOWED",
                format!(
                    "Model '{}' is not in the allowed list. Available models: {}",
                    model,
                    allowed.join(", ")
                ),
            ),
            ValidationError::StreamingNotSupported => (
                StatusCode::BAD_REQUEST,
                "STREAMING_NOT_SUPPORTED",
                "Streaming is not supported by the current provider".to_string(),
            ),
            ValidationError::ToolsNotSupported => (
                StatusCode::BAD_REQUEST,
                "TOOLS_NOT_SUPPORTED",
                "Tool/function calling is not supported by the current provider".to_string(),
            ),
        };   
    
        let body = Json(ErrorResponse {
            error: message,
            code: error_code.to_string(),
        });

        (status, body).into_response()
    }
}

pub fn validate_completion_request(request: &CompletionRequest) -> Result<(), ValidationError> {
    if request.messages.is_empty() {
        return Err(ValidationError::EmptyMessages);
    }
    
    // Check that at least one message has content or is a tool response
    let has_content = request.messages.iter().any(|msg| {
        msg.content.is_some() || 
        msg.tool_calls.is_some() || 
        msg.tool_call_id.is_some()
    });
    
    if !has_content {
        return Err(ValidationError::NoContent);
    }

    // Validate max_tokens if present
    if let Some(max_tokens) = request.max_tokens {
        if max_tokens == 0 || max_tokens > 128000 {  // GPT-4 max context
            return Err(ValidationError::InvalidMaxTokens(max_tokens));
        }
    }
    
    // Validate temperature
    if let Some(temperature) = request.temperature {    
        if temperature < 0.0 || temperature > 2.0 {
            return Err(ValidationError::InvalidTemperature(temperature));
        }
    }
    
    // Validate top_p
    if let Some(top_p) = request.top_p {
        if top_p < 0.0 || top_p > 1.0 {
            return Err(ValidationError::InvalidTopP(top_p));
        }
    }
    
    // Validate frequency_penalty
    if let Some(penalty) = request.frequency_penalty {
        if penalty < -2.0 || penalty > 2.0 {
            return Err(ValidationError::InvalidFrequencyPenalty(penalty));
        }
    }
    
    // Validate presence_penalty
    if let Some(penalty) = request.presence_penalty {
        if penalty < -2.0 || penalty > 2.0 {
            return Err(ValidationError::InvalidPresencePenalty(penalty));
        }
    }
    
    // Validate top_logprobs
    if let Some(top_logprobs) = request.top_logprobs {
        if top_logprobs > 20 {
            return Err(ValidationError::InvalidTopLogprobs(top_logprobs));
        }
    }
    
    // Validate n (number of choices)
    if let Some(n) = request.n {
        if n == 0 || n > 10 {
            return Err(ValidationError::InvalidN(n));
        }
    }
    
    Ok(())  
}

pub fn validate_model_allowed(
    requested_model: &str,
    allowed_models: Option<&HashSet<String>>,
) -> Result<(), ValidationError> {
    if let Some(allowed) = allowed_models {
        if !allowed.contains(requested_model) {
            return Err(ValidationError::ModelNotInAllowedList {
                model: requested_model.to_string(),
                allowed: allowed.iter().cloned().collect(),
            });
        }
    }
    Ok(())
}

pub fn validate_provider_capabilities(
    request: &CompletionRequest,
    supports_streaming: bool,
    supports_tools: bool,
) -> Result<(), ValidationError> {
    // Check streaming support
    if request.stream == Some(true) && !supports_streaming {
        return Err(ValidationError::StreamingNotSupported);
    }
    
    // Check tool/function support
    let needs_tools = request.tools.is_some() || 
                      request.functions.is_some() ||
                      request.messages.iter().any(|m| m.tool_calls.is_some() || m.tool_call_id.is_some());
    
    if needs_tools && !supports_tools {
        return Err(ValidationError::ToolsNotSupported);
    }
    
    Ok(())
}

pub fn determine_model<'a>(
    requested_model: Option<&'a str>,
    default_model: &'a str,
    allowed_models: Option<&HashSet<String>>,
) -> Result<&'a str, ValidationError> {
    match requested_model {
        Some(model) => {
            validate_model_allowed(model, allowed_models)?;
            Ok(model)
        },
        None => Ok(default_model),
    }   
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Message;
    
    #[test]
    fn test_validate_empty_messages() {
        let request = CompletionRequest {
            messages: vec![],
            max_tokens: Some(100),
            temperature: Some(0.7),
            model: Some("gpt-oss-20b".to_string()),
            ..Default::default()
        };
        
        let result = validate_completion_request(&request);
        assert!(matches!(result, Err(ValidationError::EmptyMessages)));
    }

    #[test]
    fn test_validate_null_content_allowed_with_tools() {
        let request = CompletionRequest {
            messages: vec![Message {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![]),
                ..Default::default()
            }],
            model: Some("gpt-4".to_string()),
            ..Default::default()
        };
        
        let result = validate_completion_request(&request);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_all_null_content_fails() {
        let request = CompletionRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: None,
                ..Default::default()
            }],
            model: Some("gpt-4".to_string()),
            ..Default::default()
        };
        
        let result = validate_completion_request(&request);
        assert!(matches!(result, Err(ValidationError::NoContent)));
    }
    
    #[test]
    fn test_validate_frequency_penalty_bounds() {
        let request = CompletionRequest {
            messages: vec![Message::new("user", "test")],
            frequency_penalty: Some(2.1),
            ..Default::default()
        };
        
        let result = validate_completion_request(&request);
        assert!(matches!(result, Err(ValidationError::InvalidFrequencyPenalty(_))));
    }
    
    #[test]
    fn test_validate_provider_capabilities() {
        let request = CompletionRequest {
            messages: vec![Message::new("user", "test")],
            stream: Some(true),
            ..Default::default()
        };
        
        let result = validate_provider_capabilities(&request, false, false);
        assert!(matches!(result, Err(ValidationError::StreamingNotSupported)));
    }
}

// Provide a default implementation for CompletionRequest
impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            model: None,
            frequency_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            max_tokens: None,
            n: None,
            presence_penalty: None,
            response_format: None,
            seed: None,
            stop: None,
            stream: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            functions: None,
            function_call: None,
            user: None,
        }
    }
}