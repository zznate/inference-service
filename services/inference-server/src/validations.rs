use crate::models::{CompletionRequest, OpenAIError, OpenAIErrorResponse};
use axum::{Json, http::StatusCode, response::IntoResponse, response::Response};
use std::collections::HashSet;

#[derive(Debug)]
pub enum ValidationError {
    EmptyMessages,
    NoContent, // All messages have null content and no tool calls
    InvalidMaxTokens(u32),
    InvalidTemperature(f32),
    InvalidTopP(f32),
    InvalidFrequencyPenalty(f32),
    InvalidPresencePenalty(f32),
    InvalidTopLogprobs(u8),
    InvalidN(u32),
    ModelNotInAllowedList { model: String, allowed: Vec<String> },
    StreamingNotSupported,
    InvalidLogitBias { token_id: String, reason: String },
}

impl ValidationError {
    pub fn status_code(&self) -> StatusCode {
        // All validation errors return 400 Bad Request
        StatusCode::BAD_REQUEST
    }

    pub fn to_openai_error(&self) -> OpenAIError {
        match self {
            ValidationError::EmptyMessages => OpenAIError {
                message: "Messages array cannot be empty".to_string(),
                error_type: "invalid_request_error".to_string(),
                param: Some("messages".to_string()),
                code: None,
            },
            ValidationError::NoContent => OpenAIError {
                message: "At least one message must have content or tool calls".to_string(),
                error_type: "invalid_request_error".to_string(),
                param: Some("messages".to_string()),
                code: None,
            },
            ValidationError::InvalidMaxTokens(max_tokens) => OpenAIError {
                message: format!("Max tokens must be between 1 and 128000, got {}", max_tokens),
                error_type: "invalid_request_error".to_string(),
                param: Some("max_tokens".to_string()),
                code: None,
            },
            ValidationError::InvalidTemperature(temperature) => OpenAIError {
                message: format!("Temperature must be between 0.0 and 2.0, got {}", temperature),
                error_type: "invalid_request_error".to_string(),
                param: Some("temperature".to_string()),
                code: None,
            },
            ValidationError::InvalidTopP(top_p) => OpenAIError {
                message: format!("Top-p must be between 0.0 and 1.0, got {}", top_p),
                error_type: "invalid_request_error".to_string(),
                param: Some("top_p".to_string()),
                code: None,
            },
            ValidationError::InvalidFrequencyPenalty(penalty) => OpenAIError {
                message: format!("Frequency penalty must be between -2.0 and 2.0, got {}", penalty),
                error_type: "invalid_request_error".to_string(),
                param: Some("frequency_penalty".to_string()),
                code: None,
            },
            ValidationError::InvalidPresencePenalty(penalty) => OpenAIError {
                message: format!("Presence penalty must be between -2.0 and 2.0, got {}", penalty),
                error_type: "invalid_request_error".to_string(),
                param: Some("presence_penalty".to_string()),
                code: None,
            },
            ValidationError::InvalidTopLogprobs(n) => OpenAIError {
                message: format!("Top logprobs must be between 0 and 20, got {}", n),
                error_type: "invalid_request_error".to_string(),
                param: Some("top_logprobs".to_string()),
                code: None,
            },
            ValidationError::InvalidN(n) => OpenAIError {
                message: format!("N (number of choices) must be between 1 and 10, got {}", n),
                error_type: "invalid_request_error".to_string(),
                param: Some("n".to_string()),
                code: None,
            },
            ValidationError::ModelNotInAllowedList { model, allowed } => OpenAIError {
                message: format!(
                    "Model '{}' is not in the allowed list. Available models: {}",
                    model,
                    allowed.join(", ")
                ),
                error_type: "invalid_request_error".to_string(),
                param: Some("model".to_string()),
                code: Some("model_not_found".to_string()),
            },
            ValidationError::StreamingNotSupported => OpenAIError {
                message: "Streaming is not supported by the current provider".to_string(),
                error_type: "invalid_request_error".to_string(),
                param: Some("stream".to_string()),
                code: Some("unsupported_parameter".to_string()),
            },
            ValidationError::InvalidLogitBias { token_id, reason } => OpenAIError {
                message: format!("Invalid logit bias for token '{}': {}", token_id, reason),
                error_type: "invalid_request_error".to_string(),
                param: Some("logit_bias".to_string()),
                code: None,
            },
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_openai_error().message)
    }
}

impl std::error::Error for ValidationError {}

impl IntoResponse for ValidationError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_response = OpenAIErrorResponse {
            error: self.to_openai_error(),
        };

        (status, Json(error_response)).into_response()
    }
}

pub fn validate_completion_request(request: &CompletionRequest) -> Result<(), ValidationError> {
    if request.messages.is_empty() {
        return Err(ValidationError::EmptyMessages);
    }

    // Check that at least one message has content or is a tool response
    let has_content = request
        .messages
        .iter()
        .any(|msg| msg.content.is_some() || msg.tool_calls.is_some() || msg.tool_call_id.is_some());

    if !has_content {
        return Err(ValidationError::NoContent);
    }

    // Validate max_tokens if present
    if let Some(max_tokens) = request.max_tokens
        && (max_tokens == 0 || max_tokens > 128000)
    {
        return Err(ValidationError::InvalidMaxTokens(max_tokens));
    }

    // Validate temperature
    if let Some(temperature) = request.temperature
        && !(0.0..=2.0).contains(&temperature)
    {
        return Err(ValidationError::InvalidTemperature(temperature));
    }

    // Validate top_p
    if let Some(top_p) = request.top_p
        && !(0.0..=1.0).contains(&top_p)
    {
        return Err(ValidationError::InvalidTopP(top_p));
    }

    // Validate frequency_penalty
    if let Some(penalty) = request.frequency_penalty
        && !(-2.0..=2.0).contains(&penalty)
    {
        return Err(ValidationError::InvalidFrequencyPenalty(penalty));
    }

    // Validate presence_penalty
    if let Some(penalty) = request.presence_penalty
        && !(-2.0..=2.0).contains(&penalty)
    {
        return Err(ValidationError::InvalidPresencePenalty(penalty));
    }

    // Validate top_logprobs
    if let Some(top_logprobs) = request.top_logprobs
        && top_logprobs > 20
    {
        return Err(ValidationError::InvalidTopLogprobs(top_logprobs));
    }

    // Validate n (number of choices)
    if let Some(n) = request.n
        && (n == 0 || n > 10)
    {
        return Err(ValidationError::InvalidN(n));
    }

    // Note: response_format.format_type is now validated at deserialization by the FormatType enum

    // Validate logit_bias
    if let Some(ref logit_bias) = request.logit_bias {
        for (token_id, bias_value) in logit_bias {
            // Validate token ID is numeric
            if token_id.parse::<i64>().is_err() {
                return Err(ValidationError::InvalidLogitBias {
                    token_id: token_id.clone(),
                    reason: "token ID must be a numeric string".to_string(),
                });
            }

            // Validate bias value is a number between -100 and 100
            if let Some(bias) = bias_value.as_f64() {
                if !(-100.0..=100.0).contains(&bias) {
                    return Err(ValidationError::InvalidLogitBias {
                        token_id: token_id.clone(),
                        reason: format!("bias value must be between -100 and 100, got {bias}"),
                    });
                }
            } else {
                return Err(ValidationError::InvalidLogitBias {
                    token_id: token_id.clone(),
                    reason: "bias value must be a number".to_string(),
                });
            }
        }
    }

    Ok(())
}

pub fn validate_model_allowed(
    requested_model: &str,
    allowed_models: Option<&HashSet<String>>,
) -> Result<(), ValidationError> {
    if let Some(allowed) = allowed_models
        && !allowed.contains(requested_model)
    {
        return Err(ValidationError::ModelNotInAllowedList {
            model: requested_model.to_string(),
            allowed: allowed.iter().cloned().collect(),
        });
    }
    Ok(())
}

pub fn validate_provider_capabilities(
    request: &CompletionRequest,
    supports_streaming: bool,
    _supports_tools: bool,
) -> Result<(), ValidationError> {
    // Check streaming support
    if request.stream == Some(true) && !supports_streaming {
        return Err(ValidationError::StreamingNotSupported);
    }

    // Note: Tool/function calling validation removed - not currently implemented
    // Can be re-added when provider tool support is implemented

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
        }
        None => Ok(default_model),
    }
}

// CompletionRequest gets Default derived automatically since all fields are Option or have defaults

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Message, Role};

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
                role: Role::Assistant,
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
                role: Role::User,
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
            messages: vec![Message::new(Role::User, "test")],
            frequency_penalty: Some(2.1),
            ..Default::default()
        };

        let result = validate_completion_request(&request);
        assert!(matches!(
            result,
            Err(ValidationError::InvalidFrequencyPenalty(_))
        ));
    }

    #[test]
    fn test_validate_provider_capabilities() {
        let request = CompletionRequest {
            messages: vec![Message::new(Role::User, "test")],
            stream: Some(true),
            ..Default::default()
        };

        let result = validate_provider_capabilities(&request, false, false);
        assert!(matches!(
            result,
            Err(ValidationError::StreamingNotSupported)
        ));
    }

    // OpenAI error format tests
    #[test]
    fn test_validation_error_to_openai_format() {
        let error = ValidationError::InvalidTemperature(3.0);
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("temperature".to_string()));
        assert!(openai_error.message.contains("3"));
        assert!(openai_error.message.contains("0"));
        assert!(openai_error.message.contains("2"));
    }

    #[test]
    fn test_empty_messages_openai_error() {
        let error = ValidationError::EmptyMessages;
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("messages".to_string()));
        assert_eq!(openai_error.code, None);
    }

    #[test]
    fn test_model_not_allowed_openai_error() {
        let error = ValidationError::ModelNotInAllowedList {
            model: "gpt-5".to_string(),
            allowed: vec!["gpt-3.5-turbo".to_string(), "gpt-4".to_string()],
        };
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("model".to_string()));
        assert_eq!(openai_error.code, Some("model_not_found".to_string()));
        assert!(openai_error.message.contains("gpt-5"));
    }

    #[test]
    fn test_streaming_not_supported_openai_error() {
        let error = ValidationError::StreamingNotSupported;
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("stream".to_string()));
        assert_eq!(openai_error.code, Some("unsupported_parameter".to_string()));
    }

    #[test]
    fn test_invalid_max_tokens_openai_error() {
        let error = ValidationError::InvalidMaxTokens(200000);
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("max_tokens".to_string()));
        assert!(openai_error.message.contains("200000"));
    }

    #[test]
    fn test_invalid_top_p_openai_error() {
        let error = ValidationError::InvalidTopP(1.5);
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("top_p".to_string()));
        assert!(openai_error.message.contains("1.5"));
    }

    #[test]
    fn test_invalid_n_openai_error() {
        let error = ValidationError::InvalidN(15);
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("n".to_string()));
        assert!(openai_error.message.contains("15"));
    }

    #[test]
    fn test_invalid_logit_bias_openai_error() {
        let error = ValidationError::InvalidLogitBias {
            token_id: "12345".to_string(),
            reason: "Value out of range".to_string(),
        };
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("logit_bias".to_string()));
        assert!(openai_error.message.contains("12345"));
        assert!(openai_error.message.contains("Value out of range"));
    }

    #[test]
    fn test_validation_error_status_code() {
        let error = ValidationError::InvalidTemperature(3.0);
        assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);

        let error = ValidationError::EmptyMessages;
        assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
    }
}
