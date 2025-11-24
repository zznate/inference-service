use axum::http::StatusCode;
use axum::response::{IntoResponse, Response, Json};

use crate::models::{OpenAIError, OpenAIErrorResponse};
use crate::providers::ProviderError;
use crate::validations::ValidationError;

#[derive(Debug)]
pub enum ApiError {
    Validation(ValidationError),
    Provider(ProviderError),
}

impl From<ProviderError> for ApiError {
    fn from(err: ProviderError) -> Self {
        ApiError::Provider(err)
    }
}

impl From<ValidationError> for ApiError {
    fn from(err: ValidationError) -> Self {
        ApiError::Validation(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, openai_error) = match self {
            ApiError::Validation(e) => (e.status_code(), e.to_openai_error()),
            ApiError::Provider(e) => (e.status_code(), e.to_openai_error()),
        };

        let error_response = OpenAIErrorResponse {
            error: openai_error,
        };

        (status, Json(error_response)).into_response()
    }
}

// Extension trait for ProviderError to convert to OpenAI format
impl ProviderError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            ProviderError::ConnectionFailed(_) => StatusCode::BAD_GATEWAY,
            ProviderError::InvalidResponse(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ProviderError::RequestFailed { status, .. } => {
                StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
            }
            ProviderError::ModelNotAvailable { .. } => StatusCode::BAD_REQUEST,
            ProviderError::Timeout => StatusCode::GATEWAY_TIMEOUT,
            ProviderError::Configuration(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ProviderError::StreamingNotSupported => StatusCode::BAD_REQUEST,
            ProviderError::StreamError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ProviderError::InvalidExtension { .. } => StatusCode::BAD_REQUEST,
        }
    }

    pub fn to_openai_error(&self) -> OpenAIError {
        match self {
            ProviderError::ConnectionFailed(msg) => OpenAIError {
                message: format!("Failed to connect to inference provider: {}", msg),
                error_type: "api_error".to_string(),
                param: None,
                code: Some("provider_connection_failed".to_string()),
            },
            ProviderError::InvalidResponse(msg) => OpenAIError {
                message: format!("Invalid response from inference provider: {}", msg),
                error_type: "api_error".to_string(),
                param: None,
                code: Some("provider_invalid_response".to_string()),
            },
            ProviderError::RequestFailed { status, message } => {
                let (error_type, code) = match status {
                    401 => ("authentication_error", "invalid_api_key"),
                    403 => ("permission_error", "insufficient_quota"),
                    429 => ("rate_limit_error", "rate_limit_exceeded"),
                    _ => ("api_error", "provider_error"),
                };
                OpenAIError {
                    message: message.clone(),
                    error_type: error_type.to_string(),
                    param: None,
                    code: Some(code.to_string()),
                }
            }
            ProviderError::ModelNotAvailable { requested, available } => {
                let available_str = if available.is_empty() {
                    "No models available".to_string()
                } else {
                    format!("Available models: {}", available.join(", "))
                };
                OpenAIError {
                    message: format!("Model '{}' is not available. {}", requested, available_str),
                    error_type: "invalid_request_error".to_string(),
                    param: Some("model".to_string()),
                    code: Some("model_not_found".to_string()),
                }
            }
            ProviderError::Timeout => OpenAIError {
                message: "Request to inference provider timed out".to_string(),
                error_type: "timeout_error".to_string(),
                param: None,
                code: Some("provider_timeout".to_string()),
            },
            ProviderError::Configuration(msg) => OpenAIError {
                message: format!("Provider configuration error: {}", msg),
                error_type: "api_error".to_string(),
                param: None,
                code: Some("configuration_error".to_string()),
            },
            ProviderError::StreamingNotSupported => OpenAIError {
                message: "Streaming is not supported by the current provider".to_string(),
                error_type: "invalid_request_error".to_string(),
                param: Some("stream".to_string()),
                code: Some("unsupported_parameter".to_string()),
            },
            ProviderError::StreamError(msg) => OpenAIError {
                message: format!("Streaming error: {}", msg),
                error_type: "api_error".to_string(),
                param: None,
                code: Some("stream_error".to_string()),
            },
            ProviderError::InvalidExtension { param, reason } => OpenAIError {
                message: format!("Invalid extension parameter '{}': {}", param, reason),
                error_type: "invalid_request_error".to_string(),
                param: Some(param.clone()),
                code: Some("invalid_extension".to_string()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_connection_failed_error() {
        let error = ProviderError::ConnectionFailed("Network error".to_string());
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "api_error");
        assert_eq!(openai_error.code, Some("provider_connection_failed".to_string()));
        assert!(openai_error.message.contains("Network error"));
        assert_eq!(openai_error.param, None);
    }

    #[test]
    fn test_provider_timeout_error() {
        let error = ProviderError::Timeout;
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "timeout_error");
        assert_eq!(openai_error.code, Some("provider_timeout".to_string()));
        assert!(openai_error.message.contains("timed out"));
    }

    #[test]
    fn test_provider_authentication_error() {
        let error = ProviderError::RequestFailed {
            status: 401,
            message: "Invalid API key".to_string(),
        };
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "authentication_error");
        assert_eq!(openai_error.code, Some("invalid_api_key".to_string()));
        assert_eq!(openai_error.message, "Invalid API key");
    }

    #[test]
    fn test_provider_rate_limit_error() {
        let error = ProviderError::RequestFailed {
            status: 429,
            message: "Rate limit exceeded".to_string(),
        };
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "rate_limit_error");
        assert_eq!(openai_error.code, Some("rate_limit_exceeded".to_string()));
    }

    #[test]
    fn test_provider_permission_error() {
        let error = ProviderError::RequestFailed {
            status: 403,
            message: "Insufficient quota".to_string(),
        };
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "permission_error");
        assert_eq!(openai_error.code, Some("insufficient_quota".to_string()));
    }

    #[test]
    fn test_model_not_available_error() {
        let error = ProviderError::ModelNotAvailable {
            requested: "gpt-5".to_string(),
            available: vec!["gpt-3.5-turbo".to_string(), "gpt-4".to_string()],
        };
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("model".to_string()));
        assert_eq!(openai_error.code, Some("model_not_found".to_string()));
        assert!(openai_error.message.contains("gpt-5"));
        assert!(openai_error.message.contains("gpt-3.5-turbo"));
    }

    #[test]
    fn test_streaming_not_supported_error() {
        let error = ProviderError::StreamingNotSupported;
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("stream".to_string()));
        assert_eq!(openai_error.code, Some("unsupported_parameter".to_string()));
    }

    #[test]
    fn test_invalid_extension_error() {
        let error = ProviderError::InvalidExtension {
            param: "custom_param".to_string(),
            reason: "Not supported".to_string(),
        };
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "invalid_request_error");
        assert_eq!(openai_error.param, Some("custom_param".to_string()));
        assert_eq!(openai_error.code, Some("invalid_extension".to_string()));
        assert!(openai_error.message.contains("custom_param"));
        assert!(openai_error.message.contains("Not supported"));
    }

    #[test]
    fn test_provider_error_status_codes() {
        assert_eq!(
            ProviderError::ConnectionFailed("test".to_string()).status_code(),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            ProviderError::Timeout.status_code(),
            StatusCode::GATEWAY_TIMEOUT
        );
        assert_eq!(
            ProviderError::RequestFailed {
                status: 401,
                message: "test".to_string()
            }
            .status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            ProviderError::ModelNotAvailable {
                requested: "test".to_string(),
                available: vec![]
            }
            .status_code(),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn test_configuration_error() {
        let error = ProviderError::Configuration("Invalid config".to_string());
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "api_error");
        assert_eq!(openai_error.code, Some("configuration_error".to_string()));
        assert!(openai_error.message.contains("Invalid config"));
    }

    #[test]
    fn test_stream_error() {
        let error = ProviderError::StreamError("Connection lost".to_string());
        let openai_error = error.to_openai_error();

        assert_eq!(openai_error.error_type, "api_error");
        assert_eq!(openai_error.code, Some("stream_error".to_string()));
        assert!(openai_error.message.contains("Connection lost"));
    }
}
