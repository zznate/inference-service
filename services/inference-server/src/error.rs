use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::providers::ProviderError;
use crate::validations::ValidationError;

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

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

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Validation(e) => e.into_response(),
            ApiError::Provider(e) => {
                // We need to implement IntoResponse for ProviderError
                // For now, let's map it to appropriate HTTP status codes
                use axum::Json;
                use axum::http::StatusCode;

                let (status, code, message) = match e {
                    ProviderError::ConnectionFailed(msg) => {
                        (StatusCode::BAD_GATEWAY, "PROVIDER_CONNECTION_FAILED", msg)
                    }
                    ProviderError::InvalidResponse(msg) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "PROVIDER_INVALID_RESPONSE",
                        msg,
                    ),
                    ProviderError::RequestFailed { status, message } => (
                        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                        "PROVIDER_REQUEST_FAILED",
                        message,
                    ),
                    ProviderError::ModelNotAvailable { requested, .. } => (
                        StatusCode::BAD_REQUEST,
                        "MODEL_NOT_AVAILABLE",
                        format!("Model '{requested}' not available"),
                    ),
                    ProviderError::Timeout => (
                        StatusCode::GATEWAY_TIMEOUT,
                        "PROVIDER_TIMEOUT",
                        "Request timed out".to_string(),
                    ),
                    ProviderError::Configuration(msg) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "PROVIDER_CONFIGURATION_ERROR",
                        msg,
                    ),
                    ProviderError::StreamingNotSupported => (
                        StatusCode::BAD_REQUEST,
                        "STREAMING_NOT_SUPPORTED",
                        "Streaming is not supported by this provider".to_string(),
                    ),
                    ProviderError::StreamError(msg) => {
                        (StatusCode::INTERNAL_SERVER_ERROR, "STREAM_ERROR", msg)
                    }
                    ProviderError::InvalidExtension { param, reason } => (
                        StatusCode::BAD_REQUEST,
                        "INVALID_EXTENSION",
                        format!("Invalid extension parameter '{param}': {reason}"),
                    ),
                };

                (
                    status,
                    Json(ErrorResponse {
                        error: message,
                        code: code.to_string(),
                    }),
                )
                    .into_response()
            }
        }
    }
}

impl From<ValidationError> for ApiError {
    fn from(err: ValidationError) -> Self {
        ApiError::Validation(err)
    }
}
