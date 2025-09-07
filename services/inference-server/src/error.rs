use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::validations::ValidationError;
use crate::lm_client_studio::LMStudioError;

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

#[derive(Debug)]
pub enum ApiError {
    Validation(ValidationError),
    LMStudio(LMStudioError),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Validation(e) => e.into_response(),
            ApiError::LMStudio(e) => e.into_response(),
        }
    }
}

impl From<ValidationError> for ApiError {
    fn from(err: ValidationError) -> Self {
        ApiError::Validation(err)
    }
}

impl From<LMStudioError> for ApiError {
    fn from(err: LMStudioError) -> Self {
        ApiError::LMStudio(err)
    }
}