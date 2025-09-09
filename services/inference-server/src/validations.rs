use axum::{http::StatusCode, response::IntoResponse, response::Response, Json};
use crate::ErrorResponse;
use crate::models::CompletionRequest;


#[derive(Debug)]
pub enum ValidationError {
    EmptyPrompt,
    InvalidMaxTokens(u32),
    InvalidTemperature(f32),
    InvalidModel(String),
}

impl IntoResponse for ValidationError {
    fn into_response(self) -> Response {

        let (status, error_code, message) = match self { 
            ValidationError::EmptyPrompt => (
                StatusCode::BAD_REQUEST,
                "EMPTY_PROMPT",
                "Prompt cannot be empty".to_string(),
            ),
            ValidationError::InvalidMaxTokens(max_tokens) => (
                StatusCode::BAD_REQUEST,
                "INVALID_MAX_TOKENS",
                format!("Max tokens must be between 1 and 4096, got {}", max_tokens),
            ),
            ValidationError::InvalidTemperature(temperature) => (
                StatusCode::BAD_REQUEST,
                "INVALID_TEMPERATURE",
                format!("Temperature must be between 0.0 and 2.0, got {}", temperature),
            ),
            ValidationError::InvalidModel(m) => (
                StatusCode::BAD_REQUEST,
                "INVALID_MODEL",
                format!("Model '{}' is not allowed or available", m),
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
    if request.prompt.trim().is_empty() {
        return Err(ValidationError::EmptyPrompt);
    }

    if let Some(max_tokens) = request.max_tokens {
        if max_tokens == 0 || max_tokens > 4096 {
            return Err(ValidationError::InvalidMaxTokens(max_tokens));
        }
    }
    
    if let Some(temperature) = request.temperature {    
        if temperature < 0.0 || temperature > 2.0 {
            return Err(ValidationError::InvalidTemperature(temperature));
        }
    }
    
    Ok(())  
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompletionRequest;  // Import from main
    
    #[test]
    fn test_validate_empty_prompt() {
        let request = CompletionRequest {
            prompt: "".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            model: Some("gpt-oss-20b".to_string()),
        };
        
        let result = validate_completion_request(&request);
        assert!(matches!(result, Err(ValidationError::EmptyPrompt)));
    }
    
    #[test]
    fn test_validate_whitespace_prompt() {
        let request = CompletionRequest {
            prompt: "   \n  ".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            model: Some("gpt-oss-20b".to_string()),
        };
        
        let result = validate_completion_request(&request);
        assert!(matches!(result, Err(ValidationError::EmptyPrompt)));
    }
    
    #[test]
    fn test_validate_invalid_max_tokens_zero() {
        let request = CompletionRequest {
            prompt: "Valid prompt".to_string(),
            max_tokens: Some(0),
            temperature: Some(0.7),
            model: Some("gpt-oss-20b".to_string()),
        };
        
        let result = validate_completion_request(&request);
        assert!(matches!(result, Err(ValidationError::InvalidMaxTokens(0))));
    }
    
    #[test]
    fn test_validate_invalid_max_tokens_too_high() {
        let request = CompletionRequest {
            prompt: "Valid prompt".to_string(),
            max_tokens: Some(5000),
            temperature: Some(0.7),
            model: Some("gpt-oss-20b".to_string()),
        };
        
        let result = validate_completion_request(&request);
        assert!(matches!(result, Err(ValidationError::InvalidMaxTokens(5000))));
    }
    
    #[test]
    fn test_validate_invalid_temperature_too_low() {
        let request = CompletionRequest {
            prompt: "Valid prompt".to_string(),
            max_tokens: Some(100),
            temperature: Some(-0.1),
            model: Some("gpt-oss-20b".to_string()),
        };
        
        let result = validate_completion_request(&request);
        assert!(matches!(result, Err(ValidationError::InvalidTemperature(_))));
    }
    
    #[test]
    fn test_validate_invalid_temperature_too_high() {
        let request = CompletionRequest {
            prompt: "Valid prompt".to_string(),
            max_tokens: Some(100),
            temperature: Some(2.1),
            model: Some("gpt-oss-20b".to_string()),
        };
        
        let result = validate_completion_request(&request);
        assert!(matches!(result, Err(ValidationError::InvalidTemperature(_))));
    }
    
    #[test]
    fn test_validate_success_with_all_fields() {
        let request = CompletionRequest {
            prompt: "What is 2+2?".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            model: Some("gpt-oss-20b".to_string()),
        };
        
        let result = validate_completion_request(&request);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_success_with_optional_fields() {
        let request = CompletionRequest {
            prompt: "What is 2+2?".to_string(),
            max_tokens: None,
            temperature: None,
            model: Some("gpt-oss-20b".to_string()),
        };
        
        let result = validate_completion_request(&request);
        assert!(result.is_ok());
    }
}