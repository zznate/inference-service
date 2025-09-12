use async_trait::async_trait;
use serde::Serialize;
use std::fmt;
use crate::config::HttpConfigSchema;
use crate::models::Message;

pub mod lmstudio;

/// Response from any inference provider
/// Returned internally for the InferenceProvider to manage
/// This is converted into a CompletionResponse in main for the API 
#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    pub text: String,
    pub model_used: String,
    pub total_tokens: Option<u32>,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

/// Error types that providers can return
#[derive(Debug)]
pub enum ProviderError {
    ConnectionFailed(String),
    InvalidResponse(String),
    ModelNotAvailable { requested: String, available: Vec<String> },
    RequestFailed { status: u16, message: String },
    Timeout,
    Configuration(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            ProviderError::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
            ProviderError::ModelNotAvailable { requested, available } => {
                write!(f, "Model '{}' not available. Available models: {:?}", requested, available)
            }
            ProviderError::RequestFailed { status, message } => {
                write!(f, "Request failed with status {}: {}", status, message)
            }
            ProviderError::Timeout => write!(f, "Request timed out"),
            ProviderError::Configuration(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for ProviderError {}

/// Common trait that all inference providers must implement
#[async_trait]
pub trait InferenceProvider: Send + Sync {
    /// Generate a completion for the given messages
    async fn generate(
        &self,
        messages: &[Message],
        model: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<InferenceResponse, ProviderError>;
    
    /// Get the name of this provider (for logging/metrics)
    fn name(&self) -> &str;

    /// Get HTTP configuration if this provider uses HTTP
    /// Returns None for local providers (e.g., Triton)
    fn http_config(&self) -> Option<&HttpConfigSchema> {
        None
    }
    
    /// Optional: List available models
    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        Err(ProviderError::Configuration("Model listing not supported".into()))
    }
    
    /// Optional: Check if the provider is healthy/reachable
    async fn health_check(&self) -> Result<(), ProviderError> {
        Ok(())
    }
}