use async_trait::async_trait;
use serde::Serialize;
use std::fmt;
use crate::config::HttpConfigSchema;
use crate::models::{Message, CompletionRequest, CompletionResponse, Choice, Usage};
use uuid::Uuid;

pub mod lmstudio;
pub mod mock;


// ===== Internal Service Models =====
// These structs represent our normalized internal format that all providers work with

/// Normalized request format that all providers understand
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    // Core fields that all providers need
    pub messages: Vec<Message>,
    pub model: String,  // Required internally (we apply defaults before this point)
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    
    // Common optional parameters
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub seed: Option<u64>,
    
    // Extension point for provider-specific parameters
    // This allows providers to pass through specific settings without
    // modifying the core struct
    pub provider_params: Option<serde_json::Value>,
}

/// Normalized response format that all providers return
#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    // Core response fields
    pub text: String,
    pub model_used: String,
    pub finish_reason: Option<String>,
    
    // Token usage information
    pub total_tokens: Option<u32>,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    
    // Additional metadata
    pub latency_ms: Option<u64>,
    pub provider_request_id: Option<String>,
    
    // Extension point for provider-specific response data
    // (e.g., Triton might return batch information, confidence scores, etc.)
    pub provider_metadata: Option<serde_json::Value>,
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
    /// Transform OpenAI-format request to our internal normalized format
    fn build_inference_request(
        &self,
        request: &CompletionRequest,
        model: &str,  // Model already determined by validation layer
    ) -> Result<InferenceRequest, ProviderError>;
    
    /// Execute the inference request against the provider
    /// Each provider implements their specific protocol here (HTTP, gRPC, etc.)
    async fn execute(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, ProviderError>;
    
    /// Transform our internal response format to OpenAI-compatible format
    fn build_completion_response(
        &self,
        response: &InferenceResponse,
        original_request: &CompletionRequest,
    ) -> CompletionResponse;
    
    /// High-level method that orchestrates the full flow
    /// Most providers can use this default implementation
    async fn generate(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> Result<CompletionResponse, ProviderError> {
        let inference_req = self.build_inference_request(request, model)?;
        let inference_resp = self.execute(&inference_req).await?;
        Ok(self.build_completion_response(&inference_resp, request))
    }
    
    /// Get the name of this provider (for logging/metrics)
    fn name(&self) -> &str;

    /// Get HTTP configuration if this provider uses HTTP
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

// ===== Helper Functions =====

/// Standard implementation for building CompletionResponse from InferenceResponse
/// This can be used by most providers as-is
pub fn standard_completion_response(
    response: &InferenceResponse,
    _original_request: &CompletionRequest,
) -> CompletionResponse {
    CompletionResponse {
        id: format!("chatcmpl-{}", Uuid::now_v7()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: response.model_used.clone(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: response.text.clone(),
            },
            finish_reason: response.finish_reason
                .as_deref()
                .unwrap_or("stop")
                .to_string(),
        }],
        usage: Usage {
            prompt_tokens: response.prompt_tokens.unwrap_or(0),
            completion_tokens: response.completion_tokens.unwrap_or(0),
            total_tokens: response.total_tokens.unwrap_or(0),
        },
    }
}