use async_trait::async_trait;
use serde::Serialize;
use std::fmt;
use crate::config::HttpConfigSchema;
use crate::models::{Message, CompletionRequest, CompletionResponse, Choice, Usage};
use uuid::Uuid;

pub mod lmstudio;
pub mod mock;
pub mod openai;

// ===== Internal Service Models =====

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
    pub stream: Option<bool>,
    pub n: Option<u32>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u8>,
    
    // Extension point for provider-specific parameters
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
    pub system_fingerprint: Option<String>,
    
    // For function/tool calls
    pub tool_calls: Option<Vec<crate::models::ToolCall>>,
    
    // For logprobs
    pub logprobs: Option<crate::models::LogProbs>,
    
    // Extension point for provider-specific response data
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
    StreamingNotSupported,
    ToolsNotSupported,
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
            ProviderError::StreamingNotSupported => write!(f, "Streaming is not supported by this provider"),
            ProviderError::ToolsNotSupported => write!(f, "Tool/function calling is not supported by this provider"),
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
    
    /// Check if streaming is supported
    fn supports_streaming(&self) -> bool {
        false
    }
    
    /// Check if tool/function calling is supported
    fn supports_tools(&self) -> bool {
        false
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
/// This handles all the optional fields properly
pub fn standard_completion_response(
    response: &InferenceResponse,
    _original_request: &CompletionRequest,
) -> CompletionResponse {
    // Build the message with optional fields
    let mut message = Message::new("assistant", &response.text);
    
    // Add tool calls if present
    if let Some(ref tool_calls) = response.tool_calls {
        message.tool_calls = Some(tool_calls.clone());
    }
    
    // Create choice with all optional fields
    let choice = Choice {
        index: 0,
        message: Some(message),
        delta: None,
        finish_reason: response.finish_reason.clone(),
        logprobs: response.logprobs.clone(),
    };
    
    // Build usage with optional fields
    let usage = if response.prompt_tokens.is_some() || 
                   response.completion_tokens.is_some() || 
                   response.total_tokens.is_some() {
        Some(Usage {
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
            total_tokens: response.total_tokens,
        })
    } else {
        None
    };
    
    CompletionResponse {
        id: response.provider_request_id
            .clone()
            .unwrap_or_else(|| format!("chatcmpl-{}", Uuid::now_v7())),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: response.model_used.clone(),
        choices: vec![choice],
        usage,
        system_fingerprint: response.system_fingerprint.clone(),
    }
}

/// Helper to convert stop sequences from various formats
pub fn normalize_stop_sequences(stop: &Option<crate::models::StringOrArray>) -> Option<Vec<String>> {
    stop.as_ref().map(|s| match s {
        crate::models::StringOrArray::String(s) => vec![s.clone()],
        crate::models::StringOrArray::Array(a) => a.clone(),
    })
}