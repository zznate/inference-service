use crate::config::HttpConfigSchema;
use crate::models::{Choice, CompletionRequest, CompletionResponse, Message, Usage};
use async_trait::async_trait;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
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
    pub model: String, // Required internally (we apply defaults before this point)
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,

    // Common optional parameters
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub seed: Option<u64>,
    #[allow(dead_code)] // TODO: Implement streaming at InferenceRequest level
    pub stream: Option<bool>,
    #[allow(dead_code)] // TODO: Implement n-completions (multiple choices per request)
    pub n: Option<u32>,
    #[allow(dead_code)] // TODO: Implement logprobs support
    pub logprobs: Option<bool>,
    #[allow(dead_code)] // TODO: Implement logprobs support
    pub top_logprobs: Option<u8>,
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

    // Provider-specific extension data (for extended response mode)
    pub provider_data: Option<HashMap<String, serde_json::Value>>,
}

/// Error types that providers can return
#[derive(Debug)]
pub enum ProviderError {
    ConnectionFailed(String),
    InvalidResponse(String),
    ModelNotAvailable {
        requested: String,
        available: Vec<String>,
    },
    RequestFailed {
        status: u16,
        message: String,
    },
    Timeout,
    Configuration(String),
    StreamingNotSupported,
    StreamError(String),
    InvalidExtension {
        param: String,
        reason: String,
    },
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::ConnectionFailed(msg) => write!(f, "Connection failed: {msg}"),
            ProviderError::InvalidResponse(msg) => write!(f, "Invalid response: {msg}"),
            ProviderError::ModelNotAvailable {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Model '{requested}' not available. Available models: {available:?}"
                )
            }
            ProviderError::RequestFailed { status, message } => {
                write!(f, "Request failed with status {status}: {message}")
            }
            ProviderError::Timeout => write!(f, "Request timed out"),
            ProviderError::Configuration(msg) => write!(f, "Configuration error: {msg}"),
            ProviderError::StreamingNotSupported => {
                write!(f, "Streaming is not supported by this provider")
            }
            ProviderError::StreamError(msg) => write!(f, "Streaming error: {msg}"),
            ProviderError::InvalidExtension { param, reason } => {
                write!(f, "Invalid extension parameter '{param}': {reason}")
            }
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
        model: &str, // Model already determined by validation layer
    ) -> Result<InferenceRequest, ProviderError>;

    /// Execute the inference request against the provider
    async fn execute(&self, request: &InferenceRequest)
    -> Result<InferenceResponse, ProviderError>;

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

    /// Stream completion tokens as they're generated
    /// Default implementation converts non-streaming response to chunked stream
    async fn stream(
        &self,
        _request: &CompletionRequest,
        _model: &str,
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures_util::Stream<Item = Result<crate::models::StreamChunk, ProviderError>>
                    + Send,
            >,
        >,
        ProviderError,
    > {
        // Default: not supported
        Err(ProviderError::StreamingNotSupported)
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

    /// Get list of supported extension parameters for this provider
    /// Returns empty vec if no extensions are supported
    fn supported_extensions(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// Validate provider-specific extension parameters
    /// Default implementation rejects all extensions
    fn validate_extensions(
        &self,
        extensions: &HashMap<String, serde_json::Value>,
    ) -> Result<(), ProviderError> {
        // By default, if any extensions are provided and provider doesn't override this,
        // we check against supported_extensions
        let supported = self.supported_extensions();

        if supported.is_empty() && !extensions.is_empty() {
            // Provider doesn't support any extensions but some were provided
            let keys: Vec<_> = extensions.keys().map(|k| k.as_str()).collect();
            return Err(ProviderError::InvalidExtension {
                param: keys.join(", "),
                reason: format!("Provider '{}' does not support any extensions", self.name()),
            });
        }

        // Check for unsupported parameters
        for key in extensions.keys() {
            if !supported.contains(&key.as_str()) {
                return Err(ProviderError::InvalidExtension {
                    param: key.clone(),
                    reason: format!(
                        "Not supported by provider '{}'. Supported extensions: {}",
                        self.name(),
                        if supported.is_empty() {
                            "none".to_string()
                        } else {
                            supported.join(", ")
                        }
                    ),
                });
            }
        }

        Ok(())
    }

    /// Optional: List available models
    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        Err(ProviderError::Configuration(
            "Model listing not supported".into(),
        ))
    }

    /// Optional: Check if the provider is healthy/reachable
    async fn health_check(&self) -> Result<(), ProviderError> {
        Ok(())
    }
}

// ===== Helper Functions =====

/// Standard implementation for building CompletionResponse from InferenceResponse
/// This handles all the optional fields properly
/// If response_mode is Extended and provider_data is present, includes provider_extensions
pub fn standard_completion_response(
    response: &InferenceResponse,
    original_request: &CompletionRequest,
    provider_name: &str,
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
    let usage = if response.prompt_tokens.is_some()
        || response.completion_tokens.is_some()
        || response.total_tokens.is_some()
    {
        Some(Usage {
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
            total_tokens: response.total_tokens,
        })
    } else {
        None
    };

    // Determine if we should include provider extensions
    let provider_extensions =
        if let Some(crate::models::ResponseMode::Extended) = original_request.response_mode {
            // Include provider data if present
            response
                .provider_data
                .as_ref()
                .map(|data| crate::models::ProviderExtensions {
                    provider: provider_name.to_string(),
                    data: data.clone(),
                })
        } else {
            None
        };

    CompletionResponse {
        id: response
            .provider_request_id
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
        provider_extensions,
    }
}

/// Helper to convert stop sequences from various formats
pub fn normalize_stop_sequences(
    stop: &Option<crate::models::StringOrArray>,
) -> Option<Vec<String>> {
    stop.as_ref().map(|s| match s {
        crate::models::StringOrArray::String(s) => vec![s.clone()],
        crate::models::StringOrArray::Array(a) => a.clone(),
    })
}

// ===== Streaming Utilities =====

use std::time::{SystemTime, UNIX_EPOCH};

/// Convert text to chunked tokens for streaming
pub fn tokenize_for_streaming(text: &str) -> Vec<String> {
    // Simple word-based tokenization for now
    // In production, you might want more sophisticated tokenization
    text.split_whitespace()
        .map(|word| format!("{word} "))
        .collect()
}

/// Create a properly formatted first chunk
pub fn create_first_chunk(id: &str, model: &str, role: &str) -> crate::models::StreamChunk {
    crate::models::StreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: model.to_string(),
        choices: vec![crate::models::StreamChoice {
            index: 0,
            delta: crate::models::Delta {
                role: Some(role.to_string()),
                content: None,
                tool_calls: None,
                refusal: None,
            },
            finish_reason: None,
            logprobs: None,
        }],
        system_fingerprint: None,
        usage: None,
    }
}

/// Create a content chunk
pub fn create_content_chunk(id: &str, model: &str, content: &str) -> crate::models::StreamChunk {
    crate::models::StreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: model.to_string(),
        choices: vec![crate::models::StreamChoice {
            index: 0,
            delta: crate::models::Delta {
                role: None,
                content: Some(content.to_string()),
                tool_calls: None,
                refusal: None,
            },
            finish_reason: None,
            logprobs: None,
        }],
        system_fingerprint: None,
        usage: None,
    }
}

/// Create a properly formatted final chunk
pub fn create_final_chunk(
    id: &str,
    model: &str,
    finish_reason: &str,
    usage: Option<crate::models::Usage>,
) -> crate::models::StreamChunk {
    crate::models::StreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: model.to_string(),
        choices: vec![crate::models::StreamChoice {
            index: 0,
            delta: crate::models::Delta::default(),
            finish_reason: Some(finish_reason.to_string()),
            logprobs: None,
        }],
        system_fingerprint: None,
        usage,
    }
}
