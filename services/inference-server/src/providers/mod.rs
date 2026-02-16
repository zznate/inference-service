use crate::config::HttpConfigSchema;
use crate::models::{Choice, CompletionRequest, CompletionResponse, FinishReason, Message, Role, Usage};
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use tracing::{debug, error};
use uuid::Uuid;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub mod lmstudio;
pub mod mock;
pub mod openai;

// ===== HttpProviderClient =====

/// Shared HTTP client for providers that communicate over HTTP.
/// Consolidates HTTP client construction, URL building, error mapping, and retry logic.
pub struct HttpProviderClient {
    client: reqwest::Client,
    base_url: url::Url,
    http_config: HttpConfigSchema,
}

impl HttpProviderClient {
    /// Create a new HTTP provider client.
    /// `default_headers` can include auth headers (e.g., OpenAI Bearer token).
    pub fn new(
        base_url: &str,
        http_config: Option<&HttpConfigSchema>,
        default_headers: Option<reqwest::header::HeaderMap>,
    ) -> Result<Self, ProviderError> {
        let parsed_url = url::Url::parse(base_url).map_err(|e| {
            ProviderError::Configuration(format!("Invalid base URL '{base_url}': {e}"))
        })?;

        let config = http_config.cloned().unwrap_or_default();

        let mut builder = reqwest::Client::builder()
            .timeout(config.timeout())
            .connect_timeout(config.connect_timeout())
            .pool_idle_timeout(config.keep_alive())
            .pool_max_idle_per_host(config.max_idle_connections.unwrap_or(10));

        if let Some(headers) = default_headers {
            builder = builder.default_headers(headers);
        }

        let client = builder.build().map_err(|e| {
            ProviderError::Configuration(format!("Failed to build HTTP client: {e}"))
        })?;

        Ok(Self {
            client,
            base_url: parsed_url,
            http_config: config,
        })
    }

    /// Build a full URL by joining a path onto the base URL.
    pub fn url(&self, path: &str) -> String {
        // Use simple string concatenation since base_url may or may not have trailing slash
        let base = self.base_url.as_str().trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// Get a reference to the HTTP config.
    pub fn http_config(&self) -> &HttpConfigSchema {
        &self.http_config
    }

    /// Map a reqwest error to a ProviderError.
    pub fn map_reqwest_error(e: &reqwest::Error) -> ProviderError {
        if e.is_timeout() {
            ProviderError::Timeout
        } else if e.is_connect() {
            ProviderError::ConnectionFailed(format!("Connection failed: {e}"))
        } else {
            ProviderError::RequestFailed {
                status: 0,
                message: e.to_string(),
            }
        }
    }

    /// Send a POST request with JSON body and return the parsed JSON response.
    /// Includes exponential backoff retry on timeout and connection failures.
    pub async fn post_json(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ProviderError> {
        let url = self.url(path);
        let max_retries = self.http_config.max_retries;
        let backoff_ms = self.http_config.retry_backoff_ms;

        let mut last_error = None;

        for attempt in 0..=max_retries {
            if attempt > 0 {
                let delay = backoff_ms * 2u64.pow(attempt - 1);
                debug!("Retrying request (attempt {}/{}) after {}ms", attempt + 1, max_retries + 1, delay);
                tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
            }

            match self.client.post(&url).json(body).send().await {
                Ok(response) => {
                    let status = response.status();
                    if !status.is_success() {
                        let error_text = response
                            .text()
                            .await
                            .unwrap_or_else(|_| "Unknown error".to_string());
                        // Don't retry on 4xx errors
                        if status.is_client_error() {
                            return Err(ProviderError::RequestFailed {
                                status: status.as_u16(),
                                message: error_text,
                            });
                        }
                        // Retry on 5xx errors
                        last_error = Some(ProviderError::RequestFailed {
                            status: status.as_u16(),
                            message: error_text,
                        });
                        continue;
                    }
                    return response.json().await.map_err(|e| {
                        ProviderError::InvalidResponse(format!("Invalid JSON response: {e}"))
                    });
                }
                Err(e) => {
                    let provider_err = Self::map_reqwest_error(&e);
                    match provider_err {
                        ProviderError::Timeout | ProviderError::ConnectionFailed(_) => {
                            error!("Request failed (attempt {}/{}): {}", attempt + 1, max_retries + 1, e);
                            last_error = Some(provider_err);
                            continue;
                        }
                        _ => return Err(provider_err),
                    }
                }
            }
        }

        Err(last_error.unwrap_or(ProviderError::ConnectionFailed(
            "All retry attempts exhausted".to_string(),
        )))
    }

    /// Send a GET request and return the response.
    pub async fn get(&self, path: &str) -> Result<reqwest::Response, ProviderError> {
        let url = self.url(path);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Self::map_reqwest_error(&e))?;
        Ok(response)
    }

    /// Send a POST request for SSE streaming and return the raw response.
    /// Does NOT retry - streaming requests should not be retried.
    pub async fn post_stream(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response, ProviderError> {
        let url = self.url(path);
        let response = self
            .client
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| {
                error!("Failed to send streaming request: {}", e);
                Self::map_reqwest_error(&e)
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProviderError::RequestFailed {
                status: status.as_u16(),
                message: error_text,
            });
        }

        Ok(response)
    }
}

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
    pub n: Option<u32>, // Number of completions to generate
    #[allow(dead_code)] // TODO: Implement logprobs support
    pub logprobs: Option<bool>,
    #[allow(dead_code)] // TODO: Implement logprobs support
    pub top_logprobs: Option<u8>,

    // User tracking
    pub user: Option<String>, // Unique identifier for end-user

    // Response formatting
    pub response_format: Option<crate::models::ResponseFormat>, // JSON mode control

    // Token biasing
    pub logit_bias: Option<serde_json::Map<String, serde_json::Value>>, // Token ID to bias value (-100 to 100)
}

/// Normalized response format that all providers return
#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    // Core response fields
    pub text: String,
    pub model_used: String,
    pub finish_reason: Option<FinishReason>,

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

/// Type alias for the stream type returned by providers
pub type ProviderStream =
    Pin<Box<dyn futures_util::Stream<Item = Result<crate::models::StreamChunk, ProviderError>> + Send>>;

/// Common trait that all inference providers must implement
pub trait InferenceProvider: Send + Sync {
    /// Transform OpenAI-format request to our internal normalized format
    fn build_inference_request(
        &self,
        request: &CompletionRequest,
        model: &str, // Model already determined by validation layer
    ) -> Result<InferenceRequest, ProviderError> {
        Ok(InferenceRequest {
            messages: request.messages.clone(),
            model: model.to_string(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop_sequences: normalize_stop_sequences(&request.stop),
            seed: request.seed,
            stream: request.stream,
            n: request.n,
            logprobs: request.logprobs,
            top_logprobs: request.top_logprobs,
            user: request.user.clone(),
            response_format: request.response_format.clone(),
            logit_bias: request.logit_bias.clone(),
        })
    }

    /// Execute the inference request against the provider
    fn execute(
        &self,
        request: &InferenceRequest,
    ) -> BoxFuture<'_, Result<InferenceResponse, ProviderError>>;

    /// Transform our internal response format to OpenAI-compatible format
    fn build_completion_response(
        &self,
        response: &InferenceResponse,
        original_request: &CompletionRequest,
    ) -> CompletionResponse;

    /// High-level method that orchestrates the full flow
    fn generate(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> BoxFuture<'_, Result<CompletionResponse, ProviderError>> {
        let inference_req = match self.build_inference_request(request, model) {
            Ok(req) => req,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let request_clone = request.clone();
        Box::pin(async move {
            let inference_resp = self.execute(&inference_req).await?;
            Ok(self.build_completion_response(&inference_resp, &request_clone))
        })
    }

    /// Stream completion tokens as they're generated
    /// Default implementation returns streaming not supported
    fn stream(
        &self,
        _request: &CompletionRequest,
        _model: &str,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        Box::pin(async { Err(ProviderError::StreamingNotSupported) })
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
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>, ProviderError>> {
        Box::pin(async {
            Err(ProviderError::Configuration(
                "Model listing not supported".into(),
            ))
        })
    }

    /// Optional: Check if the provider is healthy/reachable
    fn health_check(&self) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async { Ok(()) })
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
    let mut message = Message::new(Role::Assistant, &response.text);

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
pub fn create_first_chunk(id: &str, model: &str, role: Role) -> crate::models::StreamChunk {
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
                role: Some(role),
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
    finish_reason: FinishReason,
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
            finish_reason: Some(finish_reason),
            logprobs: None,
        }],
        system_fingerprint: None,
        usage,
    }
}
