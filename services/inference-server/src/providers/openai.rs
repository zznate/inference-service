use super::{
    InferenceProvider, InferenceRequest, InferenceResponse, ProviderError,
    standard_completion_response,
};
use crate::config::{HttpConfigSchema, Settings};
use crate::models::{CompletionRequest, CompletionResponse, StreamChunk};
use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json;
use std::sync::Arc;
use tracing::{debug, error, instrument};

pub struct OpenAIProvider {
    client: reqwest::Client,
    settings: Arc<Settings>,
}

impl OpenAIProvider {
    pub fn new(settings: Arc<Settings>) -> Result<Self, ProviderError> {
        let (api_key, organization_id) = match &settings.inference.provider {
            crate::config::InferenceProvider::OpenAI {
                api_key,
                organization_id,
            } => (api_key.clone(), organization_id.clone()),
            _ => {
                return Err(ProviderError::Configuration(
                    "Invalid provider configuration for OpenAIProvider".to_string(),
                ));
            }
        };

        // Get HTTP config (use defaults if not provided)
        let http_config = settings.inference.http.as_ref().cloned().unwrap_or({
            // Provide sane defaults for development/testing
            HttpConfigSchema {
                timeout_secs: 30,
                connect_timeout_secs: 10,
                max_retries: 3,
                retry_backoff_ms: 100,
                keep_alive_secs: Some(30),
                max_idle_connections: Some(10),
            }
        });

        // Build headers with authentication
        let mut headers = HeaderMap::new();

        // Add API key
        let auth_value = format!("Bearer {api_key}");
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth_value).map_err(|e| {
                ProviderError::Configuration(format!("Invalid API key format: {e}"))
            })?,
        );

        // Add organization ID if provided
        if let Some(ref org_id) = organization_id {
            headers.insert(
                "OpenAI-Organization",
                HeaderValue::from_str(org_id).map_err(|e| {
                    ProviderError::Configuration(format!("Invalid organization ID: {e}"))
                })?,
            );
        }

        // Always set content type
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Build HTTP client with our config and headers
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(http_config.timeout())
            .connect_timeout(http_config.connect_timeout())
            .pool_idle_timeout(http_config.keep_alive())
            .pool_max_idle_per_host(http_config.max_idle_connections.unwrap_or(10))
            .build()
            .map_err(|e| {
                ProviderError::Configuration(format!("Failed to build HTTP client: {e}"))
            })?;

        debug!(
            "Initialized OpenAI provider with base URL: {}",
            settings.inference.base_url
        );
        if organization_id.is_some() {
            debug!(
                "Using organization ID: {}",
                organization_id.as_ref().unwrap()
            );
        }

        Ok(Self { client, settings })
    }

    /// Build request body for OpenAI (already in OpenAI format)
    fn build_request_body(&self, request: &InferenceRequest) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": request.model,
            "messages": request.messages,
        });

        // Add optional parameters if present
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(freq_penalty) = request.frequency_penalty {
            body["frequency_penalty"] = serde_json::json!(freq_penalty);
        }
        if let Some(pres_penalty) = request.presence_penalty {
            body["presence_penalty"] = serde_json::json!(pres_penalty);
        }
        if let Some(ref stop) = request.stop_sequences {
            body["stop"] = serde_json::json!(stop);
        }
        if let Some(seed) = request.seed {
            body["seed"] = serde_json::json!(seed);
        }

        // Always set n=1 and stream=false for now
        body["n"] = serde_json::json!(1);
        body["stream"] = serde_json::json!(false);

        body
    }

    /// Parse OpenAI response into our internal format
    fn parse_response_body(
        &self,
        response: serde_json::Value,
    ) -> Result<InferenceResponse, ProviderError> {
        // Try to parse as CompletionResponse first (success case)
        if let Ok(completion_response) =
            serde_json::from_value::<CompletionResponse>(response.clone())
        {
            // Extract data from CompletionResponse into InferenceResponse
            let choice = completion_response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| {
                    ProviderError::InvalidResponse("No choices in response".to_string())
                })?;

            return Ok(InferenceResponse {
                text: choice
                    .message
                    .as_ref()
                    .and_then(|m| m.content.as_ref())
                    .cloned()
                    .unwrap_or_else(|| "".to_string()),
                model_used: completion_response.model,
                total_tokens: completion_response
                    .usage
                    .as_ref()
                    .and_then(|u| u.total_tokens),
                prompt_tokens: completion_response
                    .usage
                    .as_ref()
                    .and_then(|u| u.prompt_tokens),
                completion_tokens: completion_response
                    .usage
                    .as_ref()
                    .and_then(|u| u.completion_tokens),
                finish_reason: choice.finish_reason,
                latency_ms: None,
                provider_request_id: Some(completion_response.id),
                system_fingerprint: completion_response.system_fingerprint,
                tool_calls: choice.message.as_ref().and_then(|m| m.tool_calls.clone()),
                logprobs: choice.logprobs,
                provider_data: None,
            });
        }

        // Check if it's an error response
        if let Some(error) = response.get("error") {
            let error_message = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            let error_type = error
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("unknown");
            let error_code = error.get("code").and_then(|c| c.as_str());

            // Map OpenAI error types to our ProviderError types
            return match error_type {
                "insufficient_quota" | "rate_limit_exceeded" => Err(ProviderError::RequestFailed {
                    status: 429,
                    message: format!("OpenAI API error: {error_message}"),
                }),
                "model_not_found" => {
                    Err(ProviderError::ModelNotAvailable {
                        requested: self.extract_model_from_error(error_message),
                        available: vec![], // OpenAI doesn't tell us available models in error
                    })
                }
                "invalid_api_key" | "invalid_organization" => Err(ProviderError::Configuration(
                    format!("Authentication error: {error_message}"),
                )),
                _ => Err(ProviderError::RequestFailed {
                    status: 500,
                    message: format!(
                        "OpenAI API error ({}): {}",
                        error_code.unwrap_or(error_type),
                        error_message
                    ),
                }),
            };
        }

        Err(ProviderError::InvalidResponse(
            "Unexpected response format from OpenAI".to_string(),
        ))
    }

    fn extract_model_from_error(&self, error_message: &str) -> String {
        // Try to extract model name from error message
        // OpenAI errors often include the model name
        error_message
            .split_whitespace()
            .find(|word| {
                word.starts_with("gpt") || word.starts_with("text-") || word.starts_with("davinci")
            })
            .unwrap_or("unknown")
            .to_string()
    }
}

#[async_trait]
impl InferenceProvider for OpenAIProvider {
    fn build_inference_request(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> Result<InferenceRequest, ProviderError> {
        // Transform OpenAI format to our internal format (mostly pass-through)
        Ok(InferenceRequest {
            messages: request.messages.clone(),
            model: model.to_string(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            // OpenAI supports all these additional parameters
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop_sequences: super::normalize_stop_sequences(&request.stop),
            seed: request.seed,
            stream: request.stream,
            n: request.n,
            logprobs: request.logprobs,
            top_logprobs: request.top_logprobs,
        })
    }

    #[instrument(skip(self, request), fields(
        provider = "openai",
        model = %request.model,
        message_count = request.messages.len(),
    ))]
    async fn execute(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, ProviderError> {
        // Build request body
        let request_body = self.build_request_body(request);

        debug!("Sending request to OpenAI: {}", request_body);

        // Track request timing
        let start = std::time::Instant::now();

        // Execute HTTP request
        let response = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.settings.inference.base_url
            ))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                error!("Failed to send request to OpenAI: {}", e);
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
            })?;

        let latency_ms = start.elapsed().as_millis() as u64;

        // Check HTTP status
        let status = response.status();

        // Get response body as JSON regardless of status
        // OpenAI returns JSON errors even on non-200 status
        let response_body: serde_json::Value = response.json().await.map_err(|e| {
            error!("Failed to parse OpenAI response: {}", e);
            ProviderError::InvalidResponse(format!("Invalid JSON response: {e}"))
        })?;

        debug!("OpenAI response (status {}): {}", status, response_body);

        // Parse response (handles both success and error cases)
        let mut inference_response = self.parse_response_body(response_body)?;
        inference_response.latency_ms = Some(latency_ms);

        Ok(inference_response)
    }

    fn build_completion_response(
        &self,
        response: &InferenceResponse,
        original_request: &CompletionRequest,
    ) -> CompletionResponse {
        // Use the standard helper function
        standard_completion_response(response, original_request, self.name())
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn http_config(&self) -> Option<&HttpConfigSchema> {
        self.settings.inference.http.as_ref()
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        // Try to list models as a health check
        let response = self
            .client
            .get(format!("{}/models", self.settings.inference.base_url))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProviderError::Timeout
                } else if e.is_connect() {
                    ProviderError::ConnectionFailed(format!("Health check failed: {e}"))
                } else {
                    ProviderError::RequestFailed {
                        status: 0,
                        message: format!("Health check failed: {e}"),
                    }
                }
            })?;

        if response.status().is_success() {
            Ok(())
        } else if response.status() == 401 {
            Err(ProviderError::Configuration("Invalid API key".to_string()))
        } else {
            Err(ProviderError::RequestFailed {
                status: response.status().as_u16(),
                message: "Health check failed".to_string(),
            })
        }
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        #[derive(serde::Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelInfo>,
        }

        #[derive(serde::Deserialize)]
        struct ModelInfo {
            id: String,
        }

        let response = self
            .client
            .get(format!("{}/models", self.settings.inference.base_url))
            .send()
            .await
            .map_err(|e| ProviderError::ConnectionFailed(format!("Failed to list models: {e}")))?;

        if !response.status().is_success() {
            return Err(ProviderError::RequestFailed {
                status: response.status().as_u16(),
                message: "Failed to list models".to_string(),
            });
        }

        let models_response: ModelsResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::InvalidResponse(format!("Invalid models response: {e}")))?;

        // Filter to only chat models (ones that work with chat completions)
        let chat_models: Vec<String> = models_response
            .data
            .into_iter()
            .map(|m| m.id)
            .filter(|id| id.contains("gpt") || id.contains("turbo") || id.contains("davinci"))
            .collect();

        Ok(chat_models)
    }

    // ===== Streaming Support =====

    /// OpenAI provider supports native streaming
    fn supports_streaming(&self) -> bool {
        true
    }

    /// Stream completion using OpenAI's native SSE streaming
    async fn stream(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures_util::Stream<Item = Result<crate::models::StreamChunk, ProviderError>>
                    + Send,
            >,
        >,
        ProviderError,
    > {
        use eventsource_stream::Eventsource;
        use futures_util::stream::{StreamExt, TryStreamExt};

        // Reuse existing request building logic but add stream: true
        let inference_req = self.build_inference_request(request, model)?;
        let mut request_body = self.build_request_body(&inference_req);
        request_body["stream"] = serde_json::json!(true);

        debug!("Sending streaming request to OpenAI: {}", request_body);

        // Execute HTTP request
        let response = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.settings.inference.base_url
            ))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                error!("Failed to send streaming request to OpenAI: {}", e);
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
            })?;

        // Check HTTP status
        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!(
                "OpenAI streaming returned error status {}: {}",
                status, error_text
            );
            return Err(ProviderError::RequestFailed {
                status: status.as_u16(),
                message: format!("OpenAI streaming error: {error_text}"),
            });
        }

        // Parse SSE stream from OpenAI using correct API
        let bytes_stream = response
            .bytes_stream()
            .map_err(|e| std::io::Error::other(e));

        let sse_stream = bytes_stream
            .eventsource()
            .filter_map(|event_result| async move {
                match event_result {
                    Ok(event) => {
                        let data = &event.data;
                        debug!("Received SSE event type: {:?}, data: {}", event.event, data);

                        if data == "[DONE]" {
                            debug!("OpenAI stream completed with [DONE] marker");
                            None // End of stream marker
                        } else {
                            // Parse streaming chunk
                            match serde_json::from_str::<StreamChunk>(data) {
                                Ok(chunk) => {
                                    debug!(
                                        "Received OpenAI stream chunk: {:?}",
                                        chunk.choices.first().map(|c| &c.delta.content)
                                    );
                                    Some(Ok(chunk))
                                }
                                Err(e) => {
                                    error!(
                                        "Failed to parse OpenAI stream chunk: {} - Data: {}",
                                        e, data
                                    );
                                    Some(Err(ProviderError::StreamError(format!(
                                        "Invalid stream chunk: {e}"
                                    ))))
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("SSE parsing error: {}", e);
                        Some(Err(ProviderError::StreamError(format!("SSE error: {e}"))))
                    }
                }
            });

        Ok(Box::pin(sse_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{InferenceConfig, LogFormat, LogOutput, LoggingConfig, ServerConfig};
    use crate::models::Message;

    fn create_test_settings() -> Arc<Settings> {
        Arc::new(Settings {
            server: ServerConfig {
                host: "localhost".to_string(),
                port: 3000,
            },
            inference: InferenceConfig {
                base_url: "https://api.openai.com/v1".to_string(),
                default_model: "gpt-3.5-turbo".to_string(),
                allowed_models: None,
                timeout_secs: 30,
                http: Some(HttpConfigSchema::default()),
                provider: crate::config::InferenceProvider::OpenAI {
                    api_key: "test-key".to_string(),
                    organization_id: None,
                },
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: LogFormat::Pretty,
                output: LogOutput::Stdout,
                file: None,
            },
        })
    }

    #[test]
    fn test_build_request_body() {
        let provider = OpenAIProvider::new(create_test_settings()).unwrap();

        let request = InferenceRequest {
            messages: vec![Message::new("user", "Hello")],
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            frequency_penalty: Some(0.5),
            presence_penalty: None,
            stop_sequences: Some(vec!["STOP".to_string()]),
            seed: Some(42),
            stream: None,
            n: None,
            logprobs: None,
            top_logprobs: None,
        };

        let body = provider.build_request_body(&request);

        assert_eq!(body["model"], "gpt-3.5-turbo");
        assert_eq!(body["max_tokens"], 100);
        assert!((body["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
        assert!((body["top_p"].as_f64().unwrap() - 0.9).abs() < 0.001);
        assert!((body["frequency_penalty"].as_f64().unwrap() - 0.5).abs() < 0.001);
        assert_eq!(body["stop"], serde_json::json!(["STOP"]));
        assert_eq!(body["seed"], 42);
        assert_eq!(body["n"], 1);
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn test_parse_error_response() {
        let provider = OpenAIProvider::new(create_test_settings()).unwrap();

        let error_response = serde_json::json!({
            "error": {
                "message": "You exceeded your current quota",
                "type": "insufficient_quota",
                "code": "insufficient_quota"
            }
        });

        let result = provider.parse_response_body(error_response);

        assert!(result.is_err());
        match result.unwrap_err() {
            ProviderError::RequestFailed { status, message } => {
                assert_eq!(status, 429);
                assert!(message.contains("quota"));
            }
            _ => panic!("Expected RequestFailed error"),
        }
    }
}
