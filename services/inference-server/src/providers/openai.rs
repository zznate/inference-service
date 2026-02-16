use super::{
    BoxFuture, HttpProviderClient, InferenceProvider, InferenceRequest, InferenceResponse,
    ProviderError, ProviderStream, standard_completion_response,
};
use crate::config::{HttpConfigSchema, Settings};
use crate::models::{CompletionRequest, CompletionResponse, StreamChunk};
use futures_util::TryStreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json;
use std::sync::Arc;
use tracing::debug;

pub struct OpenAIProvider {
    http: HttpProviderClient,
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

        // Build headers with authentication
        let mut headers = HeaderMap::new();
        let auth_value = format!("Bearer {api_key}");
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth_value).map_err(|e| {
                ProviderError::Configuration(format!("Invalid API key format: {e}"))
            })?,
        );
        if let Some(ref org_id) = organization_id {
            headers.insert(
                "OpenAI-Organization",
                HeaderValue::from_str(org_id).map_err(|e| {
                    ProviderError::Configuration(format!("Invalid organization ID: {e}"))
                })?,
            );
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let http = HttpProviderClient::new(
            &settings.inference.base_url,
            settings.inference.http.as_ref(),
            Some(headers),
        )?;

        debug!(
            "Initialized OpenAI provider with base URL: {}",
            settings.inference.base_url
        );

        Ok(Self { http })
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
        if let Some(ref user) = request.user {
            body["user"] = serde_json::json!(user);
        }
        if let Some(n) = request.n {
            body["n"] = serde_json::json!(n);
        }
        if let Some(ref response_format) = request.response_format {
            body["response_format"] = serde_json::json!(response_format);
        }
        if let Some(ref logit_bias) = request.logit_bias {
            body["logit_bias"] = serde_json::json!(logit_bias);
        }
        if let Some(logprobs) = request.logprobs {
            body["logprobs"] = serde_json::json!(logprobs);
        }
        if let Some(top_logprobs) = request.top_logprobs {
            body["top_logprobs"] = serde_json::json!(top_logprobs);
        }

        // Always set stream=false for now (streaming handled separately)
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

impl InferenceProvider for OpenAIProvider {
    fn execute(
        &self,
        request: &InferenceRequest,
    ) -> BoxFuture<'_, Result<InferenceResponse, ProviderError>> {
        let request_body = self.build_request_body(request);

        Box::pin(async move {
            debug!("Sending request to OpenAI: {}", request_body);
            let start = std::time::Instant::now();

            // OpenAI returns JSON errors even on non-200 status, so we use post_json with retry
            let response_body = self.http.post_json("chat/completions", &request_body).await?;

            let latency_ms = start.elapsed().as_millis() as u64;
            debug!("OpenAI response: {}", response_body);

            let mut inference_response = self.parse_response_body(response_body)?;
            inference_response.latency_ms = Some(latency_ms);
            Ok(inference_response)
        })
    }

    fn build_completion_response(
        &self,
        response: &InferenceResponse,
        original_request: &CompletionRequest,
    ) -> CompletionResponse {
        standard_completion_response(response, original_request, self.name())
    }

    /// Override generate to properly handle n > 1 completions
    fn generate(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> BoxFuture<'_, Result<CompletionResponse, ProviderError>> {
        let inference_req = match self.build_inference_request(request, model) {
            Ok(req) => req,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let request_body = self.build_request_body(&inference_req);

        Box::pin(async move {
            debug!("Sending request to OpenAI: {}", request_body);
            let start = std::time::Instant::now();
            let response_body = self.http.post_json("chat/completions", &request_body).await?;
            let latency_ms = start.elapsed().as_millis() as u64;
            debug!("OpenAI response: {}", response_body);

            // Parse as full CompletionResponse (handles all n choices)
            if let Ok(completion_response) =
                serde_json::from_value::<CompletionResponse>(response_body.clone())
            {
                debug!(
                    "OpenAI request completed in {}ms with {} choices",
                    latency_ms,
                    completion_response.choices.len()
                );
                return Ok(completion_response);
            }

            // Check if it's an error response
            if let Some(error) = response_body.get("error") {
                let error_message = error
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown error");
                let error_type = error
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("unknown");
                let error_code = error.get("code").and_then(|c| c.as_str());

                return match error_type {
                    "insufficient_quota" | "rate_limit_exceeded" => {
                        Err(ProviderError::RequestFailed {
                            status: 429,
                            message: format!("OpenAI API error: {error_message}"),
                        })
                    }
                    "model_not_found" => Err(ProviderError::ModelNotAvailable {
                        requested: self.extract_model_from_error(error_message),
                        available: vec![],
                    }),
                    "invalid_api_key" | "invalid_organization" => {
                        Err(ProviderError::Configuration(format!(
                            "Authentication error: {error_message}"
                        )))
                    }
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
        })
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn http_config(&self) -> Option<&HttpConfigSchema> {
        Some(self.http.http_config())
    }

    fn health_check(&self) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move {
            let response = self.http.get("models").await?;
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
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>, ProviderError>> {
        Box::pin(async move {
            #[derive(serde::Deserialize)]
            struct ModelsResponse {
                data: Vec<ModelInfo>,
            }
            #[derive(serde::Deserialize)]
            struct ModelInfo {
                id: String,
            }

            let response = self.http.get("models").await?;
            if !response.status().is_success() {
                return Err(ProviderError::RequestFailed {
                    status: response.status().as_u16(),
                    message: "Failed to list models".to_string(),
                });
            }
            let models_response: ModelsResponse = response.json().await.map_err(|e| {
                ProviderError::InvalidResponse(format!("Invalid models response: {e}"))
            })?;
            Ok(models_response
                .data
                .into_iter()
                .map(|m| m.id)
                .filter(|id| {
                    id.contains("gpt") || id.contains("turbo") || id.contains("davinci")
                })
                .collect())
        })
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn stream(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let inference_req = match self.build_inference_request(request, model) {
            Ok(req) => req,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let mut request_body = self.build_request_body(&inference_req);
        request_body["stream"] = serde_json::json!(true);

        Box::pin(async move {
            use eventsource_stream::Eventsource;
            use futures_util::stream::StreamExt;

            debug!("Sending streaming request to OpenAI: {}", request_body);
            let response = self.http.post_stream("chat/completions", &request_body).await?;

            let bytes_stream = response.bytes_stream().map_err(std::io::Error::other);
            let sse_stream = bytes_stream
                .eventsource()
                .filter_map(|event_result| async move {
                    match event_result {
                        Ok(event) => {
                            let data = &event.data;
                            if data == "[DONE]" {
                                None
                            } else {
                                match serde_json::from_str::<StreamChunk>(data) {
                                    Ok(chunk) => Some(Ok(chunk)),
                                    Err(e) => Some(Err(ProviderError::StreamError(format!(
                                        "Invalid stream chunk: {e}"
                                    )))),
                                }
                            }
                        }
                        Err(e) => Some(Err(ProviderError::StreamError(format!("SSE error: {e}")))),
                    }
                });

            Ok(Box::pin(sse_stream) as ProviderStream)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{InferenceConfig, LogFormat, LogOutput, LoggingConfig, ServerConfig};
    use crate::models::{Message, Role};

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
            messages: vec![Message::new(Role::User, "Hello")],
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
            user: None,
            response_format: None,
            logit_bias: None,
        };

        let body = provider.build_request_body(&request);

        assert_eq!(body["model"], "gpt-3.5-turbo");
        assert_eq!(body["max_tokens"], 100);
        assert!((body["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
        assert!((body["top_p"].as_f64().unwrap() - 0.9).abs() < 0.001);
        assert!((body["frequency_penalty"].as_f64().unwrap() - 0.5).abs() < 0.001);
        assert_eq!(body["stop"], serde_json::json!(["STOP"]));
        assert_eq!(body["seed"], 42);
        // n should not be set if not provided in request
        assert!(body.get("n").is_none());
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

    #[test]
    fn test_build_request_body_with_n_completions() {
        let provider = OpenAIProvider::new(create_test_settings()).unwrap();

        let request = InferenceRequest {
            messages: vec![Message::new(Role::User, "Hello")],
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            n: Some(3), // Request 3 completions
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
            stream: None,
            logprobs: None,
            top_logprobs: None,
            user: None,
            response_format: None,
            logit_bias: None,
        };

        let body = provider.build_request_body(&request);

        assert_eq!(body["model"], "gpt-3.5-turbo");
        assert_eq!(body["max_tokens"], 100);
        assert_eq!(body["n"], 3); // Verify n is set to 3
        assert!((body["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_build_request_body_with_logprobs() {
        let provider = OpenAIProvider::new(create_test_settings()).unwrap();

        let request = InferenceRequest {
            messages: vec![Message::new(Role::User, "Hello")],
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            logprobs: Some(true),
            top_logprobs: Some(5),
            n: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
            stream: None,
            user: None,
            response_format: None,
            logit_bias: None,
        };

        let body = provider.build_request_body(&request);

        assert_eq!(body["model"], "gpt-3.5-turbo");
        assert_eq!(body["logprobs"], true);
        assert_eq!(body["top_logprobs"], 5);
    }
}
