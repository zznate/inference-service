use super::{
    BoxFuture, HttpProviderClient, InferenceProvider, InferenceRequest, InferenceResponse,
    ProviderError, ProviderStream, standard_completion_response,
};
use crate::config::{HttpConfigSchema, Settings};
use crate::models::{CompletionRequest, CompletionResponse, StreamChunk};
use futures_util::TryStreamExt;
use serde::Deserialize;
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error};

/// LM Studio supported extension parameters
/// These are parameters beyond the standard OpenAI spec that LM Studio supports
const LM_STUDIO_EXTENSIONS: &[&str] = &[
    "top_k",          // Top-k sampling
    "min_p",          // Minimum probability threshold
    "repeat_penalty", // Repetition penalty
    "mirostat_mode",  // Mirostat sampling mode (0, 1, or 2)
    "mirostat_tau",   // Mirostat target entropy
    "mirostat_eta",   // Mirostat learning rate
    "grammar",        // Grammar-based constrained generation
    "cache_prompt",   // Whether to cache the prompt
    "n_probs",        // Number of probabilities to return (alternative to top_logprobs)
    "tfs_z",          // Tail-free sampling parameter
    "typical_p",      // Typical sampling parameter
    "min_tokens",     // Minimum number of tokens to generate
];

pub struct LMStudioProvider {
    http: HttpProviderClient,
}

impl LMStudioProvider {
    /// Validate LM Studio-specific extension parameters
    fn validate_lm_studio_extensions(
        extensions: &HashMap<String, serde_json::Value>,
    ) -> Result<(), ProviderError> {
        for (key, value) in extensions {
            match key.as_str() {
                "top_k" => {
                    if let Some(v) = value.as_i64() {
                        if v <= 0 {
                            return Err(ProviderError::InvalidExtension {
                                param: key.clone(),
                                reason: "must be > 0".to_string(),
                            });
                        }
                    } else {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be an integer".to_string(),
                        });
                    }
                }
                "min_p" => {
                    if let Some(v) = value.as_f64() {
                        if !(0.0..=1.0).contains(&v) {
                            return Err(ProviderError::InvalidExtension {
                                param: key.clone(),
                                reason: "must be between 0.0 and 1.0".to_string(),
                            });
                        }
                    } else {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be a number".to_string(),
                        });
                    }
                }
                "repeat_penalty" => {
                    if let Some(v) = value.as_f64() {
                        if v < 0.0 {
                            return Err(ProviderError::InvalidExtension {
                                param: key.clone(),
                                reason: "must be >= 0.0".to_string(),
                            });
                        }
                    } else {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be a number".to_string(),
                        });
                    }
                }
                "mirostat_mode" => {
                    if let Some(v) = value.as_i64() {
                        if !(0..=2).contains(&v) {
                            return Err(ProviderError::InvalidExtension {
                                param: key.clone(),
                                reason: "must be 0, 1, or 2".to_string(),
                            });
                        }
                    } else {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be an integer".to_string(),
                        });
                    }
                }
                "mirostat_tau" => {
                    if let Some(v) = value.as_f64() {
                        if v <= 0.0 {
                            return Err(ProviderError::InvalidExtension {
                                param: key.clone(),
                                reason: "must be > 0.0".to_string(),
                            });
                        }
                    } else {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be a number".to_string(),
                        });
                    }
                }
                "mirostat_eta" => {
                    if let Some(v) = value.as_f64() {
                        if !(0.0..=1.0).contains(&v) {
                            return Err(ProviderError::InvalidExtension {
                                param: key.clone(),
                                reason: "must be between 0.0 and 1.0".to_string(),
                            });
                        }
                    } else {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be a number".to_string(),
                        });
                    }
                }
                "typical_p" => {
                    if let Some(v) = value.as_f64() {
                        if !(0.0..=1.0).contains(&v) {
                            return Err(ProviderError::InvalidExtension {
                                param: key.clone(),
                                reason: "must be between 0.0 and 1.0".to_string(),
                            });
                        }
                    } else {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be a number".to_string(),
                        });
                    }
                }
                "tfs_z" => {
                    if let Some(v) = value.as_f64() {
                        if v < 0.0 {
                            return Err(ProviderError::InvalidExtension {
                                param: key.clone(),
                                reason: "must be >= 0.0".to_string(),
                            });
                        }
                    } else {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be a number".to_string(),
                        });
                    }
                }
                "min_tokens" => {
                    if let Some(v) = value.as_i64() {
                        if v < 0 {
                            return Err(ProviderError::InvalidExtension {
                                param: key.clone(),
                                reason: "must be >= 0".to_string(),
                            });
                        }
                    } else {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be an integer".to_string(),
                        });
                    }
                }
                "n_probs" => {
                    if let Some(v) = value.as_i64() {
                        if v < 0 {
                            return Err(ProviderError::InvalidExtension {
                                param: key.clone(),
                                reason: "must be >= 0".to_string(),
                            });
                        }
                    } else {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be an integer".to_string(),
                        });
                    }
                }
                "grammar" => {
                    // Grammar should be a string
                    if !value.is_string() {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be a string".to_string(),
                        });
                    }
                }
                "cache_prompt" => {
                    // cache_prompt should be a boolean
                    if !value.is_boolean() {
                        return Err(ProviderError::InvalidExtension {
                            param: key.clone(),
                            reason: "must be a boolean".to_string(),
                        });
                    }
                }
                _ => {
                    // This shouldn't happen if supported_extensions() is correct
                    // but good to have as a safety check
                    return Err(ProviderError::InvalidExtension {
                        param: key.clone(),
                        reason: format!(
                            "unknown parameter (supported: {})",
                            LM_STUDIO_EXTENSIONS.join(", ")
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn new(settings: Arc<Settings>) -> Result<Self, ProviderError> {
        let http = HttpProviderClient::new(
            &settings.inference.base_url,
            settings.inference.http.as_ref(),
            None,
        )?;

        Ok(Self { http })
    }

    /// Build request body for LM Studio (OpenAI-compatible format)
    /// Accepts optional extensions for provider-specific parameters
    fn build_request_body(
        &self,
        request: &InferenceRequest,
        extensions: Option<&HashMap<String, serde_json::Value>>,
    ) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens.unwrap_or(100),
            "temperature": request.temperature.unwrap_or(0.7),
        });

        // Add optional OpenAI parameters if present
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

        // Merge validated extensions if present
        if let Some(exts) = extensions {
            for (key, value) in exts {
                body[key] = value.clone();
            }
            debug!(
                "Added {} extension parameters to LM Studio request",
                exts.len()
            );
        }

        body
    }

    /// Parse LM Studio response (OpenAI format) into our internal format
    /// Extracts provider-specific extension data if present
    fn parse_response_body(
        &self,
        response: serde_json::Value,
        requested_model: &str,
    ) -> Result<InferenceResponse, ProviderError> {
        // Extract provider-specific fields before parsing into CompletionResponse
        // LM Studio may return additional fields like timings, model_info, etc.
        let mut provider_data = HashMap::new();

        // Extract known LM Studio-specific response fields
        if let Some(obj) = response.as_object() {
            // Timing information
            if let Some(timings) = obj.get("timings") {
                provider_data.insert("timings".to_string(), timings.clone());
            }
            // Model information
            if let Some(model_info) = obj.get("model_info") {
                provider_data.insert("model_info".to_string(), model_info.clone());
            }
            // Truncated flag
            if let Some(truncated) = obj.get("truncated") {
                provider_data.insert("truncated".to_string(), truncated.clone());
            }
            // Slot ID (for session management)
            if let Some(slot_id) = obj.get("slot_id") {
                provider_data.insert("slot_id".to_string(), slot_id.clone());
            }
        }

        // Parse into standard CompletionResponse
        let completion_response: CompletionResponse =
            serde_json::from_value(response).map_err(|e| {
                ProviderError::InvalidResponse(format!("Failed to parse response: {e}"))
            })?;

        // Validate that LM Studio used the requested model
        if completion_response.model != requested_model {
            error!(
                "LM Studio used different model: requested '{}', got '{}'",
                requested_model, completion_response.model
            );
            return Err(ProviderError::ModelNotAvailable {
                requested: requested_model.to_string(),
                available: vec![completion_response.model.clone()],
            });
        }

        // Extract data from CompletionResponse into InferenceResponse
        let choice = completion_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::InvalidResponse("No choices in response".to_string()))?;

        Ok(InferenceResponse {
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
            tool_calls: choice.message.and_then(|m| m.tool_calls),
            logprobs: choice.logprobs,
            provider_data: if provider_data.is_empty() {
                None
            } else {
                Some(provider_data)
            },
        })
    }
}

impl InferenceProvider for LMStudioProvider {
    fn execute(
        &self,
        request: &InferenceRequest,
    ) -> BoxFuture<'_, Result<InferenceResponse, ProviderError>> {
        let request_body = self.build_request_body(request, None);
        let model = request.model.clone();

        Box::pin(async move {
            debug!("Sending request to LM Studio: {}", request_body);
            let response_body = self.http.post_json("v1/chat/completions", &request_body).await?;
            debug!("LM Studio response: {}", response_body);
            self.parse_response_body(response_body, &model)
        })
    }

    fn build_completion_response(
        &self,
        response: &InferenceResponse,
        original_request: &CompletionRequest,
    ) -> CompletionResponse {
        standard_completion_response(response, original_request, self.name())
    }

    fn name(&self) -> &str {
        "lmstudio"
    }

    fn http_config(&self) -> Option<&HttpConfigSchema> {
        Some(self.http.http_config())
    }

    fn health_check(&self) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move {
            let response = self.http.get("v1/models").await?;
            if response.status().is_success() {
                Ok(())
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
            #[derive(Deserialize)]
            struct ModelsResponse {
                data: Vec<ModelInfo>,
            }
            #[derive(Deserialize)]
            struct ModelInfo {
                id: String,
            }

            let response = self.http.get("v1/models").await?;
            if !response.status().is_success() {
                return Err(ProviderError::RequestFailed {
                    status: response.status().as_u16(),
                    message: "Failed to list models".to_string(),
                });
            }
            let models_response: ModelsResponse = response.json().await.map_err(|e| {
                ProviderError::InvalidResponse(format!("Invalid models response: {e}"))
            })?;
            Ok(models_response.data.into_iter().map(|m| m.id).collect())
        })
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        LM_STUDIO_EXTENSIONS.to_vec()
    }

    fn validate_extensions(
        &self,
        extensions: &HashMap<String, serde_json::Value>,
    ) -> Result<(), ProviderError> {
        Self::validate_lm_studio_extensions(extensions)
    }

    fn generate(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> BoxFuture<'_, Result<CompletionResponse, ProviderError>> {
        let extensions_validated = if let Some(ref exts) = request.extensions {
            match self.validate_extensions(exts) {
                Ok(()) => Some(exts.clone()),
                Err(e) => return Box::pin(async move { Err(e) }),
            }
        } else {
            None
        };

        let inference_req = match self.build_inference_request(request, model) {
            Ok(req) => req,
            Err(e) => return Box::pin(async move { Err(e) }),
        };

        let request_body = self.build_request_body(&inference_req, extensions_validated.as_ref());
        let model = model.to_string();
        let request_clone = request.clone();

        Box::pin(async move {
            debug!("Sending request to LM Studio: {}", request_body);
            let response_body = self.http.post_json("v1/chat/completions", &request_body).await?;
            debug!("LM Studio response: {}", response_body);

            // Parse as full CompletionResponse (handles all n choices)
            if let Ok(completion_response) =
                serde_json::from_value::<CompletionResponse>(response_body.clone())
            {
                debug!(
                    "LM Studio request completed with {} choices",
                    completion_response.choices.len()
                );
                return Ok(completion_response);
            }

            // Fallback: parse as single-choice response
            let inference_resp = self.parse_response_body(response_body, &model)?;
            Ok(self.build_completion_response(&inference_resp, &request_clone))
        })
    }

    fn stream(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let extensions_validated = if let Some(ref exts) = request.extensions {
            match self.validate_extensions(exts) {
                Ok(()) => Some(exts.clone()),
                Err(e) => return Box::pin(async move { Err(e) }),
            }
        } else {
            None
        };

        let inference_req = match self.build_inference_request(request, model) {
            Ok(req) => req,
            Err(e) => return Box::pin(async move { Err(e) }),
        };

        let mut request_body =
            self.build_request_body(&inference_req, extensions_validated.as_ref());
        request_body["stream"] = serde_json::json!(true);

        Box::pin(async move {
            use eventsource_stream::Eventsource;
            use futures_util::stream::StreamExt;

            debug!("Sending streaming request to LM Studio: {}", request_body);
            let response = self.http.post_stream("v1/chat/completions", &request_body).await?;

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
    use crate::models::{FinishReason, Message, Role};

    fn create_test_settings() -> Arc<Settings> {
        Arc::new(Settings {
            server: ServerConfig {
                host: "localhost".to_string(),
                port: 3000,
            },
            inference: InferenceConfig {
                base_url: "http://localhost:1234".to_string(),
                default_model: "test-model".to_string(),
                allowed_models: None,
                timeout_secs: 30,
                http: Some(HttpConfigSchema::default()),
                provider: crate::config::InferenceProvider::LMStudio,
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
    fn test_build_inference_request() {
        let provider = LMStudioProvider::new(create_test_settings()).unwrap();

        let completion_req = CompletionRequest {
            messages: vec![Message::new(Role::User, "Hello")],
            model: Some("gpt-4".to_string()),
            max_tokens: Some(100),
            temperature: Some(0.7),
            ..Default::default()
        };

        let inference_req = provider
            .build_inference_request(&completion_req, "gpt-4")
            .unwrap();

        assert_eq!(inference_req.model, "gpt-4");
        assert_eq!(inference_req.messages.len(), 1);
        assert_eq!(inference_req.max_tokens, Some(100));
        assert_eq!(inference_req.temperature, Some(0.7));
        // Check that optional fields are None
        assert_eq!(inference_req.top_p, None);
    }

    #[test]
    fn test_parse_response_body() {
        let provider = LMStudioProvider::new(create_test_settings()).unwrap();

        // Create a response in OpenAI format (what LM Studio returns)
        let response_json = serde_json::json!({
            "id": "test-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });

        let inference_resp = provider
            .parse_response_body(response_json, "gpt-4")
            .unwrap();

        assert_eq!(inference_resp.text, "Hello there!");
        assert_eq!(inference_resp.model_used, "gpt-4");
        assert_eq!(inference_resp.total_tokens, Some(15));
        assert_eq!(inference_resp.prompt_tokens, Some(10));
        assert_eq!(inference_resp.completion_tokens, Some(5));
        assert_eq!(inference_resp.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_parse_response_wrong_model() {
        let provider = LMStudioProvider::new(create_test_settings()).unwrap();

        let response_json = serde_json::json!({
            "id": "test-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5",  // Different model than requested
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });

        let result = provider.parse_response_body(response_json, "gpt-4");

        assert!(result.is_err());
        match result.unwrap_err() {
            ProviderError::ModelNotAvailable {
                requested,
                available,
            } => {
                assert_eq!(requested, "gpt-4");
                assert_eq!(available, vec!["gpt-3.5"]);
            }
            _ => panic!("Expected ModelNotAvailable error"),
        }
    }

    #[test]
    fn test_build_completion_response() {
        let provider = LMStudioProvider::new(create_test_settings()).unwrap();

        let inference_resp = InferenceResponse {
            text: "Hello!".to_string(),
            model_used: "gpt-4".to_string(),
            total_tokens: Some(15),
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            finish_reason: Some(FinishReason::Stop),
            latency_ms: None,
            provider_request_id: None,
            system_fingerprint: None,
            tool_calls: None,
            logprobs: None,
            provider_data: None,
        };

        let original_req = CompletionRequest {
            messages: vec![],
            model: Some("gpt-4".to_string()),
            max_tokens: None,
            temperature: None,
            ..Default::default()
        };

        let completion_resp = provider.build_completion_response(&inference_resp, &original_req);

        assert_eq!(completion_resp.model, "gpt-4");
        assert_eq!(
            completion_resp.choices[0]
                .message
                .as_ref()
                .unwrap()
                .content
                .as_ref()
                .unwrap(),
            "Hello!"
        );
        assert_eq!(
            completion_resp
                .usage
                .as_ref()
                .unwrap()
                .total_tokens
                .unwrap(),
            15
        );
        assert_eq!(
            completion_resp.choices[0].finish_reason,
            Some(FinishReason::Stop)
        );
    }
}
