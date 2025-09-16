use super::{InferenceProvider, InferenceRequest, InferenceResponse, ProviderError, standard_completion_response};
use crate::config::HttpConfigSchema;
use crate::models::{CompletionRequest, CompletionResponse};
use async_trait::async_trait;
use reqwest;
use serde_json;
use serde::Deserialize;
use tracing::{debug, error, instrument};

pub struct LMStudioProvider {
    client: reqwest::Client,
    base_url: String,
    http_config: HttpConfigSchema,
}

impl LMStudioProvider {
    pub fn new(base_url: String, http_config: HttpConfigSchema) -> Result<Self, ProviderError> {
        // Build HTTP client with our config
        let client = reqwest::Client::builder()
            .timeout(http_config.timeout())
            .connect_timeout(http_config.connect_timeout())
            .pool_idle_timeout(http_config.keep_alive())
            .pool_max_idle_per_host(http_config.max_idle_connections.unwrap_or(10))
            .build()
            .map_err(|e| ProviderError::Configuration(format!("Failed to build HTTP client: {}", e)))?;
        
        Ok(Self {
            client,
            base_url,
            http_config,
        })
    }
    
    /// Build request body for LM Studio (OpenAI-compatible format)
    fn build_request_body(&self, request: &InferenceRequest) -> serde_json::Value {
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
        
        // If there are provider-specific params, merge them in
        // This allows passing through any LM Studio specific parameters
        if let Some(ref params) = request.provider_params {
            if let Some(obj) = params.as_object() {
                for (key, value) in obj {
                    body[key] = value.clone();
                }
            }
        }
        
        body
    }
    
    /// Parse LM Studio response (OpenAI format) into our internal format
    fn parse_response_body(
        &self,
        response: serde_json::Value,
        requested_model: &str,
    ) -> Result<InferenceResponse, ProviderError> {
        // Since LM Studio uses OpenAI format, we can parse directly into CompletionResponse
        let completion_response: CompletionResponse = serde_json::from_value(response)
            .map_err(|e| ProviderError::InvalidResponse(format!("Failed to parse response: {}", e)))?;
        
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
        let choice = completion_response.choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::InvalidResponse("No choices in response".to_string()))?;
        
        Ok(InferenceResponse {
            text: choice.message.content,
            model_used: completion_response.model,
            total_tokens: Some(completion_response.usage.total_tokens),
            prompt_tokens: Some(completion_response.usage.prompt_tokens),
            completion_tokens: Some(completion_response.usage.completion_tokens),
            finish_reason: Some(choice.finish_reason),
            latency_ms: None,  // Could track this if we measure request time
            provider_request_id: completion_response.id.into(),
            provider_metadata: None,  // LM Studio doesn't provide extra metadata
        })
    }
}

#[async_trait]
impl InferenceProvider for LMStudioProvider {
    #[instrument(skip(self, request), fields(
        provider = "lmstudio",
        model = %request.model.as_deref().unwrap_or("none"),
    ))]
    fn build_inference_request(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> Result<InferenceRequest, ProviderError> {
        // Transform OpenAI format to our internal format
        // This is straightforward since our internal format is similar
        Ok(InferenceRequest {
            messages: request.messages.clone(),
            model: model.to_string(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            // Initialize the new fields with None/defaults
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
            provider_params: None,
        })
    }
    
    #[instrument(skip(self, request), fields(
        provider = "lmstudio",
        model = %request.model,
        message_count = request.messages.len(),
    ))]
    async fn execute(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, ProviderError> {
        // Build request body using OpenAI format (since LM Studio is OpenAI-compatible)
        let request_body = self.build_request_body(request);
        
        debug!("Sending request to LM Studio: {}", request_body);
        
        // Execute HTTP request
        let response = self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                error!("Failed to send request to LM Studio: {}", e);
                if e.is_timeout() {
                    ProviderError::Timeout
                } else if e.is_connect() {
                    ProviderError::ConnectionFailed(format!("Connection failed: {}", e))
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
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            error!("LM Studio returned error status {}: {}", status, error_text);
            return Err(ProviderError::RequestFailed {
                status: status.as_u16(),
                message: format!("LM Studio error: {}", error_text),
            });
        }
        
        // Parse response as JSON
        let response_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| {
                error!("Failed to parse LM Studio response: {}", e);
                ProviderError::InvalidResponse(format!("Invalid JSON response: {}", e))
            })?;
        
        debug!("LM Studio response: {}", response_body);
        
        // Parse into our internal format
        self.parse_response_body(response_body, &request.model)
    }
    
    fn build_completion_response(
        &self,
        response: &InferenceResponse,
        original_request: &CompletionRequest,
    ) -> CompletionResponse {
        // Use the standard helper function
        standard_completion_response(response, original_request)
    }
    
    fn name(&self) -> &str {
        "lmstudio"
    }
    
    fn http_config(&self) -> Option<&HttpConfigSchema> {
        Some(&self.http_config)
    }
    
    async fn health_check(&self) -> Result<(), ProviderError> {
        // Try to get models list as a health check
        let response = self.client
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProviderError::Timeout
                } else if e.is_connect() {
                    ProviderError::ConnectionFailed(format!("Health check failed: {}", e))
                } else {
                    ProviderError::RequestFailed {
                        status: 0,
                        message: format!("Health check failed: {}", e),
                    }
                }
            })?;
        
        if response.status().is_success() {
            Ok(())
        } else {
            Err(ProviderError::RequestFailed {
                status: response.status().as_u16(),
                message: "Health check failed".to_string(),
            })
        }
    }
    
    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelInfo>,
        }
        
        #[derive(Deserialize)]
        struct ModelInfo {
            id: String,
        }
        
        let response = self.client
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await
            .map_err(|e| ProviderError::ConnectionFailed(format!("Failed to list models: {}", e)))?;
        
        if !response.status().is_success() {
            return Err(ProviderError::RequestFailed {
                status: response.status().as_u16(),
                message: "Failed to list models".to_string(),
            });
        }
        
        let models_response: ModelsResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::InvalidResponse(format!("Invalid models response: {}", e)))?;
        
        Ok(models_response.data.into_iter().map(|m| m.id).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::models::Message;
    use super::*;
    
    #[test]
    fn test_build_inference_request() {
        let provider = LMStudioProvider::new(
            "http://localhost:1234".to_string(),
            HttpConfigSchema::default(),
        ).unwrap();
        
        let completion_req = CompletionRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            model: Some("gpt-4".to_string()),
            max_tokens: Some(100),
            temperature: Some(0.7),
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
        assert_eq!(inference_req.provider_params, None);
    }
    
    #[test]
    fn test_parse_response_body() {
        let provider = LMStudioProvider::new(
            "http://localhost:1234".to_string(),
            HttpConfigSchema::default(),
        ).unwrap();
        
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
        assert_eq!(inference_resp.finish_reason, Some("stop".to_string()));
    }
    
    #[test]
    fn test_parse_response_wrong_model() {
        let provider = LMStudioProvider::new(
            "http://localhost:1234".to_string(),
            HttpConfigSchema::default(),
        ).unwrap();
        
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
            ProviderError::ModelNotAvailable { requested, available } => {
                assert_eq!(requested, "gpt-4");
                assert_eq!(available, vec!["gpt-3.5"]);
            }
            _ => panic!("Expected ModelNotAvailable error"),
        }
    }
    
    #[test]
    fn test_build_completion_response() {
        let provider = LMStudioProvider::new(
            "http://localhost:1234".to_string(),
            HttpConfigSchema::default(),
        ).unwrap();
        
        let inference_resp = InferenceResponse {
            text: "Hello!".to_string(),
            model_used: "gpt-4".to_string(),
            total_tokens: Some(15),
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            finish_reason: Some("stop".to_string()),
            latency_ms: None,
            provider_request_id: None,
            provider_metadata: None,
        };
        
        let original_req = CompletionRequest {
            messages: vec![],
            model: Some("gpt-4".to_string()),
            max_tokens: None,
            temperature: None,
        };
        
        let completion_resp = provider.build_completion_response(&inference_resp, &original_req);
        
        assert_eq!(completion_resp.model, "gpt-4");
        assert_eq!(completion_resp.choices[0].message.content, "Hello!");
        assert_eq!(completion_resp.usage.total_tokens, 15);
        assert_eq!(completion_resp.choices[0].finish_reason, "stop");
    }
}