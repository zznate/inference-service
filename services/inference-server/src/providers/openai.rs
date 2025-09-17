use super::{InferenceProvider, InferenceRequest, InferenceResponse, ProviderError, standard_completion_response};
use crate::config::HttpConfigSchema;
use crate::models::{CompletionRequest, CompletionResponse};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json;
use tracing::{debug, error, instrument};

pub struct OpenAIProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    organization_id: Option<String>,
    http_config: HttpConfigSchema,
}

impl OpenAIProvider {
    pub fn new(
        api_key: String,
        organization_id: Option<String>,
        base_url: Option<String>,
        http_config: HttpConfigSchema,
    ) -> Result<Self, ProviderError> {
        // Build headers with authentication
        let mut headers = HeaderMap::new();
        
        // Add API key
        let auth_value = format!("Bearer {}", api_key);
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth_value)
                .map_err(|e| ProviderError::Configuration(format!("Invalid API key format: {}", e)))?
        );
        
        // Add organization ID if provided
        if let Some(ref org_id) = organization_id {
            headers.insert(
                "OpenAI-Organization",
                HeaderValue::from_str(org_id)
                    .map_err(|e| ProviderError::Configuration(format!("Invalid organization ID: {}", e)))?
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
            .map_err(|e| ProviderError::Configuration(format!("Failed to build HTTP client: {}", e)))?;
        
        // Use official OpenAI API URL unless overridden (e.g., for Azure OpenAI)
        let base_url = base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        
        info!("Initialized OpenAI provider with base URL: {}", base_url);
        if organization_id.is_some() {
            info!("Using organization ID: {}", organization_id.as_ref().unwrap());
        }
        
        Ok(Self {
            client,
            base_url,
            api_key,
            organization_id,
            http_config,
        })
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
        
        // Add any provider-specific parameters
        if let Some(ref params) = request.provider_params {
            if let Some(obj) = params.as_object() {
                for (key, value) in obj {
                    // Only add OpenAI-specific parameters we haven't already handled
                    if !["model", "messages", "max_tokens", "temperature", "top_p", 
                         "frequency_penalty", "presence_penalty", "stop", "seed"].contains(&key.as_str()) {
                        body[key] = value.clone();
                    }
                }
            }
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
        if let Ok(completion_response) = serde_json::from_value::<CompletionResponse>(response.clone()) {
            // Extract data from CompletionResponse into InferenceResponse
            let choice = completion_response.choices
                .into_iter()
                .next()
                .ok_or_else(|| ProviderError::InvalidResponse("No choices in response".to_string()))?;
            
            return Ok(InferenceResponse {
                text: choice.message.content,
                model_used: completion_response.model,
                total_tokens: Some(completion_response.usage.total_tokens),
                prompt_tokens: Some(completion_response.usage.prompt_tokens),
                completion_tokens: Some(completion_response.usage.completion_tokens),
                finish_reason: Some(choice.finish_reason),
                latency_ms: None,
                provider_request_id: Some(completion_response.id),
                provider_metadata: None,
            });
        }
        
        // Check if it's an error response
        if let Some(error) = response.get("error") {
            let error_message = error.get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            let error_type = error.get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("unknown");
            let error_code = error.get("code")
                .and_then(|c| c.as_str());
            
            // Map OpenAI error types to our ProviderError types
            return match error_type {
                "insufficient_quota" | "rate_limit_exceeded" => {
                    Err(ProviderError::RequestFailed {
                        status: 429,
                        message: format!("OpenAI API error: {}", error_message),
                    })
                },
                "model_not_found" => {
                    Err(ProviderError::ModelNotAvailable {
                        requested: self.extract_model_from_error(error_message),
                        available: vec![], // OpenAI doesn't tell us available models in error
                    })
                },
                "invalid_api_key" | "invalid_organization" => {
                    Err(ProviderError::Configuration(format!("Authentication error: {}", error_message)))
                },
                _ => {
                    Err(ProviderError::RequestFailed {
                        status: 500,
                        message: format!("OpenAI API error ({}): {}", 
                            error_code.unwrap_or(error_type), error_message),
                    })
                }
            };
        }
        
        Err(ProviderError::InvalidResponse("Unexpected response format from OpenAI".to_string()))
    }
    
    fn extract_model_from_error(&self, error_message: &str) -> String {
        // Try to extract model name from error message
        // OpenAI errors often include the model name
        error_message
            .split_whitespace()
            .find(|word| word.starts_with("gpt") || word.starts_with("text-") || word.starts_with("davinci"))
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
            // OpenAI supports these additional parameters
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
            provider_params: None,
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
        let response = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                error!("Failed to send request to OpenAI: {}", e);
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
        
        let latency_ms = start.elapsed().as_millis() as u64;
        
        // Check HTTP status
        let status = response.status();
        
        // Get response body as JSON regardless of status
        // OpenAI returns JSON errors even on non-200 status
        let response_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| {
                error!("Failed to parse OpenAI response: {}", e);
                ProviderError::InvalidResponse(format!("Invalid JSON response: {}", e))
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
        standard_completion_response(response, original_request)
    }
    
    fn name(&self) -> &str {
        "openai"
    }
    
    fn http_config(&self) -> Option<&HttpConfigSchema> {
        Some(&self.http_config)
    }
    
    async fn health_check(&self) -> Result<(), ProviderError> {
        // Try to list models as a health check
        let response = self.client
            .get(format!("{}/models", self.base_url))
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
        
        let response = self.client
            .get(format!("{}/models", self.base_url))
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
        
        // Filter to only chat models (ones that work with chat completions)
        let chat_models: Vec<String> = models_response.data
            .into_iter()
            .map(|m| m.id)
            .filter(|id| id.contains("gpt") || id.contains("turbo") || id.contains("davinci"))
            .collect();
        
        Ok(chat_models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Message;
    
    #[test]
    fn test_build_request_body() {
        let provider = OpenAIProvider::new(
            "test-key".to_string(),
            None,
            None,
            HttpConfigSchema::default(),
        ).unwrap();
        
        let request = InferenceRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            frequency_penalty: Some(0.5),
            presence_penalty: None,
            stop_sequences: Some(vec!["STOP".to_string()]),
            seed: Some(42),
            provider_params: None,
        };
        
        let body = provider.build_request_body(&request);
        
        assert_eq!(body["model"], "gpt-3.5-turbo");
        assert_eq!(body["max_tokens"], 100);
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["frequency_penalty"], 0.5);
        assert_eq!(body["stop"], serde_json::json!(["STOP"]));
        assert_eq!(body["seed"], 42);
        assert_eq!(body["n"], 1);
        assert_eq!(body["stream"], false);
    }
    
    #[test]
    fn test_parse_error_response() {
        let provider = OpenAIProvider::new(
            "test-key".to_string(),
            None,
            None,
            HttpConfigSchema::default(),
        ).unwrap();
        
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

use tracing::info;