use super::{InferenceProvider, InferenceResponse, ProviderError};
use crate::config::HttpConfigSchema;
use crate::models::Message;
use async_trait::async_trait;
use reqwest;
use serde_json;
use tracing::{debug, error};

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
}

#[async_trait]
impl InferenceProvider for LMStudioProvider {
    async fn generate(
        &self,
        messages: &[Message],
        model: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<InferenceResponse, ProviderError> {
        let request_body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens.unwrap_or(100),
            "temperature": temperature.unwrap_or(0.7),
        });

        debug!("Sending request to LM Studio: {}", request_body);

        let response = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                error!("Failed to send request to LM Studio: {}", e);
                ProviderError::ConnectionFailed(e.to_string())
            })?;

        if !response.status().is_success() {
            error!("LM Studio returned error status: {}", response.status());
            return Err(ProviderError::RequestFailed {
                status: response.status().as_u16(),
                message: format!("LM Studio returned error status: {}", response.status()),
            });
        }

        let lm_response: serde_json::Value = response
            .json()
            .await
            .map_err(|e| {
                error!("Failed to parse LM Studio response: {}", e);
                ProviderError::InvalidResponse(e.to_string())
            })?;

        debug!("LM Studio response: {}", lm_response);
        
        // Extract the response fields
        let text = lm_response["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| ProviderError::InvalidResponse("Missing content field".to_string()))?
            .to_string();
        
        let total_tokens = lm_response["usage"]["total_tokens"]
            .as_u64()
            .map(|t| t as u32);
        
        let prompt_tokens = lm_response["usage"]["prompt_tokens"]
            .as_u64()
            .map(|t| t as u32);
            
        let completion_tokens = lm_response["usage"]["completion_tokens"]
            .as_u64()
            .map(|t| t as u32);
        
        // Check if model matches what was requested
        let actual_model = lm_response["model"].as_str();
        if let Some(actual) = actual_model {
            if actual != model {
                error!(
                    "LM Studio used different model: requested '{}', got '{}'", 
                    model, actual
                );
                return Err(ProviderError::ModelNotAvailable {
                    requested: model.to_string(),
                    available: vec![actual.to_string()],
                });
            }
        }

        Ok(InferenceResponse {
            text,
            model_used: model.to_string(),
            total_tokens,
            prompt_tokens,
            completion_tokens,
        })
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
            .get(format!("{}/models", self.base_url))
            .send()
            .await
            .map_err(|e| ProviderError::ConnectionFailed(e.to_string()))?;
        
        if response.status().is_success() {
            Ok(())
        } else {
            Err(ProviderError::RequestFailed {
                status: response.status().as_u16(),
                message: "Health check failed".to_string(),
            })
        }
    }
}