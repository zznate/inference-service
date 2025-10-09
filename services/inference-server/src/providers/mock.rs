use super::{
    InferenceProvider, InferenceRequest, InferenceResponse, ProviderError,
    standard_completion_response,
};
use crate::config::Settings;
use crate::models::{CompletionRequest, CompletionResponse};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

/// Mock provider for deterministic testing
pub struct MockProvider {
    settings: Arc<Settings>,
    // Cache loaded responses to avoid repeated file I/O
    response_cache: Arc<Mutex<HashMap<String, MockResponseFile>>>,
}

/// Structure of a mock response YAML file
#[derive(Debug, Clone, Deserialize, Serialize)]
struct MockResponseFile {
    responses: Vec<MockResponse>,
    #[serde(default)]
    settings: MockSettings,
}

/// Individual mock response
#[derive(Debug, Clone, Deserialize, Serialize)]
struct MockResponse {
    text: String,
    #[serde(default = "default_model")]
    model_used: String,
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
    #[serde(default = "default_finish_reason")]
    finish_reason: String,
    // Optional: simulate latency
    #[serde(default)]
    delay_ms: Option<u64>,
    // Additional OpenAI response fields
    #[serde(default)]
    system_fingerprint: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<crate::models::ToolCall>>,
    #[serde(default)]
    function_call: Option<crate::models::FunctionCall>,
    #[serde(default)]
    logprobs: Option<crate::models::LogProbs>,
}

/// Settings for how to serve responses
#[derive(Debug, Clone, Deserialize, Serialize)]
struct MockSettings {
    #[serde(default = "default_mode")]
    mode: ResponseMode,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum ResponseMode {
    First,      // Always return first response
    Sequential, // Cycle through responses
    Random,     // Random selection
}

fn default_model() -> String {
    "mock-model".to_string()
}

fn default_finish_reason() -> String {
    "stop".to_string()
}

fn default_mode() -> ResponseMode {
    ResponseMode::First
}

impl Default for MockSettings {
    fn default() -> Self {
        Self {
            mode: default_mode(),
        }
    }
}

impl MockProvider {
    pub fn new(settings: Arc<Settings>) -> Result<Self, ProviderError> {
        let responses_dir = match &settings.inference.provider {
            crate::config::InferenceProvider::Mock { responses_dir } => responses_dir,
            _ => {
                return Err(ProviderError::Configuration(
                    "Invalid provider configuration for MockProvider".to_string(),
                ));
            }
        };

        // Verify directory exists
        if !responses_dir.exists() {
            return Err(ProviderError::Configuration(format!(
                "Mock responses directory does not exist: {responses_dir:?}"
            )));
        }

        if !responses_dir.is_dir() {
            return Err(ProviderError::Configuration(format!(
                "Mock responses path is not a directory: {responses_dir:?}"
            )));
        }

        info!(
            "Initialized mock provider with responses from: {:?}",
            responses_dir
        );

        Ok(Self {
            settings,
            response_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get responses directory from settings
    fn responses_dir(&self) -> &PathBuf {
        match &self.settings.inference.provider {
            crate::config::InferenceProvider::Mock { responses_dir } => responses_dir,
            _ => panic!("MockProvider misconfigured"), // This should never happen given constructor validation
        }
    }

    /// Extract scenario name from model name (strip "mock-" prefix)
    fn extract_scenario(&self, model: &str) -> Result<String, ProviderError> {
        if !model.starts_with("mock-") {
            return Err(ProviderError::Configuration(format!(
                "Mock provider requires model names starting with 'mock-', got: {model}"
            )));
        }

        Ok(model.strip_prefix("mock-").unwrap().to_string())
    }

    /// Load responses from YAML file
    fn load_responses(&self, scenario: &str) -> Result<MockResponseFile, ProviderError> {
        // Check cache first
        {
            let cache = self.response_cache.lock().unwrap();
            if let Some(responses) = cache.get(scenario) {
                debug!("Using cached responses for scenario: {}", scenario);
                return Ok(responses.clone());
            }
        }

        // Try to load from file
        let file_path = self.responses_dir().join(format!("{scenario}.yaml"));

        if !file_path.exists() {
            // Try default.yaml as fallback
            let default_path = self.responses_dir().join("default.yaml");
            if default_path.exists() {
                warn!("Scenario '{}' not found, using default.yaml", scenario);
                return self.load_file(&default_path, "default");
            }

            return Err(ProviderError::Configuration(format!(
                "No mock responses found for scenario: {scenario} (looked for {file_path:?})"
            )));
        }

        self.load_file(&file_path, scenario)
    }

    /// Load and parse a YAML file
    fn load_file(&self, path: &Path, scenario: &str) -> Result<MockResponseFile, ProviderError> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            ProviderError::Configuration(format!("Failed to read mock file {path:?}: {e}"))
        })?;

        let response_file: MockResponseFile = serde_yaml::from_str(&contents).map_err(|e| {
            ProviderError::Configuration(format!("Failed to parse YAML from {path:?}: {e}"))
        })?;

        if response_file.responses.is_empty() {
            return Err(ProviderError::Configuration(format!(
                "No responses defined in {path:?}"
            )));
        }

        // Cache the loaded responses
        {
            let mut cache = self.response_cache.lock().unwrap();
            cache.insert(scenario.to_string(), response_file.clone());
        }

        info!(
            "Loaded {} responses for scenario: {}",
            response_file.responses.len(),
            scenario
        );
        Ok(response_file)
    }

    /// Select a response based on the mode
    fn select_response(&self, responses: &MockResponseFile, scenario: &str) -> MockResponse {
        match responses.settings.mode {
            ResponseMode::First => {
                debug!("Using first response for scenario: {}", scenario);
                responses.responses[0].clone()
            }
            ResponseMode::Sequential => {
                // This is simplified - in production you'd want persistent state
                // For now, just use the first response
                // TODO: Implement proper sequential tracking
                debug!("Sequential mode - returning first response (TODO: implement cycling)");
                responses.responses[0].clone()
            }
            ResponseMode::Random => {
                use rand::Rng;
                let mut rng = rand::rng();
                let index = rng.random_range(0..responses.responses.len());
                debug!(
                    "Random mode - selected response {} of {}",
                    index + 1,
                    responses.responses.len()
                );
                responses.responses[index].clone()
            }
        }
    }
}

#[async_trait]
impl InferenceProvider for MockProvider {
    fn build_inference_request(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> Result<InferenceRequest, ProviderError> {
        // Simple pass-through for mock provider
        Ok(InferenceRequest {
            messages: request.messages.clone(),
            model: model.to_string(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
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
        provider = "mock",
        model = %request.model,
    ))]
    async fn execute(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, ProviderError> {
        // Extract scenario from model name
        let scenario = self.extract_scenario(&request.model)?;

        // Load responses for this scenario
        let response_file = self.load_responses(&scenario)?;

        // Select a response based on mode
        let mock_response = self.select_response(&response_file, &scenario);

        // Simulate latency if specified
        if let Some(delay_ms) = mock_response.delay_ms {
            debug!("Simulating {}ms latency", delay_ms);
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
        }

        // Build the inference response
        Ok(InferenceResponse {
            text: mock_response.text,
            model_used: mock_response.model_used,
            total_tokens: mock_response.total_tokens,
            prompt_tokens: mock_response.prompt_tokens,
            completion_tokens: mock_response.completion_tokens,
            finish_reason: Some(mock_response.finish_reason),
            latency_ms: mock_response.delay_ms,
            provider_request_id: Some(format!("mock-{}-{}", scenario, Uuid::now_v7())),
            system_fingerprint: mock_response.system_fingerprint,
            tool_calls: mock_response.tool_calls,
            logprobs: mock_response.logprobs,
            provider_data: Some(
                [
                    ("scenario".to_string(), serde_json::json!(scenario)),
                    (
                        "mode".to_string(),
                        serde_json::json!(format!("{:?}", response_file.settings.mode)),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
        })
    }

    fn build_completion_response(
        &self,
        response: &InferenceResponse,
        original_request: &CompletionRequest,
    ) -> CompletionResponse {
        // Use the standard helper
        standard_completion_response(response, original_request, self.name())
    }

    fn name(&self) -> &str {
        "mock"
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        // Check that we can access the responses directory
        if !self.responses_dir().exists() {
            return Err(ProviderError::Configuration(format!(
                "Mock responses directory no longer exists: {:?}",
                self.responses_dir()
            )));
        }
        Ok(())
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        // List all available mock scenarios
        let mut models = Vec::new();

        let entries = std::fs::read_dir(self.responses_dir()).map_err(|e| {
            ProviderError::Configuration(format!("Failed to read mock responses directory: {e}"))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                ProviderError::Configuration(format!("Failed to read directory entry: {e}"))
            })?;

            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("yaml") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    models.push(format!("mock-{stem}"));
                }
            }
        }

        models.sort();
        Ok(models)
    }

    // ===== Streaming Support =====

    /// Mock provider supports streaming to demonstrate chunked delivery
    fn supports_streaming(&self) -> bool {
        true
    }

    /// Stream completion by chunking the mock response with realistic delays
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
        use futures_util::stream::{self, StreamExt};
        use std::time::Duration;
        use uuid::Uuid;

        // Reuse existing logic to get the mock response
        let inference_req = self.build_inference_request(request, model)?;
        let scenario = self.extract_scenario(&inference_req.model)?;
        let response_file = self.load_responses(&scenario)?;
        let mock_response = self.select_response(&response_file, &scenario);

        // Generate a unique request ID for this stream
        let request_id = format!("mock-{}-{}", scenario, Uuid::now_v7());
        let model_name = mock_response.model_used.clone();

        // Split response into tokens for streaming
        let tokens = super::tokenize_for_streaming(&mock_response.text);

        // Clone values for the final chunk closure
        let final_request_id = request_id.clone();
        let final_model_name = model_name.clone();
        let final_finish_reason = mock_response.finish_reason.clone();
        let final_usage_info = (
            mock_response.prompt_tokens,
            mock_response.completion_tokens,
            mock_response.total_tokens,
        );

        // Create stream that yields chunks with delay
        let chunks_stream = stream::iter(tokens.into_iter().enumerate()).then(move |(i, token)| {
            let chunk_id = request_id.clone();
            let chunk_model = model_name.clone();
            let base_delay = mock_response.delay_ms.unwrap_or(50);

            async move {
                // Simulate realistic token generation delay
                tokio::time::sleep(Duration::from_millis(base_delay)).await;

                if i == 0 {
                    // First chunk includes role
                    Ok(super::create_first_chunk(
                        &chunk_id,
                        &chunk_model,
                        "assistant",
                    ))
                } else {
                    // Content chunks
                    Ok(super::create_content_chunk(&chunk_id, &chunk_model, &token))
                }
            }
        });

        // Add final chunk with finish_reason and usage
        let final_chunk_stream = stream::once(async move {
            let usage = if final_usage_info.2.is_some()
                || final_usage_info.0.is_some()
                || final_usage_info.1.is_some()
            {
                Some(crate::models::Usage {
                    prompt_tokens: final_usage_info.0,
                    completion_tokens: final_usage_info.1,
                    total_tokens: final_usage_info.2,
                })
            } else {
                None
            };

            Ok(super::create_final_chunk(
                &final_request_id,
                &final_model_name,
                &final_finish_reason,
                usage,
            ))
        });

        // Combine the streams
        let combined_stream = chunks_stream.chain(final_chunk_stream);

        Ok(Box::pin(combined_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        HttpConfigSchema, InferenceConfig, LogFormat, LogOutput, LoggingConfig, ServerConfig,
    };
    use std::fs;
    use tempfile::TempDir;

    fn create_test_settings(responses_dir: PathBuf) -> Arc<Settings> {
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
                provider: crate::config::InferenceProvider::Mock { responses_dir },
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
    fn test_extract_scenario() {
        let temp_dir = TempDir::new().unwrap();
        let provider =
            MockProvider::new(create_test_settings(temp_dir.path().to_path_buf())).unwrap();

        assert_eq!(provider.extract_scenario("mock-test").unwrap(), "test");
        assert_eq!(
            provider.extract_scenario("mock-integration").unwrap(),
            "integration"
        );
        assert!(provider.extract_scenario("not-mock").is_err());
    }

    #[tokio::test]
    async fn test_load_responses() {
        let temp_dir = TempDir::new().unwrap();

        // Create a test YAML file
        let yaml_content = r#"
responses:
  - text: "Test response"
    model_used: "mock-test"
    prompt_tokens: 5
    completion_tokens: 3
    total_tokens: 8
settings:
  mode: first
"#;

        let file_path = temp_dir.path().join("test.yaml");
        fs::write(&file_path, yaml_content).unwrap();

        let provider =
            MockProvider::new(create_test_settings(temp_dir.path().to_path_buf())).unwrap();
        let response_file = provider.load_responses("test").unwrap();

        assert_eq!(response_file.responses.len(), 1);
        assert_eq!(response_file.responses[0].text, "Test response");
    }
}
