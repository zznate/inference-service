use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===== API-Facing Models (Full OpenAI Compatibility) =====

/// Response mode determines whether provider-specific extensions are included in responses
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ResponseMode {
    /// Standard OpenAI-compatible response (default)
    #[default]
    Standard,
    /// Extended response including provider-specific data
    Extended,
}

/// Provider-specific extension data returned in extended mode
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProviderExtensions {
    pub provider: String,
    pub data: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>, // Can be null when using tools/functions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>, // Name of the author of this message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>, // Tool calls made by assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>, // ID of the tool call this message is responding to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>, // Deprecated: use tool_calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>, // If the assistant refuses a request
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // Usually "function"
    pub function: FunctionCall,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON string of arguments
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct CompletionRequest {
    #[serde(default)]
    pub messages: Vec<Message>,
    pub model: Option<String>,

    // Response mode for controlling extension inclusion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mode: Option<ResponseMode>,

    // Provider-specific extension parameters
    // Clients must explicitly nest provider-specific params under this key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<HashMap<String, serde_json::Value>>,

    // Generation parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>, // -2.0 to 2.0
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Map<String, serde_json::Value>>, // Token ID to bias (-100 to 100)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>, // Whether to return log probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>, // 0-20, number of most likely tokens to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>, // Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>, // -2.0 to 2.0
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>, // For deterministic generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>, // Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>, // Whether to stream responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>, // 0.0 to 2.0
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>, // 0.0 to 1.0

    // Tool/Function calling (reserved for future implementation)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)] // TODO: Implement tool/function calling support
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)] // TODO: Implement tool/function calling support
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    // TODO: Implement tool/function calling support (deprecated, but supported)
    pub functions: Option<Vec<Function>>, // Deprecated: use tools
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    // TODO: Implement tool/function calling support (deprecated, but supported)
    pub function_call: Option<FunctionCallOption>, // Deprecated: use tool_choice

    // Additional options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>, // Unique identifier for end-user (for logging/metrics/abuse prevention)
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String, // "text" or "json_object"
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // Usually "function"
    pub function: Function,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>, // JSON Schema
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String), // "none", "auto", "required"
    Object {
        #[serde(rename = "type")]
        choice_type: String,
        function: ToolFunction,
    },
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ToolFunction {
    pub name: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum FunctionCallOption {
    String(String), // "none", "auto"
    Object { name: String },
}

// ===== Response Models =====

#[derive(Serialize, Deserialize, Debug)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>, // Optional for compatibility with streaming
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    /// Provider-specific extension data (only included in extended response mode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_extensions: Option<ProviderExtensions>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Choice {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>, // For non-streaming
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<Message>, // For streaming
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>, // "stop", "length", "tool_calls", "content_filter", "function_call"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogProbs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TokenLogProb>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenLogProb {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogProb>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Usage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

// ===== Streaming Response Models =====

/// Streaming response chunk matching OpenAI format
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamChunk {
    pub id: String,
    pub object: String, // Always "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>, // Only in final chunk for some providers
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta, // Note: delta, not message for streaming
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>, // Only in first chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>, // Token content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ===== Error Response =====

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenAIError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

// ===== Helper implementations =====

impl Default for Message {
    fn default() -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(String::new()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            function_call: None,
            refusal: None,
        }
    }
}

impl Message {
    /// Create a simple text message
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: Some(content.to_string()),
            ..Default::default()
        }
    }

    /// Create a tool response message
    #[allow(dead_code)] // TODO: Will be used when tool calling is implemented
    pub fn tool_response(tool_call_id: &str, content: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.to_string()),
            tool_call_id: Some(tool_call_id.to_string()),
            ..Default::default()
        }
    }
}

impl Choice {
    /// Create a simple non-streaming choice
    #[allow(dead_code)] // Helper function for tests and examples
    pub fn simple(content: &str, finish_reason: &str) -> Self {
        Self {
            index: 0,
            message: Some(Message::new("assistant", content)),
            delta: None,
            finish_reason: Some(finish_reason.to_string()),
            logprobs: None,
        }
    }
}

impl Usage {
    /// Create usage with all token counts
    #[allow(dead_code)] // Helper function for tests and examples
    pub fn new(prompt: u32, completion: u32) -> Self {
        Self {
            prompt_tokens: Some(prompt),
            completion_tokens: Some(completion),
            total_tokens: Some(prompt + completion),
        }
    }
}
