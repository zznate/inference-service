use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub text: String,
    pub model: String,
    pub tokens_used: Option<u32>,
}