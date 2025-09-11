mod providers;  // Must be before config since config uses it
mod telemetry;
mod validations;
mod error;
mod config;
mod models;

use axum::{extract::{State}, routing::{get, post}, Json, Router};
use tokio::net::TcpListener;
use serde::Serialize;
use tracing::{debug, info, instrument};
use std::sync::Arc;

use providers::InferenceProvider;
use validations::{validate_completion_request, determine_model, validate_model_allowed};

use error::ErrorResponse;
use config::Settings;
use error::ApiError;
use models::{CompletionRequest, CompletionResponse, Choice, Message, Usage};

// Hold the http client and lm studio base url
#[derive(Clone)]
struct AppState {
    provider: Arc<dyn InferenceProvider>,
    settings: Settings,
}

// Root response to health check
#[derive(Serialize)]
struct RootResponse {
    message: String,
}



#[tokio::main]
async fn main() {
    let settings = Settings::new().expect("Failed to load configuration");

    let logger_provider = telemetry::init_logging(&settings.logging);

    let provider = create_provider(&settings).expect("Failed to create inference provider");

    let app_state = AppState {
        provider: provider,
        settings: settings.clone(),
    };

    let app: Router = Router::new()
        .route("/", get(root))
        .route("/v1/chat/completions", post(generate_completion))
        .with_state(app_state);

    let addr = format!("{}:{}", settings.server.host, settings.server.port);
    let listener = TcpListener::bind(&addr)
        .await
        .unwrap();
    
    info!("Server listening on {}", addr); 

    axum::serve(listener, app)
        .await
        .unwrap();

    telemetry::shutdown_logging(logger_provider);
}

// Factory function to create the right provider
fn create_provider(settings: &Settings) -> Result<Arc<dyn InferenceProvider>, Box<dyn std::error::Error>> {
    use config::{InferenceProvider as ConfigProvider, HttpConfigSchema};
    use providers::lmstudio::LMStudioProvider;
    
    // Get HTTP config - either from the new http field or fall back to timeout_secs
    let http_config = match &settings.inference.http {
        Some(http_schema) => http_schema.clone(),
        None => {
            // Fallback to old timeout_secs for backward compatibility
            HttpConfigSchema {
                timeout_secs: settings.inference.timeout_secs,
                ..Default::default()
            }
        }
    };
    
    match &settings.inference.provider {
        ConfigProvider::LMStudio => {
            Ok(Arc::new(LMStudioProvider::new(
                settings.inference.base_url.clone(),
                http_config,
            ).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?))
        },
        ConfigProvider::Triton { .. } => {
            Err("Triton provider not yet implemented".into())
        },
        ConfigProvider::OpenAI { .. } => {
            Err("OpenAI provider not yet implemented".into())
        },
    }
}


#[instrument(skip(state), fields(
    message_count = request.messages.len(),
    model = request.model.as_deref().unwrap_or("default"),
))]
async fn generate_completion(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
)   -> Result<Json<CompletionResponse>, ApiError> {

    validate_completion_request(&request)?;
    
    let model = determine_model(
        request.model.as_deref(),
        &state.settings.inference.default_model,
        state.settings.inference.allowed_models.as_ref(),
    )?;

    validate_model_allowed(model, state.settings.inference.allowed_models.as_ref())?;

    debug!("Using model: {}", model);

    let response = state.provider
        .generate(
            &request.messages,
            model,
            request.max_tokens,
            request.temperature,
        )
        .await
        .map_err(|e| ApiError::Provider(e))?;

    info!(
        model = model,
        response_length = response.text.len(),
        response_text = response.text,
        total_tokens = ?response.total_tokens,
        prompt_tokens = ?response.prompt_tokens,
        completion_tokens = ?response.completion_tokens,
        "Completion successful"
    );

    let response = CompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::now_v7()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: response.text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: response.prompt_tokens.unwrap_or(0),
            completion_tokens: response.completion_tokens.unwrap_or(0),
            total_tokens: response.total_tokens.unwrap_or(0),
        },
    };
    
    
    
    Ok(Json(response))
}

// debug!("Sending OpenAI-formatted response: {:#?}", response);

async fn root() -> Json<RootResponse> {
    Json(RootResponse {
        message: "Ok".to_string(),
    })
}

    
