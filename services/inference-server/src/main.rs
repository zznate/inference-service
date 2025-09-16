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

use config::Settings;
use error::ApiError;
use models::{CompletionRequest, CompletionResponse};
use providers::mock::MockProvider;

// Hold the http client and provider settings
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
        provider,
        settings: settings.clone(),
    };

    let app: Router = Router::new()
        .route("/", get(root))
        .route("/v1/chat/completions", post(generate_completion))
        .route("/v1/models", get(list_models))
        .route("/health", get(health_check))
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
        ConfigProvider::Mock { responses_dir } => {
            Ok(Arc::new(MockProvider::new(
                responses_dir.clone(),
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
) -> Result<Json<CompletionResponse>, ApiError> {

    // Validate the incoming request
    validate_completion_request(&request)?;
    
    // Determine which model to use (applies defaults if needed)
    let model = determine_model(
        request.model.as_deref(),
        &state.settings.inference.default_model,
        state.settings.inference.allowed_models.as_ref(),
    )?;

    // Validate the model is allowed (if restrictions are configured)
    validate_model_allowed(model, state.settings.inference.allowed_models.as_ref())?;

    debug!("Using model: {}", model);

    // Use the new provider contract - it now returns CompletionResponse directly
    let response = state.provider
        .generate(&request, model)
        .await
        .map_err(|e| ApiError::Provider(e))?;

    info!(
        model = model,
        choices_count = response.choices.len(),
        total_tokens = response.usage.total_tokens,
        prompt_tokens = response.usage.prompt_tokens,
        completion_tokens = response.usage.completion_tokens,
        "Completion successful"
    );
    
    Ok(Json(response))
}

async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<ModelsResponse>, ApiError> {
    let models = state.provider
        .list_models()
        .await
        .map_err(|e| ApiError::Provider(e))?;
    
    let model_list = models.into_iter().map(|id| ModelInfo {
        id,
        object: "model".to_string(),
        owned_by: "local".to_string(),
    }).collect();
    
    Ok(Json(ModelsResponse {
        object: "list".to_string(),
        data: model_list,
    }))
}

async fn health_check(
    State(state): State<AppState>,
) -> Result<Json<HealthResponse>, ApiError> {
    // Check if provider is healthy
    state.provider
        .health_check()
        .await
        .map_err(|e| ApiError::Provider(e))?;
    
    // Get HTTP config if available (for providers that use HTTP)
    let http_config = state.provider.http_config().map(|config| HttpConfigInfo {
        timeout_secs: config.timeout_secs,
        connect_timeout_secs: config.connect_timeout_secs,
        max_retries: config.max_retries,
    });
    
    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        provider: state.provider.name().to_string(),
        http_config,
    }))
}

async fn root() -> Json<RootResponse> {
    Json(RootResponse {
        message: "Ok".to_string(),
    })
}

// Response types for the API endpoints
#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    provider: String,
    http_config: Option<HttpConfigInfo>,
}

#[derive(Serialize)]
struct HttpConfigInfo {
    timeout_secs: u64,
    connect_timeout_secs: u64,
    max_retries: u32,
}