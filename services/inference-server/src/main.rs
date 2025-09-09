use axum::{extract::{State}, routing::{get, post}, Json, Router};
use tokio::net::TcpListener;
use serde::Serialize;
use tracing::{debug, info, instrument};

mod telemetry;
mod validations;
use validations::{validate_completion_request, determine_model, validate_model_allowed};

mod lm_client_studio;
use lm_client_studio::{call_lm_studio};

mod error;
mod config;
mod models;
use error::ErrorResponse;
use config::Settings;
use error::ApiError;
use models::{CompletionRequest, CompletionResponse};

// Hold the http client and lm studio base url
#[derive(Clone)]
struct AppState {
    http_client: reqwest::Client,
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

    let app_state = AppState {
        http_client: reqwest::Client::new(),
        settings: settings.clone(),
    };

    let app: Router = Router::new()
    .route("/", get(root))
    .route("/completions", post(generate_completion))
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

#[instrument(skip(state), fields(
    prompt_length = request.prompt.len(),
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

    let lm_response = call_lm_studio(
        &state.http_client,
        &state.settings.inference.base_url,
        &request.prompt,
        request.max_tokens,
        request.temperature,
        model,
    )
    .await?;

    info!(
        model = model,
        response_length = lm_response.text.len(),
        lm_response = lm_response.text,
        total_tokens = ?lm_response.total_tokens,
        prompt_tokens = ?lm_response.prompt_tokens,
        completion_tokens = ?lm_response.completion_tokens,
        "Completion successful"
    );

    Ok(Json(CompletionResponse {
        provider: state.settings.inference.provider_name().to_string(),
        text: lm_response.text,
        model: model.to_string(),
        tokens_used: lm_response.total_tokens,
    }))
}

async fn root() -> Json<RootResponse> {
    Json(RootResponse {
        message: "Ok".to_string(),
    })
}

    
