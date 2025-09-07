use axum::{extract::{State}, routing::{get, post}, Json, Router};
use tokio::net::TcpListener;
use serde::{Deserialize, Serialize};
use opentelemetry_sdk::logs::SdkLoggerProvider;
use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use tracing::{debug, info, instrument};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod validations;
use validations::{validate_completion_request};

mod lm_client_studio;
use lm_client_studio::{call_lm_studio};

mod error;
use error::ErrorResponse;

use crate::error::ApiError;

// Hold the http client and lm studio base url
#[derive(Clone)]
struct AppState {
    http_client: reqwest::Client,
    lm_studio_base_url: String,
}

// Root response to health check
#[derive(Serialize)]
struct RootResponse {
    message: String,
}

#[derive(Deserialize, Debug)]
struct CompletionRequest {
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

#[derive(Serialize, Debug)]
struct CompletionResponse {
    text: String,
    model: String,
    tokens_used: Option<u32>,
}

#[tokio::main]
async fn main() {
    let logger_provider = init_logging();

    let app_state = AppState {
        http_client: reqwest::Client::new(),
        lm_studio_base_url: "http://localhost:1234/v1".to_string(),
    };

    let app: Router = Router::new()
    .route("/", get(root))
    .route("/completions", post(generate_completion))
    .with_state(app_state);

    let listener = TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    axum::serve(listener, app)
        .await
        .unwrap();

    let _ = logger_provider.shutdown();
}

#[instrument(skip(state), fields(prompt_length = request.prompt.len()))]
async fn generate_completion(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
)   -> Result<Json<CompletionResponse>, ApiError> {

    debug!("Prompt: {}", request.prompt);

    validate_completion_request(&request)?;

    let lm_response = call_lm_studio(
        &state.http_client,
        &state.lm_studio_base_url,
        &request.prompt,
        request.max_tokens,
        request.temperature,
        "gpt-oss-20b",
    )
    .await?;

    info!(
        model = "gpt-oss-20b",
        response_length = lm_response.text.len(),
        total_tokens = ?lm_response.total_tokens,
        prompt_tokens = ?lm_response.prompt_tokens,
        completion_tokens = ?lm_response.completion_tokens,
        "Completion successful"
    );

    Ok(Json(CompletionResponse {
        text: lm_response.text,
        model: "gpt-oss-20b".to_string(),
        tokens_used: lm_response.total_tokens,
    }))
}

async fn root() -> Json<RootResponse> {
    Json(RootResponse {
        message: "Ok".to_string(),
    })
}

// initialize OpenTelemetry logging and tracing
fn init_logging() -> SdkLoggerProvider {
    let exporter = opentelemetry_stdout::LogExporter::default();

    let logger_provider = SdkLoggerProvider::builder()
        .with_simple_exporter(exporter)
        .build();

    // Initialize the OpenTelemetry tracing layer and bridge the logging
    let telemetry_layer = OpenTelemetryTracingBridge::new(&logger_provider);

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")))
        .with(telemetry_layer)
        .init();

    logger_provider
}


    
