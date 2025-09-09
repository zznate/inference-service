use axum::{extract::{State}, routing::{get, post}, Json, Router};
use tokio::net::TcpListener;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

mod telemetry;
mod validations;
use validations::{validate_completion_request};

mod lm_client_studio;
use lm_client_studio::{call_lm_studio};

mod error;
mod config;
use error::ErrorResponse;
use config::Settings;
use error::ApiError;

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

#[instrument(skip(state), fields(prompt_length = request.prompt.len()))]
async fn generate_completion(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
)   -> Result<Json<CompletionResponse>, ApiError> {

    debug!("Prompt: {}", request.prompt);

    validate_completion_request(&request)?;

    let lm_config = state.settings.inference.lm_studio.as_ref()
        .expect("LM Studio config not found");

    let lm_response = call_lm_studio(
        &state.http_client,
        &lm_config.base_url,
        &request.prompt,
        request.max_tokens,
        request.temperature,
        &lm_config.model,
    )
    .await?;

    info!(
        model = &lm_config.model,
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

    
