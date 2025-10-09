mod config;
mod error;
mod models;
mod providers; // Must be before config since config uses it
mod telemetry;
mod validations;

use axum::{
    Json, Router,
    extract::State,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures_util::{Stream, StreamExt};
use serde::Serialize;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{debug, info, instrument};

use providers::InferenceProvider;
use validations::{
    determine_model, validate_completion_request, validate_model_allowed,
    validate_provider_capabilities,
};

use config::Settings;
use error::ApiError;
use models::{CompletionRequest, CompletionResponse};

// Hold the http client and provider settings
#[derive(Clone)]
struct AppState {
    provider: Arc<dyn InferenceProvider>,
    settings: Arc<Settings>,
}

// Type alias for complex SSE stream type
type SseStream = Sse<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>>;

// Enum that can return either JSON response or SSE stream
enum CompletionOrStream {
    Json(Json<CompletionResponse>),
    Stream(SseStream),
}

impl IntoResponse for CompletionOrStream {
    fn into_response(self) -> Response {
        match self {
            CompletionOrStream::Json(json) => json.into_response(),
            CompletionOrStream::Stream(sse) => sse.into_response(),
        }
    }
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

    let settings = Arc::new(settings);
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
    let listener = TcpListener::bind(&addr).await.unwrap();

    info!("Server listening on {}", addr);

    axum::serve(listener, app).await.unwrap();

    telemetry::shutdown_logging(logger_provider);
}

// Factory function to create the right provider
fn create_provider(
    settings: &Arc<Settings>,
) -> Result<Arc<dyn InferenceProvider>, Box<dyn std::error::Error>> {
    use config::InferenceProvider as ConfigProvider;
    use providers::lmstudio::LMStudioProvider;
    use providers::mock::MockProvider;
    use providers::openai::OpenAIProvider;

    match &settings.inference.provider {
        ConfigProvider::LMStudio => Ok(Arc::new(
            LMStudioProvider::new(settings.clone())
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
        )),
        ConfigProvider::Mock { .. } => Ok(Arc::new(
            MockProvider::new(settings.clone())
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
        )),
        ConfigProvider::OpenAI { .. } => Ok(Arc::new(
            OpenAIProvider::new(settings.clone())
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
        )),
        ConfigProvider::Triton { .. } => Err("Triton provider not yet implemented".into()),
    }
}
// Update the generate_completion function in main.rs:

#[instrument(skip(state), fields(
    message_count = request.messages.len(),
    model = request.model.as_deref().unwrap_or("default"),
    stream = request.stream.unwrap_or(false),
    user = request.user.as_deref(),
))]
async fn generate_completion(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<CompletionOrStream, ApiError> {
    // Validate the incoming request structure
    validate_completion_request(&request)?;

    // Determine which model to use (applies defaults if needed)
    let model = determine_model(
        request.model.as_deref(),
        &state.settings.inference.default_model,
        state.settings.inference.allowed_models.as_ref(),
    )?;

    // Validate the model is allowed (if restrictions are configured)
    validate_model_allowed(model, state.settings.inference.allowed_models.as_ref())?;

    // Validate provider capabilities
    validate_provider_capabilities(
        &request,
        state.provider.supports_streaming(),
        false, // tools not yet supported
    )?;

    debug!("Using model: {}", model);

    // Check if streaming is requested
    if request.stream == Some(true) {
        // Get stream from provider
        let provider_stream = state
            .provider
            .stream(&request, model)
            .await
            .map_err(ApiError::Provider)?;

        // Convert to SSE events
        let sse_stream = provider_stream
            .map(|chunk_result| {
                match chunk_result {
                    Ok(chunk) => {
                        // Format as SSE: "data: {json}\n\n"
                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        Ok(Event::default().data(json))
                    }
                    Err(e) => {
                        // Send error in stream
                        let error = serde_json::json!({
                            "error": {
                                "message": e.to_string(),
                                "type": "stream_error"
                            }
                        });
                        Ok(Event::default().data(error.to_string()))
                    }
                }
            })
            .chain(futures_util::stream::once(async {
                // Send [DONE] marker
                Ok(Event::default().data("[DONE]"))
            }));

        info!(model = model, stream = true, "Streaming completion started");

        return Ok(CompletionOrStream::Stream(
            Sse::new(Box::pin(sse_stream)
                as Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>)
            .keep_alive(KeepAlive::default()),
        ));
    }

    // Non-streaming: Use the provider to generate completion
    let response = state
        .provider
        .generate(&request, model)
        .await
        .map_err(ApiError::Provider)?;

    // Log only if we have usage information
    if let Some(ref usage) = response.usage {
        info!(
            model = model,
            choices_count = response.choices.len(),
            total_tokens = ?usage.total_tokens,
            prompt_tokens = ?usage.prompt_tokens,
            completion_tokens = ?usage.completion_tokens,
            stream = false,
            "Completion successful"
        );
    } else {
        info!(
            model = model,
            choices_count = response.choices.len(),
            stream = false,
            "Completion successful (no usage data)"
        );
    }

    Ok(CompletionOrStream::Json(Json(response)))
}

async fn list_models(State(state): State<AppState>) -> Result<Json<ModelsResponse>, ApiError> {
    let models = state
        .provider
        .list_models()
        .await
        .map_err(ApiError::Provider)?;

    let model_list = models
        .into_iter()
        .map(|id| ModelInfo {
            id,
            object: "model".to_string(),
            owned_by: "local".to_string(),
        })
        .collect();

    Ok(Json(ModelsResponse {
        object: "list".to_string(),
        data: model_list,
    }))
}

async fn health_check(State(state): State<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    // Check if provider is healthy
    state
        .provider
        .health_check()
        .await
        .map_err(ApiError::Provider)?;

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
