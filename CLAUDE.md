# Inference Server Project Context

## Project Overview
A Rust-based REST API server that provides AI model inference by interfacing with LM Studio (and future providers like Triton, OpenAI).

## Architecture Decisions

### Core Technologies
- **Framework**: Axum 0.8 for HTTP server
- **Async Runtime**: Tokio
- **Serialization**: Serde with JSON
- **HTTP Client**: Reqwest for calling LM Studio
- **Logging**: OpenTelemetry with tracing, supporting both stdout and file outputs
- **Configuration**: Layered config system (files + env vars)

### Key Design Patterns

1. **Error Handling**: Using `IntoResponse` trait for all error types, providing structured JSON error responses
2. **Configuration**: Layered approach - defaults → config files → environment variables
3. **Validation**: Separated validation logic in `validation.rs` with comprehensive tests
4. **Provider Abstraction**: Enum-based provider configuration ready for multiple inference backends

### Project Structure

```
inference-service/
├── services/
│   └── inference-server/
│       ├── src/
│       │   ├── main.rs           # Server setup and request handler
│       │   ├── config.rs         # Configuration structures
│       │   ├── models.rs         # Request/response models
│       │   ├── error.rs          # Error types and handling
│       │   ├── validation.rs     # Request validation logic
│       │   ├── lm_studio_client.rs # LM Studio API client
│       │   └── telemetry.rs      # Logging/observability setup
│       └── config/
│           ├── default.yaml      # Default configuration
│           └── production.yaml   # Production overrides
```
### API Design

#### POST /completions
- Model is specified per request (industry standard)
- Falls back to default model if not specified
- Validates against allowed models list if configured
- Returns 400 for invalid/non-existent models

Request:
```json
{
  "prompt": "string",
  "model": "string (optional)",
  "max_tokens": "number (optional)",
  "temperature": "number (optional, 0-2)"
}
```

### Configuration Schema

- `server.host`, `server.port`: Server binding
- `inference.provider`: Currently "lmstudio", extensible to others
- `inference.default_model`: Fallback model if not specified in request
- `inference.allowed_models`: Optional HashSet for model validation
- `logging.level`, `logging.format`, `logging.output`: Observability config

### Testing Strategy

- Unit tests for validation logic
- Integration tests planned for API endpoints
- All validation rules have test coverage

### Future Enhancements Discussed

- Metrics collection (request counts, latencies, token usage)
- Distributed tracing with OpenTelemetry
- Streaming responses for real-time token generation
- Additional providers (Triton, OpenAI)
- Model availability endpoint (GET /models)

### Rust Version
Using Rust 1.88.0 (2025) - note some behavior differences from earlier versions regarding struct initialization.

### Development Commands
```bash
# Run with debug logging
RUST_LOG=debug cargo run

# Run with production config
RUN_ENV=production cargo run

# Run tests
cargo test

# Run specific test module
cargo test validation
```