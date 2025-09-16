# Inference Service

A production-ready Rust-based inference server that provides a unified REST API for multiple AI model providers. Currently supports LM Studio with a clean abstraction layer for adding additional providers. Support for OpenAI and a local Triton Inference Server is coming soon. 

## Features

- **OpenAI-Compatible API**: Full compatibility with OpenAI's chat completion format
- **Provider Abstraction**: Clean trait-based design for multiple inference backends
- **Production Ready**: 
  - Comprehensive error handling with structured JSON responses
  - Request validation with detailed error messages
  - OpenTelemetry instrumentation for logging (metrics and tracing coming soon)
  - Configurable timeouts, retries, and connection pooling
- **Flexible Configuration**: Layered configuration system (defaults → YAML files → environment variables)
- **Type Safety**: Strong typing throughout with separate internal and external data models

## Architecture

The service uses a clean separation of concerns:

- **API Layer** (`models.rs`): OpenAI-compatible request/response structures
- **Provider Layer** (`providers/`): Trait-based abstraction for different inference backends
- **Validation Layer** (`validations.rs`): Comprehensive request validation
- **Configuration** (`config.rs`): Flexible, layered configuration system
- **Telemetry** (`telemetry.rs`): Structured logging with file rotation support

## Installation

### Prerequisites

- Rust 1.88.0 or later
- LM Studio (or another supported inference provider)

### Building

```bash
# Clone the repository
git clone <repository-url>
cd inference-service

# Build the project
cargo build --release

# Run tests
cargo test
```

## Configuration

The service uses a layered configuration approach:

1. Default configuration (`config/default.yaml`)
2. Environment-specific overrides (e.g., `config/production.yaml`)
3. Environment variables (prefixed with `INFERENCE_`)

### Example Configuration

```yaml
# config/default.yaml
server:
  host: "0.0.0.0"
  port: 3000

inference:
  provider: lmstudio
  base_url: "http://127.0.0.1:1234/v1"
  default_model: "gpt-oss-20b"
  timeout_secs: 30
  # Optional: restrict allowed models
  allowed_models:
    - "gpt-oss-20b"
    - "llama-2-7b"
    - "mistral-7b"

logging:
  level: info
  format: pretty
  output: stdout
```

### Environment Variables

Override any configuration via environment variables:

```bash
export INFERENCE_SERVER_PORT=8080
export INFERENCE_INFERENCE_BASE_URL="http://localhost:1234/v1"
export INFERENCE_INFERENCE_DEFAULT_MODEL="llama-2-7b"
export INFERENCE_LOGGING_LEVEL=debug
```

## Running the Server

### Development Mode

```bash
# Start with debug logging
RUST_LOG=debug cargo run

# Use a specific environment configuration
RUN_ENV=production cargo run
```

### Production Mode

```bash
# Build optimized binary
cargo build --release

# Run with production config
RUN_ENV=production ./target/release/inference-server
```

### With Docker (coming soon)

```bash
docker build -t inference-server .
docker run -p 8080:8080 inference-server
```

## API Endpoints

### Health Check

```bash
# Basic health check
curl http://localhost:3000/

# Response
{"message":"Ok"}
```

### Provider Health Check

```bash
# Detailed health check with provider info
curl http://localhost:3000/health

# Response
{
  "status": "healthy",
  "provider": "lmstudio",
  "http_config": {
    "timeout_secs": 30,
    "connect_timeout_secs": 30,
    "max_retries": 3
  }
}
```

### List Available Models

```bash
curl http://localhost:3000/v1/models

# Response
{
  "object": "list",
  "data": [
    {
      "id": "gpt-oss-20b",
      "object": "model",
      "owned_by": "local"
    },
    {
      "id": "llama-2-7b",
      "object": "model", 
      "owned_by": "local"
    }
  ]
}
```

### Chat Completion

```bash
# Basic completion request
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'

# With specific model and parameters
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'

# Response
{
  "id": "chatcmpl-018e3c4a-2c3f-7f3f-8b9f-5a8b9f5a8b9f",
  "object": "chat.completion",
  "created": 1677649420,
  "model": "gpt-oss-20b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 7,
    "total_tokens": 19
  }
}
```

### Advanced Parameters

The service supports additional OpenAI-compatible parameters:

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "Write a creative story about a robot."}
    ],
    "temperature": 0.9,
    "max_tokens": 200,
    "top_p": 0.95,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "stop": ["The end", "END"],
    "seed": 42
  }'
```

## Error Handling

The service provides structured error responses with appropriate HTTP status codes:

### Validation Error Example

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": []
  }'

# Response (400 Bad Request)
{
  "error": "Messages array cannot be empty",
  "code": "EMPTY_MESSAGES"
}
```

### Model Not Allowed Example

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Response (400 Bad Request)
{
  "error": "Model 'gpt-4' is not in the allowed list. Available models: gpt-oss-20b, llama-2-7b",
  "code": "MODEL_NOT_ALLOWED"
}
```

### Provider Error Example

```bash
# When LM Studio is not running
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Response (502 Bad Gateway)
{
  "error": "Connection failed: error sending request",
  "code": "PROVIDER_CONNECTION_FAILED"
}
```

## Testing

```bash
# Run all tests
cargo test

# Run specific test module
cargo test validation

# Run with debug output
RUST_LOG=debug cargo test

# Run integration tests (requires LM Studio running)
cargo test --test '*' -- --test-threads=1
```

## Development

### Project Structure

```
inference-service/
├── services/
│   └── inference-server/
│       ├── src/
│       │   ├── main.rs              # Server setup and routing
│       │   ├── config.rs            # Configuration structures
│       │   ├── models.rs            # OpenAI-compatible API models
│       │   ├── error.rs             # Error types and handling
│       │   ├── validation.rs        # Request validation logic
│       │   ├── telemetry.rs         # Logging and observability
│       │   └── providers/
│       │       ├── mod.rs           # Provider trait and common types
│       │       └── lmstudio.rs      # LM Studio implementation
│       └── config/
│           ├── default.yaml         # Default configuration
│           └── production.yaml      # Production overrides
```

### Adding a New Provider

1. Create a new file in `src/providers/` (e.g., `openai.rs`)
2. Implement the `InferenceProvider` trait:
   ```rust
   impl InferenceProvider for OpenAIProvider {
       fn build_inference_request(...) -> Result<InferenceRequest, ProviderError>
       async fn execute(...) -> Result<InferenceResponse, ProviderError>
       fn build_completion_response(...) -> CompletionResponse
   }
   ```
3. Add the provider to the factory in `main.rs`
4. Update configuration structures in `config.rs`

## Roadmap

- [x] LM Studio provider support
- [x] OpenAI-compatible API
- [x] Request validation
- [x] Structured error handling
- [x] File-based logging with rotation
- [ ] Metrics collection (Prometheus)
- [ ] Distributed tracing (Jaeger/Zipkin)
- [ ] Streaming responses
- [ ] Triton Inference Server support
- [ ] OpenAI API support
- [ ] Request/response caching
- [ ] Rate limiting
- [ ] Circuit breaker pattern
- [ ] WebSocket support
- [ ] Multi-model routing
- [ ] A/B testing capabilities

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass (`cargo test`)
2. Code follows Rust idioms and conventions
3. New features include appropriate tests
4. Documentation is updated as needed

## License

[Your License Here]

## Support

For issues, questions, or suggestions, please open an issue on GitHub.