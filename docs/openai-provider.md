# OpenAI Provider - API Reference

The OpenAI provider provides **full compatibility** with the OpenAI Chat Completions API, supporting all GPT models and parameters with comprehensive validation. This acts as a drop-in replacement for direct OpenAI API calls with built-in parameter validation, intelligent defaults, and **native streaming support** using Server-Sent Events (SSE).

## Quick Start
The following configuration uses the project defaults and represents a good starting point for development and testing: 

### Minimal Configuration

```yaml
inference:
  provider: openai
  api_key: "sk-your-openai-api-key-here"  # Required
  default_model: "gpt-3.5-turbo"
  # HTTP config optional - intelligent defaults provided automatically
```

### Full Configuration

As with any production service, you will want to tune the defaults of the HTTP configuration to match your environment. The following configuration represents a good starting point for production:

```yaml
inference:
  provider: openai
  api_key: "sk-your-openai-api-key-here"
  organization_id: "org-your-org-id"      # Optional
  base_url: "https://api.openai.com/v1"   # Optional, for Azure OpenAI etc.
  default_model: "gpt-3.5-turbo"
  http:                                    # Optional - has smart defaults
    timeout_secs: 30
    connect_timeout_secs: 10
    keep_alive_secs: 30
    max_idle_connections: 10
```

The reqwest HTTP client is used under the hood, and the configuration is passed to it. For more information, see the [reqwest documentation](https://docs.rs/reqwest/latest/reqwest/struct.ClientBuilder.html). If you need to tune the HTTP client further, patches are welcome provided they include sane defaults for all parameters.

### Environment-Specific Configurations

#### Development Environment

For development, you might want faster responses and lower costs:

```yaml
# config/development.yaml
inference:
  provider: openai
  api_key: "${OPENAI_API_KEY}"
  default_model: "gpt-3.5-turbo"
  http:
    timeout_secs: 15
    connect_timeout_secs: 5
    keep_alive_secs: 30
```

#### Production Environment

For production, use more robust settings with longer timeouts:

```yaml
# config/production.yaml
inference:
  provider: openai
  default_model: "gpt-4"
  http:
    timeout_secs: 60
    connect_timeout_secs: 15
    keep_alive_secs: 60
    max_idle_connections: 20
```

#### Azure OpenAI Configuration

For Azure OpenAI Service:

```yaml
inference:
  provider: openai
  api_key: "your-azure-openai-key"
  base_url: "https://your-resource-name.openai.azure.com/openai/deployments/your-deployment-name"
  default_model: "gpt-4"
  organization_id: null  # Not used with Azure
```

## Environment Variables

You can override configuration using environment variables:

```bash
# Required
export INFERENCE_INFERENCE_API_KEY="sk-your-api-key"

# Optional overrides
export INFERENCE_INFERENCE_PROVIDER="openai"
export INFERENCE_INFERENCE_DEFAULT_MODEL="gpt-4"
export INFERENCE_INFERENCE_BASE_URL="https://api.openai.com/v1"
export INFERENCE_INFERENCE_ORGANIZATION_ID="org-your-org-id"

# HTTP configuration
export INFERENCE_INFERENCE_HTTP_TIMEOUT_SECS="30"
export INFERENCE_INFERENCE_HTTP_CONNECT_TIMEOUT_SECS="10"
```

## API Usage Examples

### Basic Chat Completion

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

### Advanced Chat Completion with Full Parameter Set

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
    "stop": ["END", "STOP"],
    "seed": 42
  }'
```

### Streaming Chat Completion

For real-time token-by-token responses, enable streaming with the `stream` parameter:

```bash
curl -N -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Write a short story about a robot."}
    ],
    "stream": true,
    "max_tokens": 200
  }'
```

**Response Format (Server-Sent Events):**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Once"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" upon"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":50,"total_tokens":60}}

data: [DONE]
```

### Parameter Validation Examples

The server validates all parameters and returns detailed error messages:

```bash
# Invalid temperature (must be 0.0-2.0)
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 3.0
  }'
# Returns: 400 Bad Request with validation error

# Invalid top_p (must be 0.0-1.0)
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "top_p": 1.5
  }'
# Returns: 400 Bad Request with validation error
```

### List Available Models

```bash
curl http://localhost:3000/v1/models
```

### Health Check

```bash
curl http://localhost:3000/health
```

## Supported OpenAI Parameters

The OpenAI provider supports all standard OpenAI chat completion parameters with comprehensive validation:

| Parameter | Type | Validation | Description |
|-----------|------|------------|-------------|
| `model` | string | Required | The OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4") |
| `messages` | array | Required, non-empty | List of messages in the conversation |
| `max_tokens` | integer | 1-131072 | Maximum number of tokens to generate |
| `temperature` | number | 0.0-2.0 | Controls randomness |
| `top_p` | number | 0.0-1.0 | Nucleus sampling parameter |
| `frequency_penalty` | number | -2.0-2.0 | Penalize frequent tokens |
| `presence_penalty` | number | -2.0-2.0 | Penalize new tokens |
| `stop` | array/string | Max 4 sequences | Stop sequences to halt generation |
| `seed` | integer | Optional | For deterministic outputs (when supported) |
| **`stream`** | **boolean** | **Optional** | **Stream partial message deltas via Server-Sent Events** |
| `n` | integer | 1-128 | Number of chat completion choices to generate |
| `logprobs` | boolean | Optional | Return log probabilities of output tokens |
| `top_logprobs` | integer | 0-20 | Number of most likely tokens to return (requires logprobs) |

### Validation Features

- **Parameter bounds checking**: All numeric parameters are validated against OpenAI's documented ranges
- **Model validation**: Checks against configured allowed models list (if specified)
- **Message validation**: Ensures messages array is non-empty with valid role/content pairs
- **Stop sequences**: Validates array length and individual sequence constraints
- **Streaming validation**: Ensures provider supports streaming when `stream: true` is requested
- **Detailed error responses**: Returns specific validation errors with parameter names and valid ranges

## Streaming Support

The OpenAI provider includes **native streaming support** that connects directly to OpenAI's streaming API:

### Features
- **Real-time responses**: Token-by-token delivery as they're generated
- **Server-Sent Events**: Standard SSE format for web compatibility
- **Graceful error handling**: Stream errors are reported within the SSE stream
- **OpenAI compatibility**: Identical format to OpenAI's native streaming
- **Automatic parsing**: Handles `[DONE]` markers and chunk formatting

### Client Implementation

**JavaScript/Browser:**
```javascript
const response = await fetch('/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'gpt-4',
    messages: [{role: 'user', content: 'Hello!'}],
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data === '[DONE]') return;

      const parsed = JSON.parse(data);
      const content = parsed.choices[0]?.delta?.content;
      if (content) {
        console.log(content); // Display token
      }
    }
  }
}
```

**Python:**
```python
import requests
import json

response = requests.post(
    'http://localhost:3000/v1/chat/completions',
    headers={'Content-Type': 'application/json'},
    json={
        'model': 'gpt-4',
        'messages': [{'role': 'user', 'content': 'Hello!'}],
        'stream': True
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = line[6:].decode('utf-8')
        if data == '[DONE]':
            break

        chunk = json.loads(data)
        content = chunk['choices'][0]['delta'].get('content', '')
        if content:
            print(content, end='', flush=True)
```

## Error Handling

The provider maps OpenAI API errors to appropriate HTTP status codes:

- **400 Bad Request**: Invalid parameters or malformed request
- **401 Unauthorized**: Invalid API key
- **403 Forbidden**: Insufficient permissions or quota exceeded
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: OpenAI API errors or network issues
- **Stream Errors**: Reported within SSE stream for streaming requests

## Model Compatibility

The provider filters the model list to only include chat-compatible models. Supported model families include:

- **GPT-3.5**: `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`
- **GPT-4**: `gpt-4`, `gpt-4-turbo`, `gpt-4-32k`
- **Other**: Any model containing "gpt", "turbo", or "davinci" in the name

## Performance Considerations

### Intelligent Defaults
The provider automatically configures sensible defaults for non-production environments:
- **Request timeout**: 30 seconds
- **Connect timeout**: 10 seconds
- **Keep-alive duration**: 30 seconds
- **Max idle connections**: 10
- **Max retries**: 3 with 100ms backoff

### Production Tuning
For production deployments, consider increasing timeouts and connection limits:
- **Recommended timeout**: 60+ seconds for GPT-4 (120+ for streaming)
- **Max idle connections**: 20+ for high traffic
- **Keep-alive duration**: 60+ seconds for sustained load
- **Streaming considerations**: Longer timeouts for streaming connections

### Rate Limiting
OpenAI enforces rate limits based on your API tier. The provider will return HTTP 429 when limits are exceeded.

## Security Best Practices

1. **Environment Variables**: Store API keys in environment variables, not config files
2. **Restricted Keys**: Use API keys with minimal required permissions
3. **Network Security**: Use HTTPS for all communication
4. **Key Rotation**: Regularly rotate API keys
5. **Monitoring**: Monitor API usage and costs

## Troubleshooting

### Common Issues

**Authentication Failed (401)**
```bash
# Check API key
export INFERENCE_INFERENCE_API_KEY="sk-your-valid-key"
```

**Rate Limit Exceeded (429)**
```bash
# Increase timeout or implement retry logic
export INFERENCE_INFERENCE_HTTP_TIMEOUT_SECS="60"
```

**Connection Timeout**
```bash
# Increase connection timeout
export INFERENCE_INFERENCE_HTTP_CONNECT_TIMEOUT_SECS="15"
```

### Debugging

Enable debug logging to see detailed request/response information:

```bash
RUST_LOG=debug cargo run
```

This will show:
- HTTP request details
- OpenAI API responses
- Streaming chunk processing
- Error details
- Performance metrics

**Streaming Debug Example:**
```bash
# Test streaming with verbose output
curl -v -N -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Count to 5"}], "stream": true}'
```

## Cost Optimization

1. **Model Selection**: Use `gpt-3.5-turbo` for cost-effective applications
2. **Token Limits**: Set appropriate `max_tokens` to control costs
3. **Monitoring**: Track token usage via response metadata
4. **Caching**: Implement response caching for repeated queries

## Integration Examples

### Docker Compose

```yaml
version: '3.8'
services:
  inference-server:
    build: .
    ports:
      - "3000:3000"
    environment:
      - INFERENCE_INFERENCE_PROVIDER=openai
      - INFERENCE_INFERENCE_API_KEY=${OPENAI_API_KEY}
      - INFERENCE_INFERENCE_DEFAULT_MODEL=gpt-3.5-turbo
      - RUN_ENV=production
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-config
data:
  production.yaml: |
    inference:
      provider: openai
      default_model: "gpt-4"
      http:
        timeout_secs: 60
        max_idle_connections: 20
---
apiVersion: v1
kind: Secret
metadata:
  name: inference-secrets
type: Opaque
stringData:
  api-key: "sk-your-openai-api-key"
```

## Monitoring and Metrics

The provider includes built-in observability:

- **Request latency**: Tracked per request (streaming and non-streaming)
- **Error rates**: Categorized by error type
- **Token usage**: Available in response metadata and final streaming chunks
- **Streaming metrics**: Track stream duration and chunk delivery
- **Health checks**: Regular connectivity verification

Access metrics via the health endpoint:
```bash
curl http://localhost:3000/health
```

### Streaming Performance Monitoring

**Key Metrics:**
- Time to first token (TTFT)
- Tokens per second during streaming
- Stream completion rate
- Connection duration for streaming requests

**Example Usage Tracking:**
```json
{
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  },
  "stream_stats": {
    "duration_ms": 2500,
    "chunks_sent": 12,
    "avg_chunk_interval_ms": 208
  }
}
```