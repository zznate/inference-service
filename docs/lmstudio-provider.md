# LM Studio Provider - API Reference

The LM Studio provider enables **seamless integration** with LM Studio's local model server, providing OpenAI-compatible chat completions with support for any models loaded in LM Studio. This provider offers full streaming support, automatic model validation, and comprehensive parameter handling for local LLM inference.

Though LM Studio supports the OpenAI API, our provider allows for all the specific extensions such as `microstat_mode` and `cache_prompt` to be used. 

## Overview

LM Studio is a desktop application that allows you to run Large Language Models (LLMs) locally on your machine. The LM Studio provider connects to LM Studio's OpenAI-compatible API server, enabling you to:

- Run models locally without cloud dependencies
- Maintain data privacy with on-premise inference
- Use any GGUF/GGML models compatible with LM Studio
- Stream responses in real-time using Server-Sent Events
- Switch between models dynamically

## Quick Start

### Prerequisites

1. **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/)
2. **Download a Model**: Use LM Studio's built-in model browser
3. **Start the Server**: In LM Studio, go to the "Server" tab and click "Start Server"
4. **Note the Port**: Default is `http://localhost:1234`

### Minimal Configuration

```yaml
inference:
  provider: lmstudio
  base_url: "http://localhost:1234"  # LM Studio default
  default_model: "local-model"        # Model loaded in LM Studio
```

### Full Configuration

```yaml
inference:
  provider: lmstudio
  base_url: "http://localhost:1234"
  default_model: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
  allowed_models:                    # Optional: restrict to specific models
    - "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    - "TheBloke/Llama-2-7B-Chat-GGUF"
  http:                              # Optional - has smart defaults
    timeout_secs: 60                # Longer for local models
    connect_timeout_secs: 10
    keep_alive_secs: 30
    max_idle_connections: 10
```

## Environment Variables

Override configuration using environment variables:

```bash
# Core settings
export INFERENCE_INFERENCE_PROVIDER="lmstudio"
export INFERENCE_INFERENCE_BASE_URL="http://localhost:1234"
export INFERENCE_INFERENCE_DEFAULT_MODEL="local-model"

# HTTP configuration
export INFERENCE_INFERENCE_HTTP_TIMEOUT_SECS="60"
export INFERENCE_INFERENCE_HTTP_CONNECT_TIMEOUT_SECS="10"
```

## API Usage Examples

### Basic Chat Completion

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ]
  }'
```

### Advanced Parameters

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to sort a list"}
    ],
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
    "stop": ["```", "END"],
    "seed": 42
  }'
```

### Streaming Responses

Enable real-time token streaming with the `stream` parameter:

```bash
curl -N -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true,
    "max_tokens": 200
  }'
```

**Response Format (Server-Sent Events):**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"local-model","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"local-model","choices":[{"index":0,"delta":{"content":"Once"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"local-model","choices":[{"index":0,"delta":{"content":" upon"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"local-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":50,"total_tokens":60}}

data: [DONE]
```

### List Available Models

```bash
curl http://localhost:3000/v1/models
```

This returns all models currently loaded in LM Studio.

### Health Check

```bash
curl http://localhost:3000/health
```

## Supported Parameters

The LM Studio provider supports all standard OpenAI parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | Required | Model loaded in LM Studio |
| `messages` | array | Required | Conversation messages |
| `max_tokens` | integer | 100 | Maximum tokens to generate |
| `temperature` | number | 0.7 | Controls randomness (0.0-2.0) |
| `top_p` | number | Optional | Nucleus sampling (0.0-1.0) |
| `frequency_penalty` | number | Optional | Reduce repetition (-2.0-2.0) |
| `presence_penalty` | number | Optional | Encourage new topics (-2.0-2.0) |
| `stop` | array/string | Optional | Stop sequences |
| `seed` | integer | Optional | For reproducible outputs |
| **`stream`** | **boolean** | **false** | **Enable SSE streaming** |

## Model Management

### Loading Models in LM Studio

1. Open LM Studio
2. Go to the "Search" tab
3. Search for models (e.g., "Mistral", "Llama")
4. Download desired models
5. Go to "Server" tab
6. Select model from dropdown
7. Start server

### Model Naming

LM Studio uses the full model path as the model name:
- Example: `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
- You can use any unique substring when configuring

### Model Validation

The provider validates that the requested model matches what LM Studio actually uses. If there's a mismatch, it returns an error with the available model.

## Streaming Support

The LM Studio provider includes **full streaming support** compatible with OpenAI's SSE format:

### Features
- Real-time token generation
- Server-Sent Events format
- Graceful error handling in streams
- Usage statistics in final chunk
- `[DONE]` marker compatibility

### Client Implementation Example

**JavaScript:**
```javascript
const response = await fetch('/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'local-model',
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
        process.stdout.write(content);
      }
    }
  }
}
```

## Performance Considerations

### Local Inference
- **CPU Models**: Expect slower generation than GPU
- **Memory Usage**: Models require 4-8GB RAM typically
- **First Token Latency**: Can be 1-5 seconds for large models

### Optimization Tips
1. **Use Quantized Models**: Q4_K_M or Q5_K_M for balance
2. **Adjust Context Length**: Lower context = faster inference
3. **Set Appropriate Timeouts**: Local models need longer timeouts
4. **Consider Model Size**: 7B models are faster than 13B+

### Recommended Settings

**For 7B Models:**
```yaml
http:
  timeout_secs: 60
  connect_timeout_secs: 10
```

**For 13B+ Models:**
```yaml
http:
  timeout_secs: 120
  connect_timeout_secs: 15
```

## Error Handling

The provider maps LM Studio errors appropriately:

- **400 Bad Request**: Invalid parameters
- **404 Not Found**: Model not loaded in LM Studio
- **500 Internal Server Error**: LM Studio server issues
- **502 Bad Gateway**: Cannot connect to LM Studio
- **504 Gateway Timeout**: Response generation timeout

## Troubleshooting

### Common Issues

**Connection Refused**
```bash
# Ensure LM Studio server is running
# Check the Server tab in LM Studio
```

**Model Not Found**
```bash
# List available models
curl http://localhost:1234/v1/models

# Use exact model name from the list
```

**Timeout Errors**
```bash
# Increase timeout for large models
export INFERENCE_INFERENCE_HTTP_TIMEOUT_SECS="120"
```

**Out of Memory**
- Close other applications
- Use smaller or more quantized models
- Reduce context length in LM Studio settings

### Debugging

Enable debug logging:
```bash
RUST_LOG=debug cargo run
```

Check LM Studio logs:
- View console output in LM Studio's Server tab
- Check for model loading errors
- Monitor memory usage

## Integration Examples

### Docker Compose

```yaml
version: '3.8'
services:
  lm-studio:
    image: lmstudio/server:latest  # If containerized
    ports:
      - "1234:1234"
    volumes:
      - ./models:/models

  inference-server:
    build: .
    ports:
      - "3000:3000"
    environment:
      - INFERENCE_INFERENCE_PROVIDER=lmstudio
      - INFERENCE_INFERENCE_BASE_URL=http://lm-studio:1234
      - INFERENCE_INFERENCE_DEFAULT_MODEL=local-model
    depends_on:
      - lm-studio
```

### SystemD Service

```ini
[Unit]
Description=Inference Server with LM Studio
After=network.target

[Service]
Type=simple
User=inference
Environment="INFERENCE_INFERENCE_PROVIDER=lmstudio"
Environment="INFERENCE_INFERENCE_BASE_URL=http://localhost:1234"
ExecStart=/usr/local/bin/inference-server
Restart=always

[Install]
WantedBy=multi-user.target
```

## Best Practices

1. **Model Selection**: Choose models appropriate for your hardware
2. **Resource Management**: Monitor RAM and CPU usage
3. **Timeout Configuration**: Set realistic timeouts for your hardware
4. **Error Handling**: Implement retry logic for transient failures
5. **Model Preloading**: Load models in LM Studio before starting the inference server
6. **Streaming for UX**: Use streaming for better user experience with slow models

## Comparison with Other Providers

| Feature | LM Studio | OpenAI | Mock |
|---------|-----------|---------|------|
| **Location** | Local | Cloud | Local |
| **Privacy** | Full | Limited | Full |
| **Cost** | Free | Per-token | Free |
| **Speed** | Hardware-dependent | Fast | Instant |
| **Model Selection** | Any GGUF/GGML | GPT models | N/A |
| **Streaming** | ✅ | ✅ | ✅ |
| **Internet Required** | ❌ | ✅ | ❌ |

## Monitoring and Metrics

- **Request Duration**: Track with debug logs
- **Token Generation Rate**: Monitor in streaming responses
- **Model Load Time**: Check LM Studio startup logs
- **Memory Usage**: Monitor system resources
- **Queue Length**: Check concurrent request handling

Access health metrics:
```bash
curl http://localhost:3000/health
```

Returns provider status and HTTP configuration details.