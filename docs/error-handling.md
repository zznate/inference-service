# Error Handling

The inference service provides OpenAI-compatible error responses for all error conditions. This ensures compatibility with OpenAI SDKs and provides clear, structured error information.

## Error Response Format

All errors follow the OpenAI API format:

```json
{
  "error": {
    "message": "Human-readable error description",
    "type": "error_type_identifier",
    "param": "parameter_name_if_applicable",
    "code": "specific_error_code_if_applicable"
  }
}
```

### Fields

- **message** (string, required): A human-readable description of the error
- **type** (string, required): The error type identifier (see Error Types below)
- **param** (string, optional): The parameter that caused the error (for validation errors)
- **code** (string, optional): A specific error code for programmatic handling

## Error Types

### invalid_request_error (400 Bad Request)

Returned when the request is malformed or contains invalid parameters.

**Common Causes:**
- Empty messages array
- Invalid parameter values (temperature, top_p, etc.)
- Model not in allowed list
- Unsupported features

**Example:**

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [],
    "model": "gpt-3.5-turbo"
  }'
```

**Response:**
```json
{
  "error": {
    "message": "Messages array cannot be empty",
    "type": "invalid_request_error",
    "param": "messages",
    "code": null
  }
}
```

### authentication_error (401 Unauthorized)

Returned when authentication fails (typically with the underlying provider).

**Example Response:**
```json
{
  "error": {
    "message": "Invalid API key provided",
    "type": "authentication_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

### permission_error (403 Forbidden)

Returned when the request is valid but the user lacks permission or has exceeded quota.

**Example Response:**
```json
{
  "error": {
    "message": "You exceeded your current quota",
    "type": "permission_error",
    "param": null,
    "code": "insufficient_quota"
  }
}
```

### rate_limit_error (429 Too Many Requests)

Returned when rate limits are exceeded.

**Example Response:**
```json
{
  "error": {
    "message": "Rate limit exceeded. Please try again later",
    "type": "rate_limit_error",
    "param": null,
    "code": "rate_limit_exceeded"
  }
}
```

### api_error (500 Internal Server Error / 502 Bad Gateway)

Returned for server-side errors or provider connection issues.

**Example Response:**
```json
{
  "error": {
    "message": "Failed to connect to inference provider: Connection refused",
    "type": "api_error",
    "param": null,
    "code": "provider_connection_failed"
  }
}
```

### timeout_error (504 Gateway Timeout)

Returned when a request times out.

**Example Response:**
```json
{
  "error": {
    "message": "Request to inference provider timed out",
    "type": "timeout_error",
    "param": null,
    "code": "provider_timeout"
  }
}
```

## Validation Errors

All validation errors return HTTP 400 with `type: "invalid_request_error"`.

| Parameter | Error Condition | Example Message |
|-----------|----------------|-----------------|
| `messages` | Empty array | Messages array cannot be empty |
| `messages` | All null content | At least one message must have content |
| `max_tokens` | < 1 or > 128000 | Max tokens must be between 1 and 128000, got 200000 |
| `temperature` | < 0.0 or > 2.0 | Temperature must be between 0.0 and 2.0, got 3.0 |
| `top_p` | < 0.0 or > 1.0 | Top-p must be between 0.0 and 1.0, got 1.5 |
| `frequency_penalty` | < -2.0 or > 2.0 | Frequency penalty must be between -2.0 and 2.0, got 3.0 |
| `presence_penalty` | < -2.0 or > 2.0 | Presence penalty must be between -2.0 and 2.0, got 3.0 |
| `top_logprobs` | > 20 | Top logprobs must be between 0 and 20, got 25 |
| `n` | < 1 or > 10 | N (number of choices) must be between 1 and 10, got 15 |
| `model` | Not in allowed list | Model 'gpt-5' is not in the allowed list |
| `stream` | Not supported | Streaming is not supported by the current provider |
| `response_format` | Invalid type | Response format type must be 'text' or 'json_object' |
| `logit_bias` | Invalid value | Invalid logit bias for token '12345': Value out of range |

### Validation Error Examples

**Invalid Temperature:**
```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 3.0
  }'
```

Response:
```json
{
  "error": {
    "message": "Temperature must be between 0.0 and 2.0, got 3.0",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": null
  }
}
```

**Model Not Allowed:**
```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Response:
```json
{
  "error": {
    "message": "Model 'gpt-5' is not in the allowed list. Available models: gpt-3.5-turbo, gpt-4",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

## Provider Errors

Errors from the underlying inference provider are mapped to appropriate OpenAI error types.

| Provider Error | HTTP Status | Error Type | Code |
|---------------|-------------|------------|------|
| Connection failed | 502 | `api_error` | `provider_connection_failed` |
| Invalid response | 500 | `api_error` | `provider_invalid_response` |
| Authentication (401) | 401 | `authentication_error` | `invalid_api_key` |
| Permission (403) | 403 | `permission_error` | `insufficient_quota` |
| Rate limit (429) | 429 | `rate_limit_error` | `rate_limit_exceeded` |
| Model not available | 400 | `invalid_request_error` | `model_not_found` |
| Timeout | 504 | `timeout_error` | `provider_timeout` |
| Configuration error | 500 | `api_error` | `configuration_error` |
| Stream error | 500 | `api_error` | `stream_error` |
| Invalid extension | 400 | `invalid_request_error` | `invalid_extension` |

### Provider Error Examples

**Connection Failed:**
```json
{
  "error": {
    "message": "Failed to connect to inference provider: Connection refused",
    "type": "api_error",
    "param": null,
    "code": "provider_connection_failed"
  }
}
```

**Provider Timeout:**
```json
{
  "error": {
    "message": "Request to inference provider timed out",
    "type": "timeout_error",
    "param": null,
    "code": "provider_timeout"
  }
}
```

## Streaming Errors

When streaming is enabled (`stream: true`), errors are sent as Server-Sent Events (SSE) in the same OpenAI-compatible format:

```
data: {"error":{"message":"Stream error occurred","type":"api_error","param":null,"code":"stream_error"}}
```

**Example:**
```bash
curl -N -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

If an error occurs during streaming:
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk",...}

data: {"error":{"message":"Connection lost","type":"api_error","code":"stream_error"}}
```

## Error Handling Best Practices

### Client-Side Handling

```javascript
try {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    const errorData = await response.json();
    const error = errorData.error;
    
    // Handle specific error types
    switch (error.type) {
      case 'invalid_request_error':
        console.error(`Invalid ${error.param}: ${error.message}`);
        break;
      case 'rate_limit_error':
        console.error('Rate limited, retry after delay');
        break;
      case 'timeout_error':
        console.error('Request timed out, retry');
        break;
      default:
        console.error(`Error: ${error.message}`);
    }
    
    return;
  }

  const data = await response.json();
  // Process successful response
} catch (err) {
  console.error('Network error:', err);
}
```

### Python Client

```python
import requests

try:
    response = requests.post(
        'http://localhost:3000/v1/chat/completions',
        json=request_data
    )
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    error = e.response.json()['error']
    
    if error['type'] == 'invalid_request_error':
        print(f"Invalid {error.get('param')}: {error['message']}")
    elif error['type'] == 'rate_limit_error':
        print("Rate limited, waiting before retry...")
        time.sleep(60)
    else:
        print(f"Error: {error['message']}")
```

## Debugging

Enable debug logging to see detailed error information:

```bash
RUST_LOG=debug cargo run
```

This will show:
- Request validation details
- Provider communication
- Error transformation
- HTTP status codes

## OpenAI SDK Compatibility

The error format is fully compatible with OpenAI's official SDKs:

**Python:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/v1")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}]
    )
except openai.BadRequestError as e:
    print(f"Validation error: {e.message}")
except openai.RateLimitError as e:
    print(f"Rate limited: {e.message}")
except openai.APIError as e:
    print(f"API error: {e.message}")
```

**Node.js:**
```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:3000/v1'
});

try {
  const response = await client.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages: [{ role: 'user', content: 'Hello' }]
  });
} catch (error) {
  if (error instanceof OpenAI.BadRequestError) {
    console.error('Validation error:', error.message);
  } else if (error instanceof OpenAI.RateLimitError) {
    console.error('Rate limited:', error.message);
  } else {
    console.error('API error:', error.message);
  }
}
```

## Testing Error Responses

Run the test suite to verify error handling:

```bash
cd services/inference-server
cargo test
```

All error types have comprehensive test coverage ensuring correct:
- Error type mapping
- HTTP status codes
- Parameter identification
- Error codes
- Message formatting

## References

- [OpenAI Error Codes Documentation](https://platform.openai.com/docs/guides/error-codes)
- [OpenAI API Reference - Errors](https://platform.openai.com/docs/api-reference/errors)
- [Implementation Plan](./error-handling-plan.md)
- [Architecture Documentation](./error-handling-architecture.md)
- [Quick Reference Guide](./error-handling-quick-reference.md)