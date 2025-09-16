# Mock Provider

The mock provider is an inference provider that returns mock responses for inference requests. It is used for testing and development purposes. The default configuration loads mock responses from the `mocks` directory. 

This provider was designed with the following goals in mind:

- No External Dependencies: Run tests without LM Studio, Ollama, etc or any external service
- Deterministic Testing: Exact, repeatable responses for CI/CD
- Load Testing: Test your application's handling of AI response delays
- Development: Work on UI/integration without running expensive models
- Multiple Scenarios: Different response patterns for different test cases (e.g. random, sequential, first)

## Configuration

The mock responses can be configured in YAML files with the following format:

```yaml
responses:
  - text: "This is the default mock response. Your input has been received and processed."
    model_used: "mock-default"
    prompt_tokens: 10
    completion_tokens: 12
    total_tokens: 22
    finish_reason: "stop"

settings:
  mode: first
```

The `responses` key is a list of mock responses. Each response can have the following fields:

- `text`: The text of the response.
- `model_used`: The model used to generate the response.
- `prompt_tokens`: The number of tokens in the prompt.
- `completion_tokens`: The number of tokens in the completion.
- `total_tokens`: The total number of tokens in the response.
- `finish_reason`: The reason the response was finished.
- `delay_ms`: The delay in milliseconds before the response is returned.

The `settings` key is a map of settings for the mock provider. It can have the following fields:

- `mode`: The mode for selecting responses. Can be `first`, `sequential`, or `random`.

Have a look through the default mock responses in the `mocks` directory to see how they are structured.

## Usage Examples

To use the mock provider, set the `RUN_ENV` environment variable to `mock`:

```bash
RUN_ENV=mock cargo run
```

This will load the mock responses from the `mocks` directory. The default configuration will load the `default.yaml` file.

You can also override the default configuration by setting the `INFERENCE_MOCK_RESPONSES_DIR` environment variable to the path of your mock responses directory:

```bash
INFERENCE_MOCK_RESPONSES_DIR=/path/to/mock/responses cargo run
```

## Example Docker File
In a containerised environment, your dockerfile would look something like:

```dockerfile
FROM rust:1.88 as builder
# ... build steps ...

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/inference-server /usr/local/bin/
COPY ./mocks /app/mocks

ENV RUN_ENV=mock
ENV INFERENCE_INFERENCE_RESPONSES_DIR=/app/mocks

CMD ["inference-server"]
```



