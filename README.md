# OpenAI Compatible API Server For StackFlow

## Overview
This server provides an OpenAI-compatible API interface supporting multiple AI model backends including LLMs, vision models, speech synthesis (TTS), and speech recognition (ASR).

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
python3 api_server.py 
```

## Supported Endpoints

### Chat Completions
- **Endpoint**: `POST /v1/chat/completions`
- **Request Format**: OpenAI-compatible chat completion request
- **Streaming**: Supported

### Text Completions
- **Endpoint**: `POST /v1/completions`
- **Request Format**: OpenAI-compatible completion request
- **Streaming**: Supported

### Speech Synthesis (TTS)
- **Endpoint**: `POST /v1/audio/speech`
- **Parameters**:
  - `model`: TTS model name
  - `input`: Text to synthesize
  - `voice`: Voice type
  - `response_format`: Audio format (mp3, wav, etc.)

### Speech Recognition (ASR)
- **Transcription**: `POST /v1/audio/transcriptions`
  - Converts speech to text in the same language
- **Translation**: `POST /v1/audio/translations`
  - Converts speech to English text
- **Parameters**:
  - `file`: Audio file
  - `model`: ASR model name
  - `language` (transcription only): Source language
  - `prompt`: Optional prompt

### List Models
- **Endpoint**: `GET /v1/models`
- **Returns**: List of available models

## FAQ

### Q: Why am I getting "Unsupported model" errors?
A: The model name must exactly match one of the configured models in your config file.

### Q: How do I enable streaming responses?
A: Set `"stream": true` in your request body for chat/completion endpoints.

### Q: What audio formats are supported for ASR?
A: The supported formats depend on your ASR backend implementation.

### Q: How do I manage model memory usage?
A: The server implements a pool system for LLM models - adjust `pool_size` in the config to control concurrent instances.

## Troubleshooting

- **Logs**: Check server logs for detailed error messages
- **Model Initialization**: Verify all required backend services are running
- **Configuration**: Double-check model names and parameters in config.yaml

## Example Requests

### Chat Completion
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_KEY" \
-d '{
  "model": "qwen2.5-0.5B-p256-ax630c",
  "messages": [{"role": "user", "content": "Hello!"}],
  "temperature": 0.7
}'
```

### Speech Synthesis
```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_KEY" \
-d '{
  "model": "melotts",
  "input": "Hello world!",
  "voice": "alloy"
}'
```

## Required Libraries:

- [StackFlow](https://github.com/m5stack/StackFlow)

## License

- [M5Module-LLM_OpenAI_API- MIT](LICENSE)
