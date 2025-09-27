# NVIDIA API Integration Setup Guide

This guide explains how to set up and use NVIDIA API endpoints in FLOAT CHAT.

## Overview

FLOAT CHAT now supports NVIDIA's API endpoints alongside Ollama and OpenAI. This integration provides access to powerful NVIDIA-hosted models including:

- **Qwen3 Coder 480B A35B Instruct** - Advanced coding model
- **Moonshot AI Kimi K2 Instruct** - Conversational AI model  
- **DeepSeek R1** - Reasoning-capable model with thinking process
- **Llama 3.1 Nemotron 70B** - Large language model
- **Llama 3.1 405B** - Ultra-large language model

## Environment Setup

### 1. Configure Environment Variables

Create or update your `.env` file with the following NVIDIA API keys:

```bash
# NVIDIA API Configuration
NVIDIA_API_KEY=''  # Fallback/default key
NVIDIA_API_BASE_URL='https://integrate.api.nvidia.com/v1'
ENABLE_NVIDIA_API=true

# Model-specific NVIDIA API Keys
NVIDIA_QWEN_API_KEY='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'
NVIDIA_KIMI_API_KEY='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'
NVIDIA_DEEPSEEK_API_KEY='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'
```

### 2. API Key Mapping

Each model uses a specific API key:

| Model | Environment Variable | API Key |
|-------|---------------------|---------|
| Qwen3 Coder 480B | `NVIDIA_QWEN_API_KEY` | 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 |
| Moonshot AI Kimi K2 | `NVIDIA_KIMI_API_KEY` | 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 |
| DeepSeek R1 | `NVIDIA_DEEPSEEK_API_KEY` | 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 |
| Llama Models | `NVIDIA_API_KEY` | (Use fallback key) |

## Features

### 1. Model Selection
- NVIDIA models appear in the model selector with "nvidia/" prefix
- Models are automatically detected and available in the UI
- Each model uses its specific API key automatically

### 2. Streaming Support
- Full streaming support for real-time responses
- Special handling for DeepSeek R1 reasoning content
- Compatible with existing chat interface

### 3. Reasoning Content (DeepSeek R1)
The DeepSeek R1 model provides reasoning content showing its thinking process:

```python
# Example usage in Python
from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-r1-0528",
  messages=[{"role":"user","content":"Explain quantum computing"}],
  temperature=0.6,
  top_p=0.7,
  max_tokens=4096,
  stream=True
)

for chunk in completion:
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(f"[Thinking] {reasoning}", end="")
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
```

### 4. Backend Integration
- Automatic model detection and API key routing
- Error handling and fallback mechanisms
- Compatible with existing OpenAI-style API calls

## Usage

### 1. Start the Application

```bash
# Frontend
npm run dev

# Backend  
cd backend
python -m uvicorn open_webui.main:app --reload --host 0.0.0.0 --port 8080
```

### 2. Select NVIDIA Models

1. Open FLOAT CHAT in your browser
2. Click on the model selector dropdown
3. Choose any model with "nvidia/" prefix:
   - `nvidia/qwen3-coder-480b-a35b-instruct`
   - `nvidia/moonshotai-kimi-k2-instruct-0905`
   - `nvidia/deepseek-r1-0528`
   - `nvidia/llama-3.1-nemotron-70b-instruct`
   - `nvidia/llama-3.1-405b-instruct`

### 3. Chat with NVIDIA Models

The models work exactly like other models in FLOAT CHAT:
- Type your message and press Enter
- Streaming responses appear in real-time
- DeepSeek R1 shows reasoning process (if supported by frontend)

## API Endpoints

### Available Endpoints

- `GET /nvidia/api/tags` - List available NVIDIA models
- `POST /nvidia/api/chat` - Chat completions (streaming/non-streaming)
- `POST /nvidia/api/generate` - Text generation (Ollama-compatible)
- `POST /nvidia/api/embeddings` - Generate embeddings
- `GET /nvidia/api/version` - Get NVIDIA API version info

### Example API Call

```bash
curl -X POST "http://localhost:8080/nvidia/api/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "model": "nvidia/qwen3-coder-480b-a35b-instruct",
    "messages": [{"role": "user", "content": "Write a Python function"}],
    "stream": true
  }'
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure the correct environment variable is set
   - Check that the API key is valid and not expired

2. **Model Not Available**
   - Verify `ENABLE_NVIDIA_API=true` in your `.env` file
   - Restart the backend after changing environment variables

3. **Connection Errors**
   - Check internet connectivity
   - Verify NVIDIA API base URL is correct
   - Ensure API keys have proper permissions

### Debug Mode

Enable debug logging by setting:
```bash
LOG_LEVEL=DEBUG
```

## Security Notes

- **Never commit API keys to version control**
- Store API keys in environment variables only
- Use different keys for development and production
- Regularly rotate API keys for security

## Model Specifications

### Qwen3 Coder 480B A35B Instruct
- **Purpose**: Advanced code generation and analysis
- **Strengths**: Programming, debugging, code explanation
- **Parameters**: Temperature: 0.7, Top-p: 0.8, Max tokens: 4096

### Moonshot AI Kimi K2 Instruct
- **Purpose**: Conversational AI and general tasks
- **Strengths**: Natural dialogue, reasoning, multilingual
- **Parameters**: Temperature: 0.6, Top-p: 0.9, Max tokens: 4096

### DeepSeek R1
- **Purpose**: Reasoning and problem-solving
- **Strengths**: Step-by-step thinking, complex reasoning
- **Special**: Provides reasoning_content showing thought process
- **Parameters**: Temperature: 0.6, Top-p: 0.7, Max tokens: 4096

## Support

For issues related to:
- **NVIDIA API**: Contact NVIDIA support
- **FLOAT CHAT Integration**: Check application logs and GitHub issues
- **Model-specific problems**: Refer to NVIDIA model documentation
