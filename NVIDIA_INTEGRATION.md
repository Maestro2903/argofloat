# NVIDIA API Integration for FLOAT CHAT

This document describes the NVIDIA API integration that replaces the local Ollama dependency with cloud-based NVIDIA models.

## Overview

The FLOAT CHAT application has been modified to route all Ollama-compatible requests through NVIDIA's API endpoints. This provides access to powerful cloud-based models without requiring local Ollama installation.

## Changes Made

### Backend Changes

1. **Modified Ollama Router** (`backend/open_webui/routers/ollama.py`):
   - Replaced local Ollama connections with NVIDIA API calls
   - Added NVIDIA model definitions with API key mappings
   - Implemented OpenAI-to-Ollama format conversion for compatibility
   - Added proper error handling and logging

2. **Available NVIDIA Models**:
   - `qwen/qwen3-coder-480b-a35b-instruct` - Qwen3 Coder 480B A35B Instruct
   - `moonshotai/kimi-k2-instruct-0905` - Moonshot AI Kimi K2 Instruct 0905
   - `deepseek-ai/deepseek-r1-0528` - DeepSeek R1 0528
   - `meta/llama-3.1-nemotron-70b-instruct` - Llama 3.1 Nemotron 70B Instruct
   - `meta/llama-3.1-405b-instruct` - Llama 3.1 405B Instruct

3. **API Endpoints Updated**:
   - `/ollama/api/chat` - Chat completions
   - `/ollama/api/generate` - Text completions
   - `/ollama/api/embeddings` - Text embeddings
   - `/ollama/api/tags` - List available models
   - `/ollama/api/version` - API version info
   - `/ollama/api/pull` - Model availability check
   - `/ollama/api/show` - Model information
   - `/ollama/config` - Configuration management

### Frontend Changes

1. **Fixed TipTap Extension Warnings**:
   - Updated `RichTextInput.svelte` to properly disable conflicting extensions
   - Added `listKeymap: false` to prevent duplication with ListKit

2. **Frontend Compatibility**:
   - No changes needed to frontend API calls
   - All existing Ollama API calls now route through NVIDIA
   - Maintains backward compatibility with existing UI

## Configuration

### Environment Variables

Set the following environment variables to configure NVIDIA API access:

```bash
# Primary NVIDIA API key (fallback for all models)
NVIDIA_API_KEY=your_nvidia_api_key_here

# Model-specific API keys (optional, will use NVIDIA_API_KEY if not set)
NVIDIA_QWEN_API_KEY=your_qwen_specific_key
NVIDIA_KIMI_API_KEY=your_kimi_specific_key
NVIDIA_DEEPSEEK_API_KEY=your_deepseek_specific_key
```

### Getting NVIDIA API Keys

1. Visit [NVIDIA API Catalog](https://integrate.api.nvidia.com/)
2. Sign up for an account
3. Navigate to the specific model you want to use
4. Generate an API key for that model
5. Set the appropriate environment variable

### Model Selection

Models are automatically filtered based on available API keys. Only models with valid API keys will appear in the UI.

## Usage

### Starting the Application

1. Set your NVIDIA API keys as environment variables
2. Start the application normally:
   ```bash
   # Backend
   cd backend
   python -m open_webui.main

   # Frontend (if running separately)
   cd frontend
   npm run dev
   ```

### Using Models

1. Models will appear in the model selector automatically
2. Select any available NVIDIA model
3. Chat and completions work exactly as before
4. All requests are routed through NVIDIA's cloud API

## API Format Conversion

The integration handles format conversion between Ollama and OpenAI formats:

### Chat Completions
- **Input**: Ollama chat format
- **Processing**: Converts to OpenAI chat completions format
- **Output**: Converts back to Ollama format for frontend compatibility

### Text Completions
- **Input**: Ollama generate format with prompt
- **Processing**: Converts to OpenAI chat format with user message
- **Output**: Ollama-compatible response format

### Embeddings
- **Input**: Ollama embeddings format
- **Processing**: Converts to OpenAI embeddings format
- **Output**: Ollama-compatible embeddings response

## Error Handling

The integration includes comprehensive error handling:

- **Missing API Keys**: Clear error messages indicating which environment variable to set
- **Model Not Found**: Automatic fallback to first available model
- **API Errors**: Proper error propagation with NVIDIA API error details
- **Network Issues**: Timeout and connection error handling

## Streaming Support

Full streaming support is maintained:
- Real-time response streaming
- Proper SSE (Server-Sent Events) formatting
- Compatible with existing frontend streaming logic
- Support for reasoning content (DeepSeek R1 models)

## Limitations

1. **Model Management**: 
   - Cannot delete cloud-based models
   - Cannot push models to NVIDIA
   - Pull operations are simulated (models are always "available")

2. **Local Features**:
   - No local model storage
   - No offline operation
   - Requires internet connection

3. **API Limits**:
   - Subject to NVIDIA API rate limits
   - Requires valid API keys for each model

## Troubleshooting

### Common Issues

1. **"No NVIDIA models available"**:
   - Ensure at least one NVIDIA API key is set
   - Check that environment variables are properly loaded

2. **"Model not found" errors**:
   - Verify the model name matches available NVIDIA models
   - Check that the specific model's API key is configured

3. **API connection errors**:
   - Verify internet connectivity
   - Check that API keys are valid and not expired
   - Ensure NVIDIA API service is accessible

### Debug Logging

Enable debug logging to troubleshoot issues:
```bash
export LOG_LEVEL=DEBUG
```

This will provide detailed logs of:
- API requests and responses
- Model selection logic
- Error details and stack traces

## Migration from Local Ollama

If migrating from a local Ollama setup:

1. **Backup**: Save any custom models or configurations
2. **Environment**: Set NVIDIA API keys
3. **Testing**: Test with a simple chat to verify connectivity
4. **Cleanup**: Remove local Ollama installation if no longer needed

## Security Considerations

1. **API Keys**: Store API keys securely as environment variables
2. **Network**: All communication uses HTTPS with NVIDIA's secure endpoints
3. **Data**: Chat data is sent to NVIDIA's cloud services
4. **Compliance**: Ensure usage complies with your organization's data policies

## Support

For issues related to:
- **NVIDIA API**: Contact NVIDIA support or check their documentation
- **Integration**: Check the application logs and this documentation
- **Models**: Refer to NVIDIA's model-specific documentation

## Future Enhancements

Potential improvements for future versions:
- Dynamic model discovery from NVIDIA API
- Model-specific parameter optimization
- Enhanced error recovery and retry logic
- Support for additional NVIDIA model types
- Integration with NVIDIA's fine-tuning services
