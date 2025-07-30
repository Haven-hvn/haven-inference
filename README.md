# SmolVLM GGUF API - Hivenet Deployment

This project provides a Docker deployment optimized for Hivenet compute.

## Overview

The SmolVLM API serves an OpenAI-compatible endpoint for vision-language model inference using:
- **Model**: SmolVLM v1.8B GGUF format
- **Backend**: llama-cpp-python with CUDA acceleration
- **API**: FastAPI with OpenAI-compatible endpoints
- **Base Image**: Ubuntu 24.04 LTS with CUDA 12.6 and JupyterLab 4.2.5

## Quick Start

### Prerequisites
- Docker with GPU support (nvidia-docker2)
- NVIDIA GPU with CUDA capability
- At least 8GB GPU memory
- 30GB+ storage for model files

### Build and Run

```bash
# Build the Docker image
docker build -t smolvlm-api .

# Run with Docker Compose (recommended)
docker-compose up -d

# Or run directly with Docker
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v smolvlm_models:/models \
  --name smolvlm-api \
  smolvlm-api
```

### Health Check

```bash
# Check if the service is running
curl http://localhost:8000/health

# Check available models
curl http://localhost:8000/v1/models
```

## API Usage

The API is compatible with OpenAI's chat completions format:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smolvlm-v1.8b-gguf",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What do you see in this image?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

## Configuration

Environment variables can be set in `docker-compose.yml` or passed to `docker run`:

- `MODEL_PATH`: Path to the GGUF model file (default: `/models/smolvlm.f16.gguf`)
- `MMPROJ_PATH`: Path to the multimodal projection file (default: `/models/mmproj-smolvlm.f16.gguf`)
- `N_GPU_LAYERS`: Number of layers to offload to GPU (-1 for all, default: -1)
- `N_CTX`: Context window size (default: 2048)
- `MODEL_ID`: Model identifier for API responses (default: `smolvlm-v1.8b-gguf`)
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)

## Hivenet Deployment

This Docker setup is optimized for Hivenet compute and requires:
- Docker containers
- GPU instances (NVIDIA)
- The specified base image

The container automatically:
1. Calls `/usr/local/bin/setup_instance.sh` as required by the cloud provider
2. Downloads model files from external URLs
3. Sets up the Python environment with CUDA support
4. Starts the FastAPI server

## Files

- `Dockerfile`: Main container definition using the specified base image
- `docker-compose.yml`: Development and testing setup
- `app.py`: FastAPI application with OpenAI-compatible endpoints
- `requirements.txt`: Python dependencies
- `entrypoint.sh`: Container startup script
- `.dockerignore`: Build context optimization

## Model Serving Best Practices

This deployment follows model serving best practices:
- Health checks for container orchestration
- Proper error handling and logging
- Environment-based configuration
- Efficient Docker layer caching
- GPU resource management
- Graceful startup and shutdown
