# Use the specified base image from the cloud provider
FROM 993cz8u0.c1.gra9.container-registry.ovh.net/hive-compute-public/base-ubuntu:latest

# Set environment variables for model serving
ENV MODEL_PATH="/models/smolvlm.f16.gguf"
ENV MMPROJ_PATH="/models/mmproj-smolvlm.f16.gguf"
ENV N_GPU_LAYERS=-1
ENV N_CTX=2048
ENV MODEL_ID="smolvlm-v1.8b-gguf"
ENV HOST="0.0.0.0"
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

# Install system dependencies required for model serving
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies with CUDA support
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --verbose -r requirements.txt

# Copy application files
COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Create models directory
RUN mkdir -p /models

# Expose the API port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use our entrypoint script which calls setup_instance.sh and starts the service
CMD ["/app/entrypoint.sh"]
