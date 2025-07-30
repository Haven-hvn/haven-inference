#!/bin/bash
set -e

echo ">>> Calling setup_instance.sh as required by cloud provider..."
/usr/local/bin/setup_instance.sh

echo ">>> Creating models directory..."
mkdir -p /models

echo ">>> Setting up environment variables..."
export MODEL_PATH=${MODEL_PATH:-"/models/smolvlm.f16.gguf"}
export MMPROJ_PATH=${MMPROJ_PATH:-"/models/mmproj-smolvlm.f16.gguf"}
export N_GPU_LAYERS=${N_GPU_LAYERS:-"-1"}
export N_CTX=${N_CTX:-"2048"}
export MODEL_ID=${MODEL_ID:-"smolvlm-v1.8b-gguf"}
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-"8000"}

echo ">>> Downloading model files..."
GGUF_URL="https://f006.backblazeb2.com/b2api/v1/b2_download_file_by_id?fileId=4_z81c5239493d6c7ca9c6f0917_f206db32ae6cb4176_d20250514_m023210_c006_v0601002_t0020_u01747189930776"
MMPROJ_URL="https://f006.backblazeb2.com/b2api/v1/b2_download_file_by_id?fileId=4_z81c5239493d6c7ca9c6f0917_f205738cb7ba0162f_d20250514_m022955_c006_v0601002_t0049_u01747189795531"

echo ">>> Downloading GGUF model file..."
curl -k -Lf --retry 5 --retry-delay 10 "${GGUF_URL}" -o "${MODEL_PATH}" || {
    echo "!! GGUF download failed! Contents of /models: $(ls -lh /models)" 1>&2
    exit 1
}

echo ">>> Downloading MMPROJ model file..."
curl -k -Lf --retry 5 --retry-delay 10 "${MMPROJ_URL}" -o "${MMPROJ_PATH}" || {
    echo "!! MMPROJ download failed! Contents of /models: $(ls -lh /models)" 1>&2
    exit 1
}

echo ">>> Verifying downloads..."
ls -lh /models
[ -f "${MODEL_PATH}" ] || { echo "Model file missing!"; exit 1; }
[ -f "${MMPROJ_PATH}" ] || { echo "MMPROJ file missing!"; exit 1; }

echo ">>> Configuring ngrok authtoken..."
ngrok config add-authtoken "${NGROK_AUTHTOKEN}"

echo ">>> Starting ngrok tunnel..."
ngrok http --log=stdout "${PORT}" &

echo ">>> Starting SmolVLM API server..."
cd /app
exec uvicorn app:app --host "${HOST}" --port "${PORT}" --forwarded-allow-ips '*' --proxy-headers
