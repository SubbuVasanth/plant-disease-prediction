#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# build_run.sh — Build and run the AgriScan-AI Docker container
# Usage:  bash build_run.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

IMAGE_NAME="agriscan-ai"
CONTAINER_NAME="agriscan-ai-container"
PORT=5000

# ── Ensure GROQ_API_KEY is set ───────────────────────────────────────────
if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "ERROR: GROQ_API_KEY environment variable is not set."
  echo "Export it first:  export GROQ_API_KEY=gsk_..."
  exit 1
fi

echo "🔨 Building Docker image: ${IMAGE_NAME} ..."
docker build -t "${IMAGE_NAME}" .

# ── Stop & remove any existing container with the same name ──────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "🛑 Stopping existing container: ${CONTAINER_NAME} ..."
  docker stop "${CONTAINER_NAME}" 2>/dev/null || true
  docker rm   "${CONTAINER_NAME}" 2>/dev/null || true
fi

echo "🚀 Running container on port ${PORT} ..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:${PORT}" \
  -e GROQ_API_KEY="${GROQ_API_KEY}" \
  "${IMAGE_NAME}"

echo ""
echo "✅ AgriScan-AI is running at http://localhost:${PORT}"
echo "   Test it:  curl -X POST -F 'file=@leaf.jpg' http://localhost:${PORT}/predict"
