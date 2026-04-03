#!/usr/bin/env bash
set -euo pipefail

# ===== 基本配置（可用环境变量覆盖） =====
PYTHON_BIN=${PYTHON_BIN:-python3}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-18080}
LOG_LEVEL=${LOG_LEVEL:-info}

# 单卡部署：默认使用 GPU0，可通过 CUDA_VISIBLE_DEVICES 覆盖
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

# 服务配置
export MODEL_PATH=${MODEL_PATH:-${CLIP_MODEL_PATH}}
export DEVICE=${DEVICE:-cuda:0}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-128}
export BATCH_TIMEOUT_MS=${BATCH_TIMEOUT_MS:-8}
export MAX_QUEUE_SIZE=${MAX_QUEUE_SIZE:-4096}
export MAX_PAIRS_PER_REQUEST=${MAX_PAIRS_PER_REQUEST:-256}
export USE_FAST_PROCESSOR=${USE_FAST_PROCESSOR:-1}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

echo "[INFO] Starting CLIP similarity API"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] HOST=$HOST PORT=$PORT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] MAX_BATCH_SIZE=$MAX_BATCH_SIZE BATCH_TIMEOUT_MS=$BATCH_TIMEOUT_MS MAX_QUEUE_SIZE=$MAX_QUEUE_SIZE"

exec "$PYTHON_BIN" -m uvicorn clip_similarity_api_server:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  --loop uvloop \
  --http httptools \
  --log-level "$LOG_LEVEL"
