#!/bin/bash
set -ex

# =============================================================================
# SVG Refine GRPO - Qwen2.5-VL-3B Smoke Test (4 GPUs)
# =============================================================================

# Activate your Python environment (conda/venv) before running:
# source /path/to/your/env/bin/activate

export PYTHONPATH="${VERL_DIR}:${WORKSPACE}:${PYTHONPATH}"

# Apply monkey-patch for Qwen2_5_VLConfig.__getattribute__ bug in transformers 4.57.0
# This must run before any transformers import (including inside Ray workers)
python3 -c "import SVG_refine_grpo.patch_qwen25vl_config; print('Qwen2.5-VL config patch verified')"

# Ensure CLIP API is running
if ! curl -s http://127.0.0.1:18080/health > /dev/null 2>&1; then
    echo "WARNING: CLIP API at http://127.0.0.1:18080 is not reachable."
    echo "Please start it first: bash ${PROJECT_DIR}/clip_similarity_api/start_clip_similarity_api.sh"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${PROJECT_DIR}/configs"

# Limit to first 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='svg_grpo_qwen25vl_smoke' \
    data.train_files="${PROJECT_DIR}/data/smoke_train_32.parquet" \
    data.val_files="${PROJECT_DIR}/data/smoke_val_16.parquet" \
    "$@"
