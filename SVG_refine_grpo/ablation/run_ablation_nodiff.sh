#!/bin/bash
set -euo pipefail

# =============================================================================
# SVG Refine GRPO - Ablation: No Diff Heatmap (2-panel feedback)
# (4 nodes × 8 GPUs = 32 GPUs)
# =============================================================================
# This ablation keeps cold-start SFT but removes the diff heatmap from
# visual feedback. The model sees only target + rendered SVG (diptych).
#
# Prerequisites:
#   1. Start CLIP/DINO/LPIPS APIs on localhost
#   2. Set up Ray cluster across 4 nodes
#   3. Update model path in config after completing no-diff SFT
#
# Usage:
#   bash run_ablation_nodiff.sh [hydra overrides...]
# =============================================================================

# Activate your Python environment (conda/venv) before running:
# source /path/to/your/env/bin/activate
# Set VERL_DIR and WORKSPACE before running, or adjust these paths:
# export VERL_DIR=/path/to/verl
# export WORKSPACE=/path/to/workspace
export PYTHONPATH="${VERL_DIR:-$(dirname $PROJECT_DIR)/verl}:$(dirname $PROJECT_DIR):${PYTHONPATH:-}"

# Apply monkey-patch for rope_scaling compat in transformers 4.57.0
python3 -c "import SVG_refine_grpo.patch_qwen25vl_config; print('Qwen VL config patch verified')"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${PROJECT_DIR}/ablation"
TRAIN_DATA="${PROJECT_DIR}/train.parquet"
VAL_DATA="${PROJECT_DIR}/val.parquet"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ablation_nodiff_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

# Verify CLIP API is reachable
if ! curl -s --connect-timeout 5 http://localhost:18080/health > /dev/null 2>&1; then
    echo "WARNING: CLIP API at http://localhost:18080 is not reachable."
    echo "Please start it first: bash ${PROJECT_DIR}/clip_similarity_api/start_clip_similarity_api.sh"
fi

# Verify Ray cluster
echo "Ray cluster status:"
ray status || echo "WARNING: Ray cluster not detected. Make sure ray is started."

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='svg_grpo_qwen25vl7b_ablation_nodiff' \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
