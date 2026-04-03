#!/bin/bash
set -euo pipefail

# =============================================================================
# SVG Refine GRPO - Qwen2.5-VL-7B Validation Only (4 nodes × 8 GPUs = 32 GPUs)
# =============================================================================
# Run val_before_train with 7B SFT checkpoint, then Ctrl+C after validation.
# Trajectory dump goes to: ${PROJECT_DIR}/trajectory_dump/
# Analyze with: python analyze_trajectories.py
# =============================================================================

# Activate your Python environment (conda/venv) before running:
# source /path/to/your/env/bin/activate
# Set VERL_DIR and WORKSPACE before running, or adjust these paths:
# export VERL_DIR=/path/to/verl
# export WORKSPACE=/path/to/workspace
export PYTHONPATH="${VERL_DIR:-$(dirname $PROJECT_DIR)/verl}:$(dirname $PROJECT_DIR):${PYTHONPATH:-}"

python3 -c "import SVG_refine_grpo.patch_qwen25vl_config; print('Qwen VL config patch verified')"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${PROJECT_DIR}/configs"
TRAIN_DATA="${PROJECT_DIR}/train.parquet"
VAL_DATA="${PROJECT_DIR}/val.parquet"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/qwen25vl7b_val_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

# Clean previous trajectory dump
rm -f "${PROJECT_DIR}/trajectory_dump/traj_pid"*.jsonl
echo "Cleaned trajectory_dump/"

# Verify CLIP API
if ! curl -s --connect-timeout 5 http://localhost:18080/health > /dev/null 2>&1; then
    echo "WARNING: CLIP API at http://localhost:18080 is not reachable."
fi

echo "Ray cluster status:"
ray status || echo "WARNING: Ray cluster not detected."

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='svg_grpo_qwen25vl7b_val' \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
