#!/usr/bin/env bash
set -euo pipefail

# Activate your Python environment:
# source /path/to/your/env/bin/activate

export NNODES=4
export NODE_RANK=0
export NPROC_PER_NODE=8
export MASTER_ADDR=<HEAD_NODE_IP>
export MASTER_PORT=29500

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=eth0  # Adjust to your network interface
export NCCL_IB_DISABLE=1

llamafactory-cli train ${LLAMA_FACTORY_DIR}/examples/train_full/qwen2_5vl_7b_svgmix256_full_sft.yaml