# !/bin/bash

set -x

source activate openmmo1

cd /path/to/LLaMA-Factory

export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}

export WANDB_PROJECT=mmo1_qwen2vl72b_full_sft
export WANDB_NAME=mmo1_qwen2vl72b_full_sft_$(date "+%Y-%m-%d-%H-%M")
export WANDB_API_KEY=XXXX

# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/mmo1_qwen2vl72b_full_sft.yaml

torchrun --nproc-per-node $GPUS_PER_NODE \
    --master-addr $MASTER_ADDR \
    --node-rank $NODE_RANK \
    --master_port $MASTER_PORT \
    --nnodes $NNODES \
    src/train.py examples/train_full/mmo1_qwen2vl7b_full_sft.yaml

# FORCE_TORCHRUN=1 NNODES=$MLP_WORKER_NUM RANK=$MLP_ROLE_INDEX MASTER_ADDR=$MLP_WORKER_0_HOST MASTER_PORT=$MLP_WORKER_0_PORT llamafactory-cli train examples/train_full/mmo1_qwen2vl72b_full_sft.yaml


