#!/bin/bash
#SBATCH --job-name=solar-gspo-train
#SBATCH --partition=normal
#SBATCH --nodelist=Slurm-GPU-Node-[1-49]
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=/mnt/weka/post_training_tmp/pt2-search-agent/SARL/logs/solar-gspo-train.log
#SBATCH --error=/mnt/weka/post_training_tmp/pt2-search-agent/SARL/logs/solar-gspo-train.log

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — torchtitan-rl GSPO training on offline SARL rollout dataset
#
# Uses GSPO (sequence-level importance ratio, tight clipping) instead of GRPO.
# 8 nodes × 8 GPUs = 64 GPUs, EP=8, FSDP shard=64, effective DP=8
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SARL_ROOT=/mnt/weka/post_training_tmp/pt2-search-agent/SARL
TORCHTITAN_ROOT=/mnt/weka/post_training_tmp/pt2-search-agent/torchtitan-rl
CONDA_SH=/mnt/weka/post_training_tmp/pt2-search-agent/miniconda3/etc/profile.d/conda.sh
TITAN_ENV=cuda128
LOG_DIR="${SARL_ROOT}/logs"
TORCHTITAN_CONFIG="${TORCHTITAN_ROOT}/torchtitan/models/solar_wbl/train_configs/solar_open_102b_gspo.toml"

GPUS_PER_NODE=8
NNODES=${SLURM_JOB_NUM_NODES}

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head="${nodes[0]}"
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head" hostname --ip-address)

mkdir -p "$LOG_DIR"

echo "Head node : $head ($head_ip)"
echo "All nodes : ${nodes[*]}"
echo "NNODES    : $NNODES"
echo "Config    : $TORCHTITAN_CONFIG"

# Verify dataset exists
POOL_READY="${SARL_ROOT}/train/outputs/rollouts/pool_ready"
if [ ! -d "$POOL_READY" ]; then
    echo "ERROR: pool_ready dataset not found at $POOL_READY"
    echo "Run Phase 1 (rollout generation) first."
    exit 1
fi
echo "Dataset   : $POOL_READY"

export MASTER_ADDR=$head_ip
export MASTER_PORT=29500
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_API_KEY="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"
export WANDB_PROJECT="solar-sarl"
export WANDB_RUN_NAME="solar-open-100b-gspo-8node"

echo "=== Starting torchtitan-rl GSPO training on $NNODES nodes ==="

srun --nodes=$NNODES --ntasks-per-node=1 bash -c "
    source ${CONDA_SH}
    conda activate ${TITAN_ENV}

    cd ${TORCHTITAN_ROOT}

    echo \"Starting torchrun on node \$SLURM_NODEID / \$(hostname)\"

    torchrun \
        --nproc_per_node=${GPUS_PER_NODE} \
        --nnodes=${NNODES} \
        --node_rank=\$SLURM_NODEID \
        --rdzv_id=102 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${head_ip}:${MASTER_PORT} \
        -m torchtitan.train \
        --job.config_file ${TORCHTITAN_CONFIG}
" 2>&1 | tee "${LOG_DIR}/torchtitan_gspo_train.log"

echo "=== Training complete ==="
