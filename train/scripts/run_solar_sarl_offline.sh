#!/bin/bash
#SBATCH --job-name=solar-sarl-offline
#SBATCH --partition=<partition>
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=path_to_SARL/logs/solar-sarl-offline.log
#SBATCH --error=path_to_SARL/logs/solar-sarl-offline.log

# ──────────────────────────────────────────────────────────────────────────────
# Two-phase offline SARL training for Solar Open 100B:
#
#   Phase 1 (all 4 nodes for vLLM generation, head node for SARL/retriever):
#     a. Retriever server     — head node, conda "retriever" env
#     b. vLLM server × 4     — one per node (head=direct, workers=ssh), TP=8
#     c. SARL rollout gen     — head node, DP=4 across all 4 vLLM instances
#        → saves HF dataset shards via offline_dataset.py hook
#     d. Consolidate shards   → pool_ready/ directory
#
#   Phase 2 (all 4 nodes, 32 GPUs):
#     torchtitan-rl GRPO training on pool_ready/
#
# Prerequisites:
#   1. SARL trainer.py must import & call save_batch_as_hf_dataset() after
#      compute_advantages_and_returns() — see offline_dataset.py for instructions.
#   2. torchtitan-rl must be pip-installed in a separate conda env (TITAN_ENV below).
#   3. solar_open_102b_sarl.toml already configured at TORCHTITAN_CONFIG.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Roots & paths ────────────────────────────────────────────────────────────
SARL_ROOT=path_to_SARL
EVAL_ROOT=path_to_evaluation
TORCHTITAN_ROOT=path_to_torchtitan-rl
CONDA_SH=path_to_miniconda3/etc/profile.d/conda.sh

MODEL_PATH=path_to_Solar-Open-100B
DATA_DIR="${SARL_ROOT}/train/train_data"
TRAIN_DATA="${DATA_DIR}/solar_train_45k_train.parquet"
TEST_DATA="${DATA_DIR}/solar_train_45k_test.parquet"
ROLLOUT_DIR="${SARL_ROOT}/train/outputs/rollouts"
LOG_DIR="${SARL_ROOT}/logs"
TORCHTITAN_CONFIG="${TORCHTITAN_ROOT}/torchtitan/models/solar_wbl/train_configs/solar_open_102b_sarl.toml"

# ─── Env names ────────────────────────────────────────────────────────────────
TITAN_ENV=cuda128

# ─── Ports ────────────────────────────────────────────────────────────────────
RETRIEVER_PORT=8010
VLLM_PORT=8000

# ─── Rollout generation settings ─────────────────────────────────────────────
N_ROLLOUT_STEPS=200
N_SAMPLES_PER_PROMPT=5
OFFLINE_DATASET_DIR="${ROLLOUT_DIR}"

# ─── Cluster topology ─────────────────────────────────────────────────────────
# The batch script runs on the first allocated node (head node). No srun needed
# for single-node operations — avoids Slurm step-slot contention with daemons.
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head="${nodes[0]}"
head_ip=$(hostname -I | awk '{print $1}')
NNODES=${SLURM_JOB_NUM_NODES}
GPUS_PER_NODE=8

mkdir -p "$LOG_DIR" "$ROLLOUT_DIR"

echo "Head node : $head ($head_ip)"
echo "All nodes : ${nodes[*]}"


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Rollout generation
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Phase 1: rollout generation ==="

# ── Kill stale vLLM processes on all nodes ────────────────────────────────
for node in "${nodes[@]}"; do
    PIDS=$(ssh -o StrictHostKeyChecking=no "$node" "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | awk -F\",\" '/VLLM::Worker_TP/ { gsub(/ /, \"\", \$1); print \$1 }' | tr '\n' ' '" 2>/dev/null || true)
    if [ -z "$PIDS" ]; then
        echo "No VLLM::Worker_TP processes found on $node"
    else
        echo "Killing stale vLLM PIDs on $node: $PIDS"
        ssh -o StrictHostKeyChecking=no "$node" "sudo kill -9 $PIDS"
        echo "Killed on $node. Waiting for cleanup..."
    fi
done
sleep 20
echo "vLLM cleanup done."

# ── 1a. Start retriever on head node ──────────────────────────────────────
echo "Starting retriever on ${head}:${RETRIEVER_PORT}..."
bash -c "
    source ${CONDA_SH}
    conda activate retriever
    cd ${SARL_ROOT}/wiki_server
    bash launch_local_server.sh ${RETRIEVER_PORT} e5_addr_dir
" > "${LOG_DIR}/retriever.log" 2>&1 &
RETRIEVER_PID=$!

echo "Waiting for retriever at ${head_ip}:${RETRIEVER_PORT} ..."
WAIT=0
until curl -s "http://${head_ip}:${RETRIEVER_PORT}/retrieve" >/dev/null 2>&1; do
    sleep 5
    WAIT=$((WAIT + 5))
    if [ "${WAIT}" -ge 1200 ]; then
        echo "ERROR: retriever did not become ready within 20 minutes"
        exit 1
    fi
done
echo "Retriever ready after ${WAIT}s"

# ── 1b. Start vLLM on all 4 nodes ─────────────────────────────────────────
# Head node: run directly. Worker nodes: run via ssh (no Slurm step slots used).
# Use getent for IP resolution (no SSH needed, immune to set -e).
# Pre-build the vLLM command string to avoid heredoc-in-loop issues.
VLLM_PIDS=()
NODE_IPS=()
VLLM_URL_PARTS=()

VLLM_CMD="source ${EVAL_ROOT}/.venv/bin/activate && \
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
    export VLLM_CACHE_ROOT=path_to_vllm && \
    vllm serve '${MODEL_PATH}' \
        --port ${VLLM_PORT} \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --enable-auto-tool-choice \
        --tool-call-parser solar_open \
        --reasoning-parser solar_open \
        --logits-processors vllm.model_executor.models.parallel_tool_call_logits_processor:ParallelToolCallLogitsProcessor \
        --logits-processors vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor"

for node in "${nodes[@]}"; do
    # DNS lookup — no SSH, immune to set -e
    node_ip=$(getent hosts "$node" | awk '{print $1}')
    NODE_IPS+=("$node_ip")
    VLLM_URL_PARTS+=("'${node_ip}:${VLLM_PORT}'")

    echo "Starting vLLM on ${node} (${node_ip}):${VLLM_PORT}..."
    if [ "$node" = "$head" ]; then
        bash -c "$VLLM_CMD" > "${LOG_DIR}/vllm_${node}.log" 2>&1 &
    else
        ssh -o StrictHostKeyChecking=no "$node" "$VLLM_CMD" > "${LOG_DIR}/vllm_${node}.log" 2>&1 &
    fi
    VLLM_PIDS+=($!)
done

# Build comma-separated URL list: ['http://ip1:8000/v1','http://ip2:8000/v1',...]
VLLM_URLS_ARG="[$(IFS=,; echo "${VLLM_URL_PARTS[*]}")]"
echo "vLLM URL list: ${VLLM_URLS_ARG}"

# Wait for all vLLM instances to be healthy
for node_ip in "${NODE_IPS[@]}"; do
    echo "Waiting for vLLM at ${node_ip}:${VLLM_PORT} ..."
    WAIT=0
    until curl -s "http://${node_ip}:${VLLM_PORT}/health" >/dev/null 2>&1; do
        sleep 5
        WAIT=$((WAIT + 5))
        if [ "${WAIT}" -ge 1200 ]; then
            echo "ERROR: vLLM on ${node_ip} did not become ready within 20 minutes"
            exit 1
        fi
    done
    echo "vLLM ready on ${node_ip} after ${WAIT}s"
done

# ── 1c. SARL rollout generation (head node, connects to all 4 vLLM servers) ──
echo "Running SARL rollout generation for $N_ROLLOUT_STEPS steps..."

bash -c "
    cd ${SARL_ROOT}/train
    unset RAY_ADDRESS
    # Remove stale Ray cluster pointer so ray.init() starts a fresh local cluster
    rm -f /tmp/ray/ray_current_cluster 2>/dev/null || true
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
    export NCCL_IBEXT_DISABLE=1
    export FLASHINFER_JIT_DIR=/tmp/flashinfer_jit_${SLURM_JOB_ID}
    export OFFLINE_DATASET_DIR=${OFFLINE_DATASET_DIR}

    uv run --frozen -m skyrl_train.entrypoints.main_base \
        data.train_data=\"['${TRAIN_DATA}']\" \
        data.val_data=\"['${TEST_DATA}']\" \
        trainer.algorithm.advantage_estimator=grpo \
        trainer.disable_train=true \
        trainer.policy.optimizer_config.lr=1.0e-6 \
        trainer.policy.optimizer_config.max_grad_norm=0.5 \
        trainer.algorithm.use_kl_loss=false \
        trainer.algorithm.eps_clip_low=0.2 \
        trainer.algorithm.eps_clip_high=0.28 \
        trainer.algorithm.grpo_norm_by_std=false \
        trainer.algorithm.loss_reduction=seq_mean_token_sum_norm \
        trainer.policy.model.path='${MODEL_PATH}' \
        trainer.placement.colocate_all=false \
        trainer.strategy=fsdp2 \
        trainer.policy.fsdp_config.cpu_offload=true \
        trainer.ref.fsdp_config.cpu_offload=true \
        trainer.placement.policy_num_gpus_per_node=${GPUS_PER_NODE} \
        trainer.placement.ref_num_gpus_per_node=${GPUS_PER_NODE} \
        generator.run_engines_locally=false \
        generator.num_inference_engines=${NNODES} \
        generator.remote_inference_engine_urls=\"${VLLM_URLS_ARG}\" \
        generator.inference_engine_tensor_parallel_size=8 \
        generator.inference_engine_data_parallel_size=${NNODES} \
        generator.apply_overlong_filtering=true \
        trainer.epochs=1 \
        trainer.steps=${N_ROLLOUT_STEPS} \
        trainer.train_batch_size=128 \
        trainer.policy_mini_batch_size=64 \
        trainer.micro_forward_batch_size_per_gpu=2 \
        trainer.micro_train_batch_size_per_gpu=2 \
        trainer.max_prompt_length=32768 \
        generator.max_input_length=32768 \
        generator.sampling_params.max_generate_length=2048 \
        generator.async_engine=true \
        generator.batched=false \
        generator.use_conversation_multi_turn=true \
        generator.n_samples_per_prompt=${N_SAMPLES_PER_PROMPT} \
        generator.max_turns=32 \
        generator.sampling_params.temperature=1.0 \
        generator.sampling_params.top_p=1.0 \
        environment.env_class=search_tool_calling \
        environment.skyrl_gym.max_env_workers=16 \
        environment.skyrl_gym.search_tool_calling.log_requests=false \
        environment.skyrl_gym.search_tool_calling.search_url=\"http://${head_ip}:${RETRIEVER_PORT}/retrieve\" \
        environment.skyrl_gym.search_tool_calling.topk=3 \
        environment.skyrl_gym.search_tool_calling.feedback_model=self \
        environment.skyrl_gym.search_tool_calling.mask_zero_format=false \
        environment.skyrl_gym.search_tool_calling.feedback_penalty=0.8 \
        environment.skyrl_gym.search_tool_calling.enable_feedback_reward=true \
        environment.skyrl_gym.search_tool_calling.feedback_coef=0.5 \
        environment.skyrl_gym.search_tool_calling.tool_call_format=solar \
        generator.sampling_params.stop=\"['<|tool_call:end|>']\" \
        trainer.logger=console \
        trainer.ckpt_interval=999999 \
        trainer.eval_interval=-1 \
        trainer.eval_before_train=false \
        trainer.dump_data_batch=true
" 2>&1 | tee "${LOG_DIR}/sarl_generation.log"

echo "SARL generation done."

# ── Shut down all vLLM instances and retriever ────────────────────────────
echo "Shutting down vLLM instances and retriever..."
for pid in "${VLLM_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
done
kill $RETRIEVER_PID 2>/dev/null || true
# Also kill vLLM worker processes on each node (ssh-launched processes survive kill of ssh)
for node in "${nodes[@]}"; do
    ssh -o StrictHostKeyChecking=no "$node" "pkill -f 'vllm serve' || true" 2>/dev/null || true
done
sleep 10

# ── 1d. Consolidate shards into pool_ready/ ────────────────────────────────
echo "Consolidating rollout shards..."
cd "${SARL_ROOT}/train"
uv run --frozen python -c "
from skyrl_train.utils.offline_dataset import consolidate_shards_to_hf_dataset
consolidate_shards_to_hf_dataset(
    output_dir='${OFFLINE_DATASET_DIR}',
    pool_ready_subdir='pool_ready',
)
print('Consolidated -> ${OFFLINE_DATASET_DIR}/pool_ready')
"
echo "pool_ready ready at ${ROLLOUT_DIR}/pool_ready"


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — torchtitan-rl GRPO training on all 4 nodes
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Phase 2: torchtitan-rl training on $NNODES nodes ==="

export MASTER_ADDR=$head_ip
export MASTER_PORT=29500
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_API_KEY="<your_wandb_api_key_here>"
export WANDB_PROJECT="solar-sarl"
export WANDB_RUN_NAME="solar-open-100b-sarl"

srun --nodes=$NNODES --ntasks-per-node=1 bash -c "
    source ${CONDA_SH}
    conda activate ${TITAN_ENV}

    cd ${TORCHTITAN_ROOT}

    NODE_RANK=\$SLURM_NODEID
    echo \"Starting torchrun on node \$NODE_RANK / \$HOSTNAME\"

    torchrun \
        --nproc_per_node=${GPUS_PER_NODE} \
        --nnodes=${NNODES} \
        --node_rank=\$NODE_RANK \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        --rdzv_backend=c10d \
        -m torchtitan.train \
        --job.config_file ${TORCHTITAN_CONFIG}
" 2>&1 | tee "${LOG_DIR}/torchtitan_train.log"

echo "=== Training complete ==="
