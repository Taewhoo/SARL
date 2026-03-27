#!/bin/bash
#SBATCH --job-name=solar-servers
#SBATCH --partition=omni
#SBATCH --nodelist=Slurm-GPU-Node-[75-90]
#SBATCH --exclude=Slurm-GPU-Node-82,Slurm-GPU-Node-90
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=/mnt/weka/post_training_tmp/pt2-search-agent/SARL/logs/solar-servers.log
#SBATCH --error=/mnt/weka/post_training_tmp/pt2-search-agent/SARL/logs/solar-servers.log
#SBATCH --time=4:00:00

# Start retriever + vLLM servers and keep them alive.
# After this is running, use run_generation_only.sh to test generation.

set -euo pipefail

SARL_ROOT=/mnt/weka/post_training_tmp/pt2-search-agent/SARL
EVAL_ROOT=/mnt/weka/post_training/pt2-search-agent/evaluation
CONDA_SH=/mnt/weka/post_training_tmp/pt2-search-agent/miniconda3/etc/profile.d/conda.sh
MODEL_PATH=/mnt/weka/post_training/checkpoints/Solar-Open-100B
LOG_DIR="${SARL_ROOT}/logs"
RETRIEVER_PORT=8010
VLLM_PORT=8000

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head="${nodes[0]}"
head_ip=$(hostname -I | awk '{print $1}')

mkdir -p "$LOG_DIR"

echo "Head node : $head ($head_ip)"
echo "All nodes : ${nodes[*]}"

# Kill stale vLLM
for node in "${nodes[@]}"; do
    PIDS=$(ssh -o StrictHostKeyChecking=no "$node" "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | awk -F\",\" '/VLLM::Worker_TP/ { gsub(/ /, \"\", \$1); print \$1 }' | tr '\n' ' '" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "Killing stale vLLM PIDs on $node: $PIDS"
        ssh -o StrictHostKeyChecking=no "$node" "sudo kill -9 $PIDS"
    fi
done
sleep 10

# Start retriever
echo "Starting retriever on ${head}:${RETRIEVER_PORT}..."
bash -c "
    source ${CONDA_SH}
    conda activate retriever
    cd ${SARL_ROOT}/wiki_server
    bash launch_local_server.sh ${RETRIEVER_PORT} e5_addr_dir
" > "${LOG_DIR}/retriever.log" 2>&1 &
RETRIEVER_PID=$!

echo "Waiting for retriever..."
WAIT=0
until curl -s "http://${head_ip}:${RETRIEVER_PORT}/retrieve" >/dev/null 2>&1; do
    sleep 5; WAIT=$((WAIT + 5))
    if [ "${WAIT}" -ge 1200 ]; then echo "ERROR: retriever timeout"; exit 1; fi
done
echo "Retriever ready after ${WAIT}s"

# Start vLLM on all nodes
VLLM_PIDS=()
VLLM_CMD="source ${EVAL_ROOT}/.venv/bin/activate && \
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
    export VLLM_CACHE_ROOT=/mnt/weka/post_training_tmp/pt2-search-agent/vllm && \
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
    node_ip=$(getent hosts "$node" | awk '{print $1}')
    echo "Starting vLLM on ${node} (${node_ip}):${VLLM_PORT}..."
    if [ "$node" = "$head" ]; then
        bash -c "$VLLM_CMD" > "${LOG_DIR}/vllm_${node}.log" 2>&1 &
    else
        ssh -o StrictHostKeyChecking=no "$node" "$VLLM_CMD" > "${LOG_DIR}/vllm_${node}.log" 2>&1 &
    fi
    VLLM_PIDS+=($!)
done

# Wait for all vLLM
for node in "${nodes[@]}"; do
    node_ip=$(getent hosts "$node" | awk '{print $1}')
    echo "Waiting for vLLM at ${node_ip}:${VLLM_PORT} ..."
    WAIT=0
    until curl -s "http://${node_ip}:${VLLM_PORT}/health" >/dev/null 2>&1; do
        sleep 5; WAIT=$((WAIT + 5))
        if [ "${WAIT}" -ge 1200 ]; then echo "ERROR: vLLM on ${node_ip} timeout"; exit 1; fi
    done
    echo "vLLM ready on ${node_ip} after ${WAIT}s"
done

# Write server info for the generation script
INFO_FILE="${SARL_ROOT}/logs/server_info.env"
{
    echo "HEAD_IP=${head_ip}"
    echo "RETRIEVER_PORT=${RETRIEVER_PORT}"
    echo "VLLM_PORT=${VLLM_PORT}"
    echo -n "VLLM_URLS=["
    first=true
    for node in "${nodes[@]}"; do
        node_ip=$(getent hosts "$node" | awk '{print $1}')
        if [ "$first" = true ]; then first=false; else echo -n ","; fi
        echo -n "'${node_ip}:${VLLM_PORT}'"
    done
    echo "]"
    echo "NNODES=${#nodes[@]}"
} > "$INFO_FILE"

echo ""
echo "============================================"
echo "ALL SERVERS READY"
echo "Server info written to: ${INFO_FILE}"
cat "$INFO_FILE"
echo ""
echo "Run generation with:"
echo "  ssh $head 'bash ${SARL_ROOT}/train/scripts/run_generation_only.sh'"
echo "============================================"

# Keep alive
echo "Keeping servers alive... (Ctrl+C or scancel to stop)"
wait
