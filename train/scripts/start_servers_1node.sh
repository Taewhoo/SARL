#!/bin/bash
#SBATCH --job-name=solar-srv-1n
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=path_to_SARL/logs/solar-srv-1n.log
#SBATCH --error=path_to_SARL/logs/solar-srv-1n.log
#SBATCH --time=2:00:00

# 1-node server: retriever + vLLM. Use run_generation_only.sh after ready.

set -euo pipefail

SARL_ROOT=path_to_SARL
EVAL_ROOT=path_to_evaluation
CONDA_SH=path_to_miniconda3/etc/profile.d/conda.sh
MODEL_PATH=path_to_Solar-Open-100B
LOG_DIR="${SARL_ROOT}/logs"
RETRIEVER_PORT=8010
VLLM_PORT=8000

head_ip=$(hostname -I | awk '{print $1}')
mkdir -p "$LOG_DIR"
echo "Node: $(hostname) ($head_ip)"

# Kill stale vLLM
PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | awk -F"," '/VLLM::Worker_TP/ { gsub(/ /, "", $1); print $1 }' | tr '\n' ' ' || true)
if [ -n "$PIDS" ]; then
    echo "Killing stale vLLM PIDs: $PIDS"
    sudo kill -9 $PIDS || true
    sleep 10
fi

# Start retriever
echo "Starting retriever on ${head_ip}:${RETRIEVER_PORT}..."
bash -c "
    source ${CONDA_SH}
    conda activate retriever
    cd ${SARL_ROOT}/wiki_server
    bash launch_local_server.sh ${RETRIEVER_PORT} e5_addr_dir
" > "${LOG_DIR}/retriever.log" 2>&1 &

echo "Waiting for retriever..."
WAIT=0
until curl -s "http://${head_ip}:${RETRIEVER_PORT}/retrieve" >/dev/null 2>&1; do
    sleep 5; WAIT=$((WAIT + 5))
    if [ "${WAIT}" -ge 600 ]; then echo "ERROR: retriever timeout"; exit 1; fi
done
echo "Retriever ready after ${WAIT}s"

# Start vLLM
echo "Starting vLLM on ${head_ip}:${VLLM_PORT}..."
bash -c "source ${EVAL_ROOT}/.venv/bin/activate && \
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
        --logits-processors vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor" \
    > "${LOG_DIR}/vllm_1n.log" 2>&1 &

echo "Waiting for vLLM..."
WAIT=0
until curl -s "http://${head_ip}:${VLLM_PORT}/health" >/dev/null 2>&1; do
    sleep 5; WAIT=$((WAIT + 5))
    if [ "${WAIT}" -ge 1200 ]; then echo "ERROR: vLLM timeout"; exit 1; fi
done
echo "vLLM ready after ${WAIT}s"

# Write server info
INFO_FILE="${SARL_ROOT}/logs/server_info.env"
{
    echo "HEAD_IP=${head_ip}"
    echo "RETRIEVER_PORT=${RETRIEVER_PORT}"
    echo "VLLM_PORT=${VLLM_PORT}"
    echo "VLLM_URLS=['${head_ip}:${VLLM_PORT}']"
    echo "NNODES=1"
} > "$INFO_FILE"

echo ""
echo "============================================"
echo "ALL SERVERS READY (1 node)"
cat "$INFO_FILE"
echo ""
echo "Run generation with:"
echo "  ssh $(hostname) 'bash ${SARL_ROOT}/train/scripts/run_generation_only.sh'"
echo "============================================"

wait
