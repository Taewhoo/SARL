#!/bin/bash
#SBATCH --job-name=solar-quick-test
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=path_to_SARL/logs/quick-test.log
#SBATCH --error=path_to_SARL/logs/quick-test.log
#SBATCH --time=2:00:00

# Quick 1-node test: starts retriever + vLLM, generates a few trajectories, exits.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

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

# ── Kill stale vLLM ──────────────────────────────────────────────────────────
PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | awk -F"," '/VLLM::Worker_TP/ { gsub(/ /, "", $1); print $1 }' | tr '\n' ' ' || true)
if [ -n "$PIDS" ]; then
    echo "Killing stale vLLM PIDs: $PIDS"
    sudo kill -9 $PIDS || true
    sleep 10
fi

# ── Start retriever ──────────────────────────────────────────────────────────
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

# ── Start vLLM ───────────────────────────────────────────────────────────────
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
    > "${LOG_DIR}/vllm_quick.log" 2>&1 &

echo "Waiting for vLLM..."
WAIT=0
until curl -s "http://${head_ip}:${VLLM_PORT}/health" >/dev/null 2>&1; do
    sleep 5; WAIT=$((WAIT + 5))
    if [ "${WAIT}" -ge 1200 ]; then echo "ERROR: vLLM timeout"; exit 1; fi
done
echo "vLLM ready after ${WAIT}s"

# ── Run generation ───────────────────────────────────────────────────────────
DATA_DIR="${SARL_ROOT}/train/train_data"
TRAIN_DATA="${DATA_DIR}/solar_train_45k_train.parquet"
TEST_DATA="${DATA_DIR}/solar_train_45k_test.parquet"
ROLLOUT_DIR="${SARL_ROOT}/train/outputs/rollouts_quick"

mkdir -p "$ROLLOUT_DIR"
cd "${SARL_ROOT}/train"
unset RAY_ADDRESS 2>/dev/null || true
rm -f /tmp/ray/ray_current_cluster 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export NCCL_IBEXT_DISABLE=1
export FLASHINFER_JIT_DIR=/tmp/flashinfer_jit_quick
export OFFLINE_DATASET_DIR="${ROLLOUT_DIR}"

echo ""
echo "============================================"
echo "Starting generation (1 node, 1 step, batch=4, n_samples=2)"
echo "============================================"

uv run --frozen -m skyrl_train.entrypoints.main_base \
    data.train_data="['${TRAIN_DATA}']" \
    data.val_data="['${TEST_DATA}']" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.disable_train=true \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.max_grad_norm=0.5 \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.eps_clip_low=0.2 \
    trainer.algorithm.eps_clip_high=0.28 \
    trainer.algorithm.grpo_norm_by_std=false \
    trainer.algorithm.loss_reduction=seq_mean_token_sum_norm \
    trainer.policy.model.path="${MODEL_PATH}" \
    trainer.placement.colocate_all=false \
    trainer.strategy=fsdp2 \
    trainer.policy.fsdp_config.cpu_offload=true \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.placement.policy_num_gpus_per_node=8 \
    trainer.placement.ref_num_gpus_per_node=8 \
    generator.run_engines_locally=false \
    generator.num_inference_engines=1 \
    generator.remote_inference_engine_urls="['${head_ip}:${VLLM_PORT}']" \
    generator.inference_engine_tensor_parallel_size=8 \
    generator.inference_engine_data_parallel_size=1 \
    generator.apply_overlong_filtering=true \
    trainer.epochs=1 \
    trainer.steps=1 \
    trainer.train_batch_size=4 \
    trainer.policy_mini_batch_size=4 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=32768 \
    generator.max_input_length=32768 \
    generator.sampling_params.max_generate_length=2048 \
    generator.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=true \
    generator.n_samples_per_prompt=2 \
    generator.max_turns=32 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    environment.env_class=search_tool_calling \
    environment.skyrl_gym.max_env_workers=8 \
    environment.skyrl_gym.search_tool_calling.log_requests=false \
    environment.skyrl_gym.search_tool_calling.search_url="http://${head_ip}:${RETRIEVER_PORT}/retrieve" \
    environment.skyrl_gym.search_tool_calling.topk=3 \
    environment.skyrl_gym.search_tool_calling.feedback_model=self \
    environment.skyrl_gym.search_tool_calling.mask_zero_format=false \
    environment.skyrl_gym.search_tool_calling.feedback_penalty=0.8 \
    environment.skyrl_gym.search_tool_calling.enable_feedback_reward=true \
    environment.skyrl_gym.search_tool_calling.feedback_coef=0.5 \
    environment.skyrl_gym.search_tool_calling.tool_call_format=solar \
    generator.sampling_params.stop="['<|tool_call:end|>']" \
    trainer.logger=console \
    trainer.ckpt_interval=999999 \
    trainer.eval_interval=-1 \
    trainer.eval_before_train=false \
    trainer.dump_data_batch=true \
    2>&1 | tee "${LOG_DIR}/quick_test.log"

echo ""
echo "============================================"
echo "Quick test done. Check: ${ROLLOUT_DIR}"
echo "============================================"
