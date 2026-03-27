#!/bin/bash
# Runs SARL generation ONLY — assumes retriever + vLLM are already running.
# Reads connection info from logs/server_info.env (written by start_servers.sh).
# Can be re-run quickly after code fixes without restarting servers.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

SARL_ROOT=/mnt/weka/post_training_tmp/pt2-search-agent/SARL
INFO_FILE="${SARL_ROOT}/logs/server_info.env"

if [ ! -f "$INFO_FILE" ]; then
    echo "ERROR: ${INFO_FILE} not found. Start servers first with start_servers.sh"
    exit 1
fi

source "$INFO_FILE"
echo "Using servers: HEAD_IP=${HEAD_IP}, RETRIEVER_PORT=${RETRIEVER_PORT}, VLLM_URLS=${VLLM_URLS}"

# Verify servers are alive
echo "Checking retriever..."
curl -s "http://${HEAD_IP}:${RETRIEVER_PORT}/retrieve" >/dev/null 2>&1 || { echo "ERROR: retriever not responding"; exit 1; }
echo "Retriever OK"

DATA_DIR="${SARL_ROOT}/train/train_data"
TRAIN_DATA="${DATA_DIR}/solar_train_45k_train.parquet"
TEST_DATA="${DATA_DIR}/solar_train_45k_test.parquet"
MODEL_PATH=/mnt/weka/post_training/checkpoints/Solar-Open-100B
ROLLOUT_DIR="${ROLLOUT_DIR:-${SARL_ROOT}/train/outputs/rollouts_test}"
GPUS_PER_NODE=8
N_ROLLOUT_STEPS="${N_ROLLOUT_STEPS:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"

mkdir -p "$ROLLOUT_DIR"

cd "${SARL_ROOT}/train"
unset RAY_ADDRESS 2>/dev/null || true
rm -f /tmp/ray/ray_current_cluster 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export NCCL_IBEXT_DISABLE=1
export FLASHINFER_JIT_DIR=/tmp/flashinfer_jit_gen_test
export OFFLINE_DATASET_DIR="${ROLLOUT_DIR}"

echo "Starting SARL generation (${N_ROLLOUT_STEPS} steps)..."

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
    trainer.placement.policy_num_gpus_per_node=${GPUS_PER_NODE} \
    trainer.placement.ref_num_gpus_per_node=${GPUS_PER_NODE} \
    generator.run_engines_locally=false \
    generator.num_inference_engines=${NNODES} \
    generator.remote_inference_engine_urls="${VLLM_URLS}" \
    generator.inference_engine_tensor_parallel_size=8 \
    generator.inference_engine_data_parallel_size=${NNODES} \
    generator.apply_overlong_filtering=true \
    trainer.epochs=1 \
    trainer.steps=${N_ROLLOUT_STEPS} \
    trainer.train_batch_size=${TRAIN_BATCH_SIZE} \
    trainer.policy_mini_batch_size=${TRAIN_BATCH_SIZE} \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
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
    environment.skyrl_gym.search_tool_calling.search_url="http://${HEAD_IP}:${RETRIEVER_PORT}/retrieve" \
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
    2>&1 | tee "${SARL_ROOT}/logs/sarl_generation_test.log"

echo "Generation done."
