set -x

# Ablation: Standard Search-R1 GRPO for Llama-3.1-8B-Instruct
#
# Pure Search-R1 baseline — no SARL components:
#   - disable_feedback=true: episode ends on first <answer>, no feedback loop,
#     reward is pure base correctness (no 0.8^(T-1) penalty, no R_fb)
#   - Standard GRPO: use_kl_loss=true, default eps_clip (0.2/0.2),
#     grpo_norm_by_std=true, loss_reduction=token_mean
#   - mask_zero_format=true (default): invalid-format trajectories masked (no gradient)
#
# vs SARL v4 (run_search_self_critique_v4_llama3_8b_instruct.sh):
#   - disable_feedback=false (default): multi-turn feedback loop with self-critique
#   - Feedback reward: enable_feedback_reward=true, feedback_coef=0.5, feedback_penalty=0.8
#   - Custom GRPO: use_kl_loss=false, eps_clip_high=0.28, grpo_norm_by_std=false,
#     loss_reduction=seq_mean_token_sum_norm, mask_zero_format=false

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook  # Use uv for Ray workers
# Avoid NCCL NET/OFI "Couldn't open CQ" (RC -22); must be forwarded to Ray workers via utils.prepare_runtime_environment
export NCCL_IBEXT_DISABLE=1
# Per-job FlashInfer JIT dir to avoid stale cache from previous Ray sessions (ninja: ... session_* ... missing)
export FLASHINFER_JIT_DIR="${FLASHINFER_JIT_DIR:-/tmp/flashinfer_jit_$$}"
export WANDB_API_KEY="<your_wandb_api_key_here>"
export HF_TOKEN="<your_hf_token_here>"

WAND_PROJECT='NQ-HotpotQA-ASearcher-45k-clean'
BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
EXPERIMENT_NAME='skyrl-search-grpo-llama3.1-8b-instruct-ablation'

# path for dataset (.parquet files) containing the prompts and metadata for each question
DATA_DIR="path_to_SARL/train/train_data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE_FILE="${SCRIPT_DIR}/llama3_instruct_chat_template.j2"

# change environment.skyrl_gym.search.search_url to your own search server
uv run --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/combined_train_45k_train_converted.parquet']" \
  data.val_data="['${DATA_DIR}/combined_train_45k_test_converted.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="${BASE_MODEL}" \
  +trainer.tokenizer_chat_template_file="${CHAT_TEMPLATE_FILE}" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=2 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.gpu_memory_utilization=0.5 \
  generator.apply_overlong_filtering=true \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=32768 \
  generator.sampling_params.max_generate_length=512 \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=5 \
  generator.max_turns=32 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</search>", "</answer>"]' \
  environment.env_class="search" \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.search.log_requests=false \
  environment.skyrl_gym.search.search_url="http://${RETRIEVER_HOST:-$(hostname)}:8010/retrieve" \
  environment.skyrl_gym.search.topk=3 \
  +environment.skyrl_gym.search.disable_feedback=true \
  trainer.logger="wandb" \
  trainer.project_name="${WAND_PROJECT}" \
  trainer.run_name="${EXPERIMENT_NAME}" \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=25 \
  trainer.max_ckpts_to_keep=999999 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="path_to_SARL/ckpt_grpo_ablation_llama3.1-8b-instruct" \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=25 \
  trainer.dump_data_batch=true \
  trainer.dump_eval_results=true \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  trainer.export_path="path_to_SARL/ckpt_grpo_ablation_llama3.1-8b-instruct/exports" \
  2>&1 | tee $EXPERIMENT_NAME.log
