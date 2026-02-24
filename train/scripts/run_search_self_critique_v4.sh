set -x

# Colocated GRPO training+generation for Qwen2.5-7B

# Feedback Model Configuration:
# - To use the model itself for feedback (default):
#   environment.skyrl_gym.search.feedback_model="self"
# - To use an external vLLM endpoint for feedback, specify the model name:
#   environment.skyrl_gym.search.feedback_model="Qwen/Qwen3-235B-A22B-Instruct-2507"
#   environment.skyrl_gym.search.feedback_url="http://192.168.0.30:8080/v1"

# Reward Configuration:
# - Total reward formula: R_total = (r * 0.8^(T-1)) + β * R_fb
#   - r: base reward (correctness)
#   - 0.8^(T-1): feedback penalty (feedback_penalty=0.8)
#   - β: feedback coefficient (feedback_coef=0.5)
#   - R_fb: feedback reward based on trajectory evolution
#     * 1.0: correct first answer, correct after feedback (maintained)
#     * 1.0: wrong first answer, correct after feedback (improved)
#     * -0.1: wrong first answer, wrong after feedback (still wrong)
#     * -2.0: correct first answer, wrong after feedback (degraded)
#     * -3.0: invalid format (no <answer> tag found after feedback)
# - Set enable_feedback_reward=false to disable R_fb component

# Format Masking:
# - mask_zero_format=true: Set reward=0 and mask loss when no valid tags detected (no gradient)
# - mask_zero_format=false: Compute reward from trajectory and train on it (even with invalid format)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook  # Use uv for Ray workers
export WANDB_API_KEY="<your_wandb_api_key_here>"

WAND_PROJECT='NQ-HotpotQA-ASearcher-45k-clean'
BASE_MODEL='Qwen/Qwen2.5-7B'
# BASE_MODEL='Qwen/Qwen2.5-14B'
# BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
EXPERIMENT_NAME='skyrl-self-critique-grpo-qwen2.5-7b-v4'
# EXPERIMENT_NAME='skyrl-self-critique-grpo-qwen2.5-14b-v4'

# path for dataset (.parquet files) containing the prompts and metadata for each question
DATA_DIR="path_to_train_data"

# change environment.skyrl_gym.search.search_url to your own search server
uv run --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/combined_train_45k_train_converted.parquet']" \
  data.val_data="['${DATA_DIR}/combined_train_45k_test_converted.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.loss_reduction="seq_mean_token_sum_norm" \
  trainer.policy.model.path="${BASE_MODEL}" \
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
  environment.skyrl_gym.search.search_url="http://192.168.0.5:8010/retrieve" \
  environment.skyrl_gym.search.topk=3 \
  environment.skyrl_gym.search.feedback_model="self" \
  environment.skyrl_gym.search.feedback_url=null \
  environment.skyrl_gym.search.mask_zero_format=false \
  environment.skyrl_gym.search.feedback_penalty=0.8 \
  environment.skyrl_gym.search.enable_feedback_reward=true \
  environment.skyrl_gym.search.feedback_coef=0.5 \
  trainer.logger="wandb" \
  trainer.project_name="${WAND_PROJECT}" \
  trainer.run_name="${EXPERIMENT_NAME}" \
  trainer.ckpt_interval=25 \
  trainer.hf_save_interval=25 \
  trainer.max_ckpts_to_keep=999999 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="path_to_ckpt_save_dir" \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=25 \
  trainer.dump_data_batch=true \
  trainer.dump_eval_results=true \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  trainer.export_path="path_to_hf_export_dir (e.g., {trainer.ckpt_path}/exports)" \
  2>&1 | tee $EXPERIMENT_NAME.log
  
