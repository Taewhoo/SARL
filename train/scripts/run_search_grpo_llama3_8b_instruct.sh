set -x

# Ablation: Standard Search-R1 GRPO for Llama-3.1-8B-Instruct
#
# Based on run_search.sh with only these changes:
#   - Model: Llama-3.1-8B-Instruct (+ chat template)
#   - Data: combined_train_45k
#   - Batch sizes: 128/64/128 (train/mini/eval)
#   - max_turns: 32
#   - Eval/save intervals: 25/5/25
#   - disable_feedback=true (no SARL)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export NCCL_IBEXT_DISABLE=1
export FLASHINFER_JIT_DIR="${FLASHINFER_JIT_DIR:-/tmp/flashinfer_jit_$$}"
export WANDB_API_KEY="de8b3531e996caccab64ea27fd8113822bc379f5"
export HF_TOKEN="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"

WAND_PROJECT='NQ-HotpotQA-ASearcher-45k-clean'
BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
EXPERIMENT_NAME='skyrl-search-ppo-llama3.1-8b-instruct-ablation'

DATA_DIR="/mnt/weka/post_training_tmp/pt2-search-agent/SARL/train/train_data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE_FILE="${SCRIPT_DIR}/llama3_instruct_chat_template.j2"

USE_TIS=true
TIS_IMP_RATIO_CAP=2.0

uv run --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/combined_train_45k_train_converted.parquet']" \
  data.val_data="['${DATA_DIR}/combined_train_45k_test_converted.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=94 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.use_tis=$USE_TIS \
  trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
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
  trainer.ckpt_path="/mnt/weka/post_training_tmp/pt2-search-agent/SARL/ckpt_ppo_ablation_llama3.1-8b-instruct" \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=25 \
  trainer.dump_data_batch=true \
  trainer.dump_eval_results=true \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  trainer.export_path="/mnt/weka/post_training_tmp/pt2-search-agent/SARL/ckpt_ppo_ablation_llama3.1-8b-instruct/exports" \
  2>&1 | tee $EXPERIMENT_NAME.log
