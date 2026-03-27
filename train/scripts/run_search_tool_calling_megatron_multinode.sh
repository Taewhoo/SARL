#!/bin/bash
set -x

# Multi-node Megatron GRPO training+generation — tool-calling search agent with MoE models
#
# Combines:
#   - Tool-calling search environment (search_tool_calling env class)
#   - Megatron backend with 5D parallelism for MoE models (expert parallelism)
#
# The model is given two explicit tools instead of XML-style <search>/<answer> tags:
#   - local_knowledge_base_retrieval(user_query)
#   - get_document(docid)
#
# Key differences from run_search_tool_calling_multinode.sh:
#   - trainer.strategy=megatron  (instead of fsdp2)
#   - trainer.policy.megatron_config.*  for TP/PP/CP/EP/ETP parallelism
#   - generator.inference_engine.*  namespace (megatron-style)
#   - Default model: Qwen/Qwen3-30B-A3B  (MoE, requires EP)
#
# Key differences from run_search_megatron.sh:
#   - environment.env_class="search_tool_calling"  (tool-calling instead of search)
#   - stop string: "</tool_call>"  (no </answer>)
#   - generator.use_conversation_multi_turn=true
#   - generator.sampling_params.max_generate_length=1024
#   - All environment.skyrl_gym.search_tool_calling.* config
#
# Requires the Megatron Docker image:
#   novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8-megatron
#
# Usage (from the scripts/ directory):
#   sbatch --job-name=pt2-taewhoo-sarl-tc-megatron --partition=omni \
#     --nodelist=Slurm-GPU-Node-[80-84] --nodes=4 --gres=gpu:8 \
#     --cpus-per-task=128 \
#     --output=../logs/train_tool_calling_megatron_qwen3_30b.log \
#     --error=../logs/train_tool_calling_megatron_qwen3_30b.log \
#     run_search_tool_calling_megatron_multinode.sh

# Feedback Model Configuration:
# - To use the model itself for feedback (default):
#   environment.skyrl_gym.search_tool_calling.feedback_model="self"
# - To use an external vLLM endpoint:
#   environment.skyrl_gym.search_tool_calling.feedback_model="Qwen/Qwen3-235B-A22B-Instruct-2507"
#   environment.skyrl_gym.search_tool_calling.feedback_url="http://192.168.0.30:8080/v1"

# Tool Configuration:
# - get_document uses search_url with /retrieve → /access substitution by default.
# - To set a custom access endpoint:
#   environment.skyrl_gym.search_tool_calling.doc_access_url="http://host:8010/access"

############################################
# 0.       Environment Setup               #
############################################

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
fi

unset VIRTUAL_ENV

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export WANDB_API_KEY="de8b3531e996caccab64ea27fd8113822bc379f5"

# Multi-node NCCL settings
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export FLASHINFER_JIT_DIR="${FLASHINFER_JIT_DIR:-/tmp/flashinfer_jit_${SLURM_JOB_ID:-$$}}"
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1

############################################
# 1.       Experiment Parameters           #
############################################

WAND_PROJECT='NQ-HotpotQA-ASearcher-45k-clean'
BASE_MODEL='Qwen/Qwen3-30B-A3B'
EXPERIMENT_NAME='skyrl-tool-calling-megatron-grpo-qwen3-30b-a3b-v1'

DATA_DIR="/mnt/weka/post_training_tmp/pt2-search-agent/SARL/train/train_data"

############################################
# 2.       Multi-Node Configuration        #
############################################

NNODES=${SLURM_JOB_NUM_NODES:-4}
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-128}

export RAY_NUM_NODES=$NNODES

# Megatron parallelism config for Qwen3-30B-A3B (MoE)
# DP=4, TP=2, EP=8 as recommended in docs for this model
MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

# Inference engine config: fewer engines with high TP for generation
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TP=8

echo "[INFO] NNODES=$NNODES"
echo "[INFO] GPUS_PER_NODE=$GPUS_PER_NODE"
echo "[INFO] CPUS_PER_TASK=$CPUS_PER_TASK"
echo "[INFO] SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "[INFO] MEGATRON: TP=$MEGATRON_TP PP=$MEGATRON_PP CP=$MEGATRON_CP EP=$MEGATRON_EP ETP=$MEGATRON_ETP"

############################################
# 3.       Start Ray Head Node             #
############################################

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')

port=6379
ip_head=$head_node_ip:$port
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$port
export ip_head

echo "[INFO] Starting Ray HEAD at $head_node ($ip_head)"

srun --nodes=1 --ntasks=1 -w "$head_node" \
    uv run --frozen -- ray start --head --node-ip-address="$head_node_ip" \
    --port=$port --num-cpus "$CPUS_PER_TASK" --num-gpus "$GPUS_PER_NODE" --block &

echo "[INFO] Waiting for Ray head node to become ready..."
for attempt in $(seq 1 30); do
    if uv run --frozen -- ray status 2>/dev/null | grep -q "node"; then
        echo "[INFO] Ray head node is ready (attempt $attempt)."
        break
    fi
    if [[ $attempt -eq 30 ]]; then
        echo "[ERROR] Ray head node did not start after 60s. Exiting."
        exit 1
    fi
    sleep 2
done

############################################
# 4.       Start Ray Worker Nodes          #
############################################

worker_num=$((NNODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "[INFO] Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" bash -c "
        until nvidia-smi > /dev/null 2>&1; do
            echo \"Waiting for GPU visibility on \$HOSTNAME...\"
            sleep 2
        done
        uv run --frozen -- ray start --address \"$ip_head\" \
            --num-cpus \"$CPUS_PER_TASK\" --num-gpus \"$GPUS_PER_NODE\" --block
    " &
done

############################################
# 5.       Verify Ray Cluster              #
############################################

EXPECTED_GPUS=$((NNODES * GPUS_PER_NODE))
echo "[INFO] Waiting for $EXPECTED_GPUS GPUs across $NNODES nodes..."

for attempt in $(seq 1 60); do
    CLUSTER_GPUS=$(uv run --frozen -- python -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
print(int(ray.cluster_resources().get('GPU', 0)))
" 2>/dev/null)
    if [[ "$CLUSTER_GPUS" == "$EXPECTED_GPUS" ]]; then
        echo "[INFO] All $EXPECTED_GPUS GPUs available in Ray cluster."
        break
    fi
    if [[ $attempt -eq 60 ]]; then
        echo "[ERROR] Only ${CLUSTER_GPUS:-0}/$EXPECTED_GPUS GPUs after 300s. Exiting."
        exit 1
    fi
    echo "[INFO] Cluster has ${CLUSTER_GPUS:-0}/$EXPECTED_GPUS GPUs (attempt $attempt)..."
    sleep 5
done

uv run --frozen -- python -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
print('Cluster resources:', ray.cluster_resources())
"

############################################
# 6.       Launch Training                 #
############################################

echo "[INFO] Launching tool-calling Megatron training on $NNODES nodes with $EXPECTED_GPUS total GPUs..."

uv run --frozen --extra megatron -m skyrl_train.entrypoints.main_base \
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
  trainer.strategy=megatron \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.placement.policy_num_gpus_per_node=$GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.inference_engine.async_engine=true \
  generator.apply_overlong_filtering=true \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=32768 \
  generator.sampling_params.max_generate_length=1024 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=5 \
  generator.max_turns=32 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</tool_call>"]' \
  environment.env_class="search_tool_calling" \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.search_tool_calling.log_requests=false \
  environment.skyrl_gym.search_tool_calling.search_url="http://${head_node_ip}:8010/retrieve" \
  environment.skyrl_gym.search_tool_calling.topk=3 \
  environment.skyrl_gym.search_tool_calling.timeout=30 \
  environment.skyrl_gym.search_tool_calling.feedback_model="self" \
  environment.skyrl_gym.search_tool_calling.feedback_url=null \
  environment.skyrl_gym.search_tool_calling.mask_zero_format=false \
  environment.skyrl_gym.search_tool_calling.feedback_penalty=0.8 \
  environment.skyrl_gym.search_tool_calling.enable_feedback_reward=true \
  environment.skyrl_gym.search_tool_calling.feedback_coef=0.5 \
  trainer.logger="wandb" \
  trainer.project_name="${WAND_PROJECT}" \
  trainer.run_name="${EXPERIMENT_NAME}" \
  trainer.ckpt_interval=25 \
  trainer.hf_save_interval=25 \
  trainer.max_ckpts_to_keep=999999 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="/mnt/weka/post_training_tmp/pt2-search-agent/SARL/ckpt_tool_calling_megatron_qwen3-30b-a3b" \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=25 \
  trainer.dump_data_batch=true \
  trainer.dump_eval_results=true \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</tool_call>"]' \
  trainer.export_path="/mnt/weka/post_training_tmp/pt2-search-agent/SARL/ckpt_tool_calling_megatron_qwen3-30b-a3b/exports" \
  2>&1 | tee $EXPERIMENT_NAME.log
