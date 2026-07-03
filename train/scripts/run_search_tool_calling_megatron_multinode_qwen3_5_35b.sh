#!/bin/bash
set -x

# Multi-node FSDP2 GRPO training+generation — tool-calling search agent
# Model: Qwen/Qwen3.5-35B-A3B (MoE, 256 experts, 8 active, ~3B active params)
#
# Architecture notes (vs Qwen3-30B-A3B):
#   - 256 experts (vs 128), 8 active per token
#   - 40 layers (vs 48), hidden_size=2048
#   - Hybrid attention: linear + full (every 4th layer)
#   - moe_intermediate_size=512 (vs 768), shared_expert_intermediate_size=512
#   - Multimodal (vision encoder present but unused for text-only training)
#   - 262K max context, vocab_size=248320
#   - model_type: qwen3_5_moe (requires transformers >= 5.0, vllm >= 0.18.0)
#
# Tool-calling environment: search_tool_calling (Qwen format)
#   - search(query)  /  browse(url)
#   - Stop token: </tool_call>
#   - Answer format: "Exact Answer: {answer}"
#
# Environment: .venv-qwen35 (pip-installed, separate from main .venv)
#   torch==2.10.0+cu128, vllm==0.18.0, transformers==5.4.0
#
# Usage:
#   sbatch --job-name=pt2-taewhoo-sarl-tc-qwen3.5-35b --partition=normal \
#     --nodes=2 --gres=gpu:8 --cpus-per-task=128 \
#     --output=../logs/train_tool_calling_megatron_qwen3_5_35b.log \
#     --error=../logs/train_tool_calling_megatron_qwen3_5_35b.log \
#     run_search_tool_calling_megatron_multinode_qwen3_5_35b.sh

############################################
# 0.       Environment Setup               #
############################################

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
fi

TRAIN_DIR=/mnt/weka/post_training_tmp/pt2-search-agent/SARL/train
source "$TRAIN_DIR/.venv-qwen35/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY="de8b3531e996caccab64ea27fd8113822bc379f5"
export HF_TOKEN="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"

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
BASE_MODEL='/mnt/weka/post_training_tmp/pt2-search-agent/models/Qwen3.5-35B-A3B'
EXPERIMENT_NAME='skyrl-tool-calling-megatron-grpo-qwen3.5-35b-a3b-v1'

DATA_DIR="/mnt/weka/post_training_tmp/pt2-search-agent/SARL/train/train_data"

############################################
# 2.       Multi-Node Configuration        #
############################################

NNODES=${SLURM_JOB_NUM_NODES:-2}
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-128}

export RAY_NUM_NODES=$NNODES

# Inference engine config: num_engines * TP must equal total GPUs when colocate_all=true
NUM_INFERENCE_ENGINES=$((NNODES * GPUS_PER_NODE / 2))
INFERENCE_ENGINE_TP=2

echo "[INFO] NNODES=$NNODES"
echo "[INFO] GPUS_PER_NODE=$GPUS_PER_NODE"
echo "[INFO] CPUS_PER_TASK=$CPUS_PER_TASK"
echo "[INFO] SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

############################################
# 3.       Start Ray Head Node             #
############################################

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | tr ' ' '\n' | grep '^10\.' | head -1)

port=6379
ip_head=$head_node_ip:$port
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$port
export ip_head

echo "[INFO] Starting Ray HEAD at $head_node ($ip_head)"

srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" \
    --port=$port --num-cpus "$CPUS_PER_TASK" --num-gpus "$GPUS_PER_NODE" --block &

echo "[INFO] Waiting for Ray head node to become ready..."
for attempt in $(seq 1 30); do
    if ray status 2>/dev/null | grep -q "node"; then
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
        source \"$TRAIN_DIR/.venv-qwen35/bin/activate\"
        until nvidia-smi > /dev/null 2>&1; do
            echo \"Waiting for GPU visibility on \$HOSTNAME...\"
            sleep 2
        done
        ray start --address \"$ip_head\" \
            --num-cpus \"$CPUS_PER_TASK\" --num-gpus \"$GPUS_PER_NODE\" --block
    " &
done

############################################
# 5.       Verify Ray Cluster              #
############################################

EXPECTED_GPUS=$((NNODES * GPUS_PER_NODE))
echo "[INFO] Waiting for $EXPECTED_GPUS GPUs across $NNODES nodes..."

for attempt in $(seq 1 60); do
    CLUSTER_GPUS=$(python -c "
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

python -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
print('Cluster resources:', ray.cluster_resources())
"

############################################
# 6.       Launch Training                 #
############################################

echo "[INFO] Launching tool-calling FSDP2 training on $NNODES nodes with $EXPECTED_GPUS total GPUs..."

python -m skyrl_train.entrypoints.main_base \
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
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.use_sample_packing=false \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.policy_num_gpus_per_node=$GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$GPUS_PER_NODE \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.gpu_memory_utilization=0.3 \
  +generator.engine_init_kwargs.language_model_only=true \
  generator.async_engine=true \
  generator.apply_overlong_filtering=true \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
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
  trainer.ckpt_path="/mnt/weka/post_training_tmp/pt2-search-agent/SARL/ckpt_tool_calling_qwen3.5-35b-a3b" \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=25 \
  trainer.dump_data_batch=true \
  trainer.dump_eval_results=true \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</tool_call>"]' \
  trainer.export_path="/mnt/weka/post_training_tmp/pt2-search-agent/SARL/ckpt_tool_calling_qwen3.5-35b-a3b/exports" \
  2>&1 | tee $EXPERIMENT_NAME.log
