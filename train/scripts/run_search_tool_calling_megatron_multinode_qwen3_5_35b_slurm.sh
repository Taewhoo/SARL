#!/bin/bash

LOG_DIR="../logs"
RETRIEVER_PORT=8010

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=pt2-taewhoo-sarl-tc-qwen3.5-35b
#SBATCH --partition=normal
#SBATCH --nodelist=Slurm-GPU-Node-[1-49]
#SBATCH --exclude=Slurm-GPU-Node-10
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=${LOG_DIR}/train_tool_calling_megatron_qwen3_5_35b.log
#SBATCH --error=${LOG_DIR}/train_tool_calling_megatron_qwen3_5_35b.log

set -e

echo "Allocated nodes:"
echo \$SLURM_JOB_NODELIST

nodes=(\$(scontrol show hostnames \$SLURM_JOB_NODELIST))
head=\${nodes[0]}

echo "Head node: \$head"

# Clean stale Ray sessions on all nodes
echo "Cleaning stale Ray caches on all nodes..."
for node in \${nodes[@]}; do
    srun --nodes=1 --ntasks=1 -w "\$node" bash -c "
        ray stop 2>/dev/null || true
        rm -rf /tmp/ray /tmp/flashinfer_jit_*
    " &
done
wait
echo "Ray caches cleaned."

# Launch retriever on head node
echo "Launching retriever on \$head:${RETRIEVER_PORT}..."

bash -c "
cd /mnt/weka/post_training_tmp/pt2-search-agent/SARL/wiki_server
source /mnt/weka/post_training_tmp/pt2-search-agent/miniconda3/etc/profile.d/conda.sh
conda activate retriever
bash launch_local_server.sh ${RETRIEVER_PORT} e5_addr_dir
" > ${LOG_DIR}/retriever_qwen3_5_35b_\${SLURM_JOB_ID}.log 2>&1 &

echo "Waiting for retriever on \$head:${RETRIEVER_PORT}..."

until curl -sf -X POST "http://\$head:${RETRIEVER_PORT}/retrieve" \
    -H "Content-Type: application/json" \
    -d '{"queries": ["test"], "topk": 1}' >/dev/null 2>&1; do
    sleep 5
done

echo "Retriever ready"

# Launch training (the training script handles Ray cluster setup internally)
echo "Starting Qwen3.5-35B-A3B tool-calling Megatron training..."

cd /mnt/weka/post_training_tmp/pt2-search-agent/SARL/train
bash scripts/run_search_tool_calling_megatron_multinode_qwen3_5_35b.sh

EOF
