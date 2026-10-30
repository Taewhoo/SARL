#!/bin/bash

# Submit from the train/ directory so SLURM_SUBMIT_DIR points there (where pyproject.toml lives)
cd "$(dirname "$0")/.."

LOG_DIR="logs"
RETRIEVER_PORT=8010

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sarl-32b
#SBATCH --partition=<partition>
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=${LOG_DIR}/train_qwen2_5_32b_multinode.log
#SBATCH --error=${LOG_DIR}/train_qwen2_5_32b_multinode.log

set -e

echo "Allocated nodes:"
echo \$SLURM_JOB_NODELIST

nodes=(\$(scontrol show hostnames \$SLURM_JOB_NODELIST))
head=\${nodes[0]}

echo "Head node: \$head"

# Clean stale Ray sessions on all allocated nodes
echo "Cleaning stale Ray caches on all nodes..."
for node in "\${nodes[@]}"; do
    srun --nodes=1 --ntasks=1 -w "\$node" bash -c "ray stop 2>/dev/null; rm -rf /tmp/ray/session_* /tmp/flashinfer_jit_*" &
done
wait
echo "Ray caches cleaned."

echo "Launching retriever natively on \$head..."

bash -c "
cd path_to_SARL/wiki_server
source path_to_miniconda3/etc/profile.d/conda.sh
conda activate retriever
bash launch_local_server.sh ${RETRIEVER_PORT} e5_addr_dir
" > ${LOG_DIR}/launch_local_server_8010.log 2>&1 &

echo "Waiting for retriever on \$head:${RETRIEVER_PORT}..."

until curl -s "http://\$head:${RETRIEVER_PORT}" >/dev/null; do
    sleep 2
done

echo "Retriever ready"

echo "Starting training..."

cd path_to_SARL/train
bash scripts/run_search_self_critique_v4_multinode_32b.sh

EOF
