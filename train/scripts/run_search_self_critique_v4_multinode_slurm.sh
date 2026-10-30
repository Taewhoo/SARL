#!/bin/bash

LOG_DIR="../logs"
RETRIEVER_PORT=8010

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sarl
#SBATCH --partition=<partition>
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=${LOG_DIR}/train_qwen2_5_32b_multinode4.log
#SBATCH --error=${LOG_DIR}/train_qwen2_5_32b_multinode4.log

set -e

echo "Allocated nodes:"
echo \$SLURM_JOB_NODELIST

nodes=(\$(scontrol show hostnames \$SLURM_JOB_NODELIST))
head=\${nodes[0]}

echo "Head node: \$head"

echo "Launching retriever natively on \$head..."

bash -c "
cd path_to_SARL/wiki_server
source path_to_miniconda3/etc/profile.d/conda.sh
conda activate retriever
# Optional: Add CUDA_VISIBLE_DEVICES=\"\" before bash if the retriever is CPU-only
bash launch_local_server.sh ${RETRIEVER_PORT} e5_addr_dir
" > ${LOG_DIR}/launch_local_server_8010.log 2>&1 &

echo "Waiting for retriever on \$head:${RETRIEVER_PORT}..."

until curl -s "http://\$head:${RETRIEVER_PORT}" >/dev/null; do
    sleep 2
done

echo "Retriever ready"

echo "Starting training..."

cd path_to_SARL/train
bash scripts/run_search_self_critique_v4_multinode.sh

EOF
