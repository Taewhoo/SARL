#!/bin/bash

# Submit from the train/ directory so SLURM_SUBMIT_DIR points there (where pyproject.toml lives)
cd "$(dirname "$0")/.."

LOG_DIR="logs"
RETRIEVER_PORT=8010

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=solar-open-sarl
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=${LOG_DIR}/train_llama3_8b_instruct_searchr1.log
#SBATCH --error=${LOG_DIR}/train_llama3_8b_instruct_searchr1.log

set -e

echo "Allocated node:"
echo \$SLURM_JOB_NODELIST

head=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -1)

echo "Head node: \$head"

# Clean stale Ray sessions
echo "Cleaning stale Ray caches..."
ray stop 2>/dev/null || true
rm -rf /tmp/ray /tmp/flashinfer_jit_*
echo "Ray caches cleaned."

echo "Launching retriever on \$head..."

bash -c "
cd path_to_SARL/wiki_server
source path_to_miniconda3/etc/profile.d/conda.sh
conda activate retriever
bash launch_local_server.sh ${RETRIEVER_PORT} e5_addr_dir
" > ${LOG_DIR}/launch_local_server_8010_llama_instruct.log 2>&1 &

echo "Waiting for retriever on \$head:${RETRIEVER_PORT}..."

until curl -s "http://\$head:${RETRIEVER_PORT}" >/dev/null; do
    sleep 2
done

echo "Retriever ready"

echo "Starting Llama-3.1-8B-Instruct training..."

cd path_to_SARL/train
export RETRIEVER_HOST=\$head
bash scripts/run_search_grpo_llama3_8b_instruct.sh

EOF
