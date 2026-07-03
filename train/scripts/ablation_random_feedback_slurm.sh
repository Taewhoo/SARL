#!/bin/bash

# Submit from the train/ directory so SLURM_SUBMIT_DIR points there (where pyproject.toml lives)
cd "$(dirname "$0")/.."

LOG_DIR="logs"
RETRIEVER_PORT=8010

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=pt2-taewhoo-rand-fb
#SBATCH --partition=normal
#SBATCH --nodelist=Slurm-GPU-Node-[1-49]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=${LOG_DIR}/ablation_random_feedback.log
#SBATCH --error=${LOG_DIR}/ablation_random_feedback.log

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
cd /mnt/weka/post_training_tmp/pt2-search-agent/SARL/wiki_server
source /mnt/weka/post_training_tmp/pt2-search-agent/miniconda3/etc/profile.d/conda.sh
conda activate retriever
bash launch_local_server.sh ${RETRIEVER_PORT} e5_addr_dir
" > ${LOG_DIR}/launch_local_server_ablation_random_feedback.log 2>&1 &

echo "Waiting for retriever on \$head:${RETRIEVER_PORT}..."

until curl -s "http://\$head:${RETRIEVER_PORT}" >/dev/null; do
    sleep 2
done

echo "Retriever ready"

echo "Starting random feedback ablation training..."

cd /mnt/weka/post_training_tmp/pt2-search-agent/SARL/train
export RETRIEVER_HOST=\$head
bash scripts/ablation_random_feedback.sh

EOF
