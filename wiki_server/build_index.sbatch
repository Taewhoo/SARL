#!/bin/bash
#SBATCH --job-name=wiki_build_index
#SBATCH --partition=omni
#SBATCH --nodelist=Slurm-GPU-Node-[75-90]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/weka/post_training_tmp/pt2-search-agent/logs/wiki26_index_%j.log
#SBATCH --error=/mnt/weka/post_training_tmp/pt2-search-agent/logs/wiki26_index_%j.log

source activate retriever

bash build_index.sh
