#!/bin/bash
#SBATCH -p omni
#SBATCH -J test
#SBATCH -w Slurm-GPU-Node-81
#SBATCH --nodes=1
#SBATCH --time=00:05:00

hostname
nvidia-smi
sleep 60