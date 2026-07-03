#!/bin/bash
# Usage: ./submit_with_reservation.sh <job.sbatch>
# Submits the given sbatch script to Slurm-GPU-Node-[80-84] and waits for it to finish.

set -euo pipefail

NODES="Slurm-GPU-Node-[80-84]"
PARTITION="omni"

# --- Resolve sbatch script ---
if [[ -n "${1:-}" ]]; then
    SBATCH_SCRIPT="$1"
else
    read -rp "Enter path to sbatch script: " SBATCH_SCRIPT
fi
SBATCH_SCRIPT="$(realpath "$SBATCH_SCRIPT")"
[[ -f "$SBATCH_SCRIPT" ]] || { echo "Error: '$SBATCH_SCRIPT' not found."; exit 1; }

SBATCH_BASENAME="$(basename "$SBATCH_SCRIPT" .sbatch)"

cancel_job() {
    echo ""
    echo "Interrupted — cancelling job $JOB_ID ..."
    scancel "$JOB_ID" && echo "Job $JOB_ID cancelled." || echo "Warning: could not cancel job $JOB_ID."
    exit 1
}

# --- Submit the job ---
echo "Submitting $SBATCH_SCRIPT ..."
JOB_ID=$(sbatch --parsable "$SBATCH_SCRIPT")

echo "Job submitted — ID: $JOB_ID"
trap cancel_job INT TERM
echo "Logs: $(dirname "$SBATCH_SCRIPT")/${SBATCH_BASENAME}_${JOB_ID}.log"

# --- Poll until the job is no longer in the queue ---
echo ""
echo "Waiting for job $JOB_ID to finish..."
while squeue -j "$JOB_ID" -h &>/dev/null; do
    STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
    echo "  [$(date '+%H:%M:%S')] Job $JOB_ID state: $STATE"
    sleep 30
done

# Check final state via sacct
FINAL_STATE=$(sacct -j "$JOB_ID" --format=State --noheader | head -1 | tr -d ' ')
echo ""
echo "Job $JOB_ID finished with state: $FINAL_STATE"
