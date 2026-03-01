#!/bin/bash
#SBATCH --job-name=eval_ep1_bc_pair1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/skowshik/r_sac_best/logs/eval_ep1_bc_pair1_%j.out
#SBATCH --error=/data/user_data/skowshik/r_sac_best/logs/eval_ep1_bc_pair1_%j.err

# Usage: sbatch examples/scripts/slurm/eval_pairs/bookincaddy_ep1_pair1.sh
# Runs: bookincaddy_ep1_chkpt/evaluate_libero_base_4000
#       bookincaddy_ep1_chkpt/evaluate_libero_base_8000

ROOT_DIR="/home/skowshik/vla/codebase/dsrl_pi0"
EVAL_DIR="$ROOT_DIR/evals"
mkdir -p "$EVAL_DIR"

# Output file: evals/<parent_dir>_<script_name>.txt
run() {
    local script="$1"
    local tag="$(basename "$(dirname "$script")")_$(basename "$script" .sh)"
    echo "===== Starting $tag =====" | tee -a "$EVAL_DIR/$tag.txt"
    bash "$script" 2>&1 | tee -a "$EVAL_DIR/$tag.txt"
    echo "===== Finished $tag =====" | tee -a "$EVAL_DIR/$tag.txt"
}

run "$ROOT_DIR/examples/scripts/evaluate/bookincaddy_ep1_chkpt/evaluate_libero_base_4000.sh"
run "$ROOT_DIR/examples/scripts/evaluate/bookincaddy_ep1_chkpt/evaluate_libero_base_8000.sh"
