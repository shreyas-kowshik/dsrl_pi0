#!/bin/bash
#SBATCH --job-name=cartpole_all            # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # GPUs per node
#SBATCH --cpus-per-task=12                 # CPU cores per task
#SBATCH --mem=128G                         # Memory per node
#SBATCH --time=48:00:00                    # Walltime (hh:mm:ss)
#SBATCH --partition=general                # Partition/queue name
#SBATCH --output=/data/user_data/sreyasv/cartpole-all/logs/cartpole_all_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/sreyasv/cartpole-all/logs/cartpole_all_%x_%j.err    # Stderr log

# =============================================================================
# CartPole Test: Run ALL algorithms sequentially
# =============================================================================
# Tests sac, residual_sac, q_weighted_pg, and residual_grpo on CartPole.
# No Pi-0.5 model or LIBERO/ALOHA dependencies needed.
#
# Usage:
#   bash examples/scripts/run_cartpole_all.sh
#   sbatch examples/scripts/run_cartpole_all.sh
# =============================================================================

set -e

# -------------------------------
# Environment setup
# -------------------------------
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0

mkdir -p /data/user_data/sreyasv/cartpole-all/logs/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  CartPole Integration Test - All Algorithms"
echo "============================================================"
echo ""

echo ">>> [1/4] Plain SAC"
bash "${SCRIPT_DIR}/run_cartpole_sac.sh"
echo ""

echo ">>> [2/4] Residual SAC"
bash "${SCRIPT_DIR}/run_cartpole_residual_sac.sh"
echo ""

echo ">>> [3/4] Q-Weighted Policy Gradient"
bash "${SCRIPT_DIR}/run_cartpole_qwpg.sh"
echo ""

echo ">>> [4/4] Residual GRPO"
bash "${SCRIPT_DIR}/run_cartpole_grpo.sh"
echo ""

echo "============================================================"
echo "  All CartPole tests completed successfully!"
echo "============================================================"
