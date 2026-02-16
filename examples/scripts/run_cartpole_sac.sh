#!/bin/bash
#SBATCH --job-name=cartpole_sac            # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # GPUs per node
#SBATCH --cpus-per-task=12                 # CPU cores per task
#SBATCH --mem=128G                         # Memory per node
#SBATCH --time=48:00:00                    # Walltime (hh:mm:ss)
#SBATCH --partition=general                # Partition/queue name
#SBATCH --output=/data/user_data/skowshik/cartpole-sac/logs/cartpole_sac_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/skowshik/cartpole-sac/logs/cartpole_sac_%x_%j.err    # Stderr log

# =============================================================================
# CartPole Test: Plain SAC (no residual, no base policy)
# =============================================================================
# Standard SAC on CartPole — no residual composition, no Pi-0.5.
# Actor output IS the executed action directly.
#
# Usage:
#   bash examples/scripts/run_cartpole_sac.sh
#   sbatch examples/scripts/run_cartpole_sac.sh
# =============================================================================

set -e

# -------------------------------
# Environment setup
# -------------------------------
# source /home/skowshik/miniconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0

mkdir -p /data/user_data/skowshik/cartpole-sac/logs/

# -------------------------------
# Configuration
# -------------------------------
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export EXP=${EXP:-/data/user_data/skowshik/dsrl_exp/logs/cartpole-sac}
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

proj_name=test-cartpole-sac

# -------------------------------
# Launch Plain SAC Training
# -------------------------------
python -m examples.launch_train_sim_residual \
    --algo sac \
    --algorithm sac \
    --env cartpole \
    --prefix cartpole-sac \
    --wandb_project ${proj_name} \
    --batch_size 16 \
    --discount 0.99 \
    --seed 0 \
    --max_steps 100000 \
    --eval_interval 5000 \
    --log_interval 500 \
    --checkpoint_interval -1 \
    --eval_episodes 5 \
    --multi_grad_step 5 \
    --start_online_updates 200 \
    --resize_image 64 \
    --action_magnitude 1.0 \
    --query_freq 1 \
    --chunk_len 1 \
    --residual_alpha 0.0 \
    --use_zero_residual_initially 0 \
    --predict_a_exec 1 \
    --target_entropy -0.5 \
    --num_critic_updates 1 \
    --num_actor_updates 1 \
    --hidden_dims 128 \
    --num_qs 2 \
    --cartpole_horizon 100 \
    --reward_type dense \
    --learn_std 1 \

echo "=== CartPole Plain SAC test finished ==="
