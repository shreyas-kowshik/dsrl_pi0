#!/bin/bash
#SBATCH --job-name=fbc_2k_pi05                         # Job name
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:1                           # GPUs per node
#SBATCH --cpus-per-task=12                     # CPU cores per task
#SBATCH --mem=128G                             # Memory per node
#SBATCH --time=48:00:00                        # Walltime (hh:mm:ss)
#SBATCH --partition=general                    # Partition/queue name
#SBATCH --output=/data/user_data/skowshik/parl_logs/logs/filtered_bc_2k_libero_%x_%j.out
#SBATCH --error=/data/user_data/skowshik/parl_logs/logs/filtered_bc_2k_libero_%x_%j.err

# =============================================================================
# LIBERO: Filtered Behavior Cloning (non-cumulative) - 2k steps
# =============================================================================
#
# Non-cumulative filtered BC: only successful trajectories from the current
# round are used for training (no data retention across rounds).
# Uses 2000 train steps per round and collects 50 trajectories per round.
#
# Usage:
#   sbatch examples/scripts/slurm/run_libero_filtered_bc_2k.sh
# =============================================================================

# -------------------------------
# Environment setup
# -------------------------------
source /data/user_data/skowshik/anaconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0

mkdir -p /data/user_data/skowshik/parl_logs/logs/

# -------------------------------
# Configuration
# -------------------------------
proj_name=libero-filtered-bc
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export EXP=/data/user_data/skowshik/parl_exp/logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# -------------------------------
# Launch Filtered BC Training (non-cumulative, 2k steps)
# -------------------------------
python -m examples.launch_train_sim_base_policy_distillation \
    --env libero \
    --prefix filtered_bc_pi05-mokaPots-2k \
    --wandb_project ${proj_name} \
    --seed 0 \
    \
    --pi_05_config pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k \
    --pi_05_ckpt_dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml-v1/4000/ \
    \
    --chunk_len 10 \
    --query_freq 10 \
    \
    --num_rounds 500 \
    --num_collect_trajectories 50 \
    --num_train_steps_per_round 2000 \
    --batch_size 16 \
    \
    --eval_episodes 50 \
    --log_interval 50 \
    --checkpoint_interval 5000000 \
    \
    --drop_short_actions 1 \
    --cumulative_data 0
