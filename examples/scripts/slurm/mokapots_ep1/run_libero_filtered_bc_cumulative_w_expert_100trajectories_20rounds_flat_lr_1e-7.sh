#!/bin/bash
#SBATCH --job-name=fbc_mokapots_ep1_100t_flr1e7 # Job name
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:1                           # GPUs per node
#SBATCH --cpus-per-task=12                     # CPU cores per task
#SBATCH --mem=128G                             # Memory per node
#SBATCH --time=48:00:00                        # Walltime (hh:mm:ss)
#SBATCH --partition=general                    # Partition/queue name
#SBATCH --output=/data/user_data/skowshik/parl_logs/logs/filtered_bc_cumul_expert_libero_%x_%j.out
#SBATCH --error=/data/user_data/skowshik/parl_logs/logs/filtered_bc_cumul_expert_libero_%x_%j.err

# =============================================================================
# LIBERO: Filtered BC — Mokapots ep1, 10k checkpoint, 100 traj/round, 20 rounds, flat LR 1e-7
# =============================================================================
#
# Usage:
#   sbatch examples/scripts/slurm/mokapots_ep1/run_libero_filtered_bc_cumulative_w_expert_100trajectories_20rounds_flat_lr_1e-7.sh
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
export EXP=/data/hf_cache/models/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# -------------------------------
# Launch Filtered BC Training (cumulative + expert data)
# -------------------------------
python -m examples.launch_train_sim_base_policy_distillation \
    --env libero \
    --prefix filtered_bc_pi05-mokaPots-ep1-10k-cumul-expert-20rounds-100traj-flat_lr_1e-7 \
    --wandb_project ${proj_name} \
    --seed 0 \
    \
    --pi_05_config pi05_libero_custom_low_mem_ep1_discrete_state_input_False_10k \
    --pi_05_ckpt_dir /data/hf_cache/models/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep1_bs32_v2_icml/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep1_bs32_v2_icml-v1/10000/ \
    \
    --chunk_len 10 \
    --query_freq 10 \
    \
    --num_rounds 20 \
    --num_collect_trajectories 100 \
    --num_train_steps_per_round 500 \
    --batch_size 16 \
    \
    --eval_episodes 50 \
    --log_interval 50 \
    --checkpoint_interval 10000 \
    \
    --drop_short_actions 1 \
    --cumulative_data 1 \
    --load_expert_data 1 \
    --expert_data_path /home/skowshik/vla/codebase/openpi/data_dumps/filtered_put_both_moka_pots_on_the_stove_ep1.json \
    --flat_lr 1e-7
