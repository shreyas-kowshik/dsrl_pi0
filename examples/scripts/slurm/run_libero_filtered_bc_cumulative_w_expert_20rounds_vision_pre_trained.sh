#!/bin/bash
#SBATCH --job-name=fbc_cumul_expert_pi05       # Job name
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:1                           # GPUs per node
#SBATCH --cpus-per-task=12                     # CPU cores per task
#SBATCH --mem=128G                             # Memory per node
#SBATCH --time=48:00:00                        # Walltime (hh:mm:ss)
#SBATCH --partition=general                    # Partition/queue name
#SBATCH --output=/data/user_data/skowshik/parl_logs/logs/filtered_bc_cumul_expert_libero_%x_%j.out
#SBATCH --error=/data/user_data/skowshik/parl_logs/logs/filtered_bc_cumul_expert_libero_%x_%j.err

# =============================================================================
# LIBERO: Filtered Behavior Cloning with Cumulative Data + Expert Demos
# =============================================================================
#
# Same as filtered BC with cumulative_data=1, plus expert trajectories loaded
# from the LeRobot dataset (via JSON dumps in openpi/data_dumps/).  Expert
# trajectories are always kept in the training buffer as successful data.
#
# Usage:
#   sbatch examples/scripts/slurm/run_libero_filtered_bc_cumulative_w_expert.sh
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
    --prefix filtered_bc_pi05-mokaPots-cumul-expert-20rounds-vision-pre-trained \
    --wandb_project ${proj_name} \
    --seed 0 \
    \
    --pi_05_config pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k_vision_init_fullft_action_4k \
    --pi_05_ckpt_dir /data/hf_cache/models/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml_init_vision_full_data_trained/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml_init_vision_full_data_trained-v1/4000/ \
    \
    --chunk_len 10 \
    --query_freq 10 \
    \
    --num_rounds 10 \
    --num_collect_trajectories 20 \
    --num_train_steps_per_round 500 \
    --batch_size 16 \
    \
    --eval_episodes 50 \
    --log_interval 50 \
    --checkpoint_interval 1 \
    \
    --drop_short_actions 1 \
    --cumulative_data 1 \
    --load_expert_data 1 \
    --expert_data_path /home/skowshik/vla/codebase/openpi/data_dumps
