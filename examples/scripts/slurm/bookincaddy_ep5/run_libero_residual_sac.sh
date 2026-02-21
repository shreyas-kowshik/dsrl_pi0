#!/bin/bash
#SBATCH --job-name=residual_sac_bookcaddy_ep5    # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # GPUs per node
#SBATCH --cpus-per-task=12                 # CPU cores per task
#SBATCH --mem=128G                         # Memory per node
#SBATCH --time=48:00:00                    # Walltime (hh:mm:ss)
#SBATCH --partition=general                # Partition/queue name
#SBATCH --output=/data/user_data/skowshik/r_sac_best/logs/residual_sac_libero_%x_%j.out
#SBATCH --error=/data/user_data/skowshik/r_sac_best/logs/residual_sac_libero_%x_%j.err

# =============================================================================
# LIBERO: Residual SAC — Book Caddy ep5, 4k checkpoint
# =============================================================================
#
# Usage:
#   sbatch examples/scripts/slurm/bookincaddy_ep5/run_libero_residual_sac.sh
# =============================================================================

# -------------------------------
# Environment setup
# -------------------------------
source /data/user_data/skowshik/anaconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0

mkdir -p /data/user_data/skowshik/r_sac_best/logs/

# -------------------------------
# Configuration
# -------------------------------
proj_name=libero-residual-sac
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export EXP=/data/user_data/skowshik/r_sac_best/logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# -------------------------------
# Launch Residual SAC Training
# -------------------------------
python -m examples.launch_train_sim_residual \
    --algorithm residual_sac \
    --env libero \
    --prefix r-sac-a-exec_pi5_bookCaddy-ep5-4k-vlm-embedding-bc-warmup \
    --wandb_project ${proj_name} \
    --seed 0 \
    --batch_size 64 \
    --discount 0.999 \
    --max_steps 1000000 \
    --multi_grad_step 1 \
    \
    --eval_interval 20000 \
    --eval_episodes 50 \
    --log_interval 500 \
    --checkpoint_interval 25000 \
    --diagnostic_freq 5 \
    \
    --start_online_updates 500 \
    --resize_image 100 \
    --query_freq 10 \
    --chunk_len 10 \
    --action_magnitude 1.0 \
    --hidden_dims 512 \
    \
    --pi_05_config pi05_libero_custom_low_mem_ep5_bookcaddy_discrete_state_input_False_4k \
    --pi_05_ckpt_dir /data/hf_cache/models/pi05_libero_lora_vision_fullft_action_placebookincaddy_task_ep5_bs32_v2_icml/pi05_libero_lora_vision_fullft_action_placebookincaddy_task_ep5_bs32_v2_icml-v1/4000/ \
    \
    --residual_alpha 0.5 \
    --predict_a_exec 0 \
    --use_zero_residual_initially 1 \
    \
    --num_critic_updates 20 \
    --num_actor_updates 10 \
    --use_huber_loss 0 \
    --huber_delta 1.0 \
    \
    --target_entropy -35.0 \
    --learn_std 1 \
    \
    --bc_reg_coeff 1.0 \
    --bc_on_success_only 1 \
    --success_buffer_ratio 0.20 \
    --success_buffer_min_size 300 \
    --bc_warmup_steps 10000 \
    --bc_warmup_num_critic_updates 8 \
    --bc_warmup_num_actor_updates 4 \
    \
    --tau 0.05 \
    --use_vlm_embedding 1 \
    --task_suite_name libero_10 \
    --task_id 5
