#!/bin/bash
#SBATCH --job-name=rsac_small_actor         # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # GPUs per node
#SBATCH --cpus-per-task=12                 # CPU cores per task
#SBATCH --mem=128G                         # Memory per node
#SBATCH --time=48:00:00                    # Walltime (hh:mm:ss)
#SBATCH --partition=general                # Partition/queue name
#SBATCH --output=/data/user_data/skowshik/parl_logs/logs/rsac_small_actor_%x_%j.out
#SBATCH --error=/data/user_data/skowshik/parl_logs/logs/rsac_small_actor_%x_%j.err

# =============================================================================
# LIBERO: Residual SAC — smaller actor MLP (256,256 instead of 512,512,512)
# =============================================================================
#
# Usage:
#   sbatch examples/scripts/slurm/run_libero_residual_sac_small_actor.sh
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
proj_name=libero-residual-sac-small-actor
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
# Launch Residual SAC Training
# -------------------------------
python -m examples.launch_train_sim_residual \
    --algorithm residual_sac \
    --env libero \
    --prefix rsac-small-actor-mokaPots-4k-vlm-a-exec \
    --wandb_project ${proj_name} \
    --seed 0 \
    --batch_size 64 \
    --discount 0.999 \
    --max_steps 2500000 \
    --multi_grad_step 1 \
    --encoder_type small \
    --start_online_updates 500 \
    \
    --eval_interval 10000 \
    --eval_episodes 50 \
    --log_interval 500 \
    --checkpoint_interval 100000 \
    --diagnostic_freq 5 \
    \
    --resize_image 100 \
    --query_freq 10 \
    --chunk_len 10 \
    --action_magnitude 1.0 \
    --hidden_dims 512 \
    --actor_hidden_dims 256,256 \
    \
    --pi_05_config pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k \
    --pi_05_ckpt_dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml-v1/4000/ \
    \
    --residual_alpha 0.5 \
    --predict_a_exec 1 \
    --use_zero_residual_initially 1 \
    \
    --num_critic_updates 20 \
    --num_actor_updates 10 \
    --use_huber_loss 0 \
    --huber_delta 1.0 \
    --max_grad_norm 1.0 \
    \
    --target_entropy -200.0 \
    --learn_std 1 \
    --log_std_min -5.0 \
    --log_std_max 2.0 \
    --backup_entropy 0 \
    \
    --bc_reg_coeff 0.0 \
    --bc_on_success_only 0 \
    --success_buffer_ratio 0.0 \
    --success_buffer_min_size 100 \
    --bc_warmup_steps 10000 \
    --bc_warmup_num_critic_updates 8 \
    --bc_warmup_num_actor_updates 4 \
    \
    --tau 0.05 \
    --add_states 1 \
    --reward_type sparse \
    --use_vlm_embedding 1
    # To resume from checkpoint, add:
    # --restore_path /path/to/checkpoint_dir
