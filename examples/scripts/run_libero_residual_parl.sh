#!/bin/bash
#SBATCH --job-name=residual_parl_pi0      # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # GPUs per node
#SBATCH --cpus-per-task=12                 # CPU cores per task
#SBATCH --mem=128G                         # Memory per node
#SBATCH --time=48:00:00                    # Walltime (hh:mm:ss)
#SBATCH --partition=general                # Partition/queue name
#SBATCH --output=/data/user_data/sreyasv/parl_logs/logs/residual_parl_libero_pi0_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/sreyasv/parl_logs/logs/residual_parl_libero_pi0_%x_%j.err    # Stderr log

# =============================================================================
# LIBERO: Residual PA-RL (Policy-Agnostic RL) with Pi-0.5
# =============================================================================
# Best-of-N sampling from actor + base policy → Q-evaluation → top-K elites
# → gradient ascent on Q w.r.t. actions → MSE distillation back to actor.
# Base policy actions are included as candidates by default, ensuring the
# residual never degrades below base policy quality.
#
# Usage:
#   bash examples/scripts/run_libero_residual_parl.sh
#   sbatch examples/scripts/run_libero_residual_parl.sh
# =============================================================================

# -------------------------------
# Environment setup
# -------------------------------
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0

mkdir -p /data/user_data/sreyasv/parl_logs/logs/

# -------------------------------
# Configuration
# -------------------------------
proj_name=libero-residual-parl
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export EXP=/data/user_data/sreyasv/parl_exp/logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

pip install mujoco==3.3.1

# -------------------------------
# Launch Residual PA-RL Training
# -------------------------------
python -m examples.launch_train_sim_residual \
    --algorithm residual_parl \
    --algo residual_parl \
    --env libero \
    --prefix residual_parl_pi05-mokaPots-4k-vlm-a-exec \
    --wandb_project ${proj_name} \
    --batch_size 64 \
    --discount 0.999 \
    --seed 0 \
    --max_steps 2500000 \
    --eval_interval 25000 \
    --log_interval 500 \
    --checkpoint_interval 100000 \
    --eval_episodes 50 \
    --multi_grad_step 1 \
    --encoder_type small \
    --start_online_updates 500 \
    --resize_image 100 \
    --action_magnitude 1.0 \
    --query_freq 10 \
    --hidden_dims 512 \
    --pi_05_config pi05_libero_finetuned_two_moka_pots \
    --pi_05_ckpt_dir /data/hf_cache/models/pi05_libero_ep5_mokapots_4k/ \
    --residual_alpha 0.5 \
    --chunk_len 10 \
    --use_zero_residual_initially 1 \
    --use_huber_loss 0 \
    --num_critic_updates 20 \
    --num_actor_updates 10 \
    --bc_reg_coeff 0.0 \
    --bc_on_success_only 0 \
    --success_buffer_ratio 0.0 \
    --success_buffer_min_size 100 \
    --predict_a_exec 1 \
    --parl_num_samples 16 \
    --parl_num_elites 8 \
    --parl_num_grad_steps 30 \
    --parl_step_size 0.001 \
    --bc_warmup_steps 10000 \
    --bc_warmup_num_critic_updates 8 \
    --bc_warmup_num_actor_updates 4 \
    --tau 0.05 \
    --use_vlm_embedding 1 \
