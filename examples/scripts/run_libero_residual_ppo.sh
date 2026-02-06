#!/bin/bash
#SBATCH --job-name=residual_qpg_pi0       # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # GPUs per node
#SBATCH --cpus-per-task=12                 # CPU cores per task
#SBATCH --mem=128G                         # Memory per node
#SBATCH --time=48:00:00                    # Walltime (hh:mm:ss)
#SBATCH --partition=general                # Partition/queue name
#SBATCH --output=/data/user_data/sreyasv/qpg_logs/logs/residual_qpg_libero_pi0_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/sreyasv/qpg_logs/logs/residual_qpg_libero_pi0_%x_%j.err    # Stderr log

# -------------------------------
# Environment setup
# -------------------------------
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0

mkdir -p /data/user_data/sreyasv/qpg_logs/logs/

# -------------------------------
# Configuration
# -------------------------------
proj_name=libero-residual-q_weighted_pg
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export EXP=/data/user_data/sreyasv/qpg_exp/logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

pip install mujoco==3.3.1

# -------------------------------
# Launch Q-weighted PG Training
# -------------------------------
# Algorithm options: residual_sac, q_weighted_pg, residual_grpo
# - q_weighted_pg: Uses raw Q as advantage (no baseline)
# - residual_grpo: Uses Q - mean(Q) as advantage (GRPO baseline) 
# start_online_updates=500 and batch_size=256 eval_episodes=10
python -m examples.launch_train_sim_residual \
    --algorithm q_weighted_pg \
    --algo q_weighted_pg \
    --env libero \
    --prefix residual_qpg-pi05-new_ckpt_mokaPots-4k-huber-loss \
    --wandb_project ${proj_name} \
    --batch_size 256 \
    --discount 0.999 \
    --seed 0 \
    --max_steps 1000000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --checkpoint_interval 500000 \
    --eval_episodes 10 \
    --multi_grad_step 20 \
    --start_online_updates 500 \
    --resize_image 100 \
    --action_magnitude 1.0 \
    --query_freq 10 \
    --hidden_dims 128 \
    --pi_05_config pi05_libero_finetuned_two_moka_pots \
    --pi_05_ckpt_dir /data/hf_cache/models/pi05_libero_ep5_mokapots_4k/ \
    --residual_alpha 0.1 \
    --chunk_len 10 \
    --use_zero_residual_initially 1 \
    --grpo_num_samples 8 \
    --clip_epsilon 0.2 \
    --entropy_coeff 1e-3 \
    --advantage_critic_reduction mean \
    --log_ratio_clip 20.0 \
    --log_prob_clip 50.0 \
    --max_grad_norm 1.0 \
    --use_huber_loss 1 \
    --num_critic_updates 2 \
    --num_actor_updates 4
