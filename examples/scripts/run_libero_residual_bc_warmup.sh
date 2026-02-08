#!/bin/bash
#SBATCH --job-name=residual_sac_pi0      # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # GPUs per node
#SBATCH --cpus-per-task=12                 # CPU cores per task
#SBATCH --mem=128G                         # Memory per node
#SBATCH --time=48:00:00                    # Walltime (hh:mm:ss)
#SBATCH --partition=general                # Partition/queue name
#SBATCH --output=/data/user_data/sreyasv/r-sac-bc-w/logs/residual_sac_libero_pi0_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/sreyasv/r-sac/logs-bc-w/residual_sac_libero_pi0_%x_%j.err    # Stderr log

# -------------------------------
# Environment setup
# -------------------------------
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0

mkdir -p /data/user_data/sreyasv/r-sac/logs/

# -------------------------------
# Configuration
# -------------------------------
proj_name=debug-libero-residual-sac-f-a
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export EXP=/data/user_data/sreyasv/dsrl_exp/logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

pip install mujoco==3.3.1

# -------------------------------
# Launch Residual SAC Training
# -------------------------------
# note current script will work only with chunk len and query freq being equal
python -m examples.launch_train_sim_residual \
    --algorithm residual_sac \
    --env libero \
    --prefix r-sac-a-exec_pi5_2moka-pots-ckpt-crct-bc-wrrm-up \
    --wandb_project ${proj_name} \
    --batch_size 256 \
    --discount 0.999 \
    --seed 0 \
    --max_steps 1000000 \
    --eval_interval 20000 \
    --log_interval 500 \
    --checkpoint_interval 500000 \
    --eval_episodes 10 \
    --multi_grad_step 10 \
    --start_online_updates 500 \
    --resize_image 100 \
    --action_magnitude 1.0 \
    --query_freq 10 \
    --hidden_dims 128 \
    --pi_05_config pi05_libero_finetuned_two_moka_pots \
    --pi_05_ckpt_dir /data/hf_cache/models/pi05_libero_ep5_mokapots_4k/ \
    --residual_alpha 1.0 \
    --chunk_len 10 \
    --use_zero_residual_initially 1 \
    --target_entropy -105.0 \
    --num_critic_updates 1 \
    --num_actor_updates 8 \
    --use_huber_loss 0 \
    --huber_delta 1.0 \
    --bc_reg_coeff 0.02 \
    --bc_on_success_only 1 \
    --success_buffer_ratio 0.2 \
    --success_buffer_min_size 300 \
    --predict_a_exec 1 \
    --bc_warmup_steps 50000 \
    --bc_warmup_num_critic_updates 10 \
    --bc_warmup_num_actor_updates 5 \
