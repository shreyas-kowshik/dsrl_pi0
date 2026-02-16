#!/bin/bash
#SBATCH --job-name=pi_05_dsrl             # Job name
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:1                           # GPUs per node
#SBATCH --cpus-per-task=12                      # CPU cores per task
#SBATCH --mem=128G                              # Memory per node
#SBATCH --time=48:00:00                        # Walltime (hh:mm:ss)
#SBATCH --partition=general                    # Partition/queue name
#SBATCH --output=/data/user_data/skowshik/dsrl_logs/logs/dsrl_libero_10_pi05_put_the_two_mocha_pots_on_the_stove-clean-new-ckpt_%x_%j.out   # Stdout log
#SBATCH --error=/data/user_data/skowshik/dsrl_logs/logs/dsrl_libero_10_pi05_put_the_two_mocha_pots_on_the_stove-clean-new-ckpt_%x_%j.err    # Stderr log

# -------------------------------
# Environment setup
# -------------------------------
# source /home/skowshik/miniconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0     # use the correct env (adjust if it's 'seer' or something else)

mkdir -p /data/user_data/skowshik/dsrl_logs/logs/
# -------------------------------
proj_name=DSRL_pi05_Libero
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export EXP=/data/user_data/skowshik/dsrl_exp/logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# pip install mujoco==3.3.1

python -m examples.launch_train_sim \
--algorithm pixel_sac \
--env libero \
--prefix dsrl_pi05_libero_finetuned_put-both-moka-pots-on-stove-clean-new-ckpt-vlm-state \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 50000 \
--log_interval 500 \
--checkpoint_interval 500000 \
--eval_episodes 50 \
--multi_grad_step 20 \
--start_online_updates 500 \
--resize_image 100 \
--action_magnitude 1.0 \
--query_freq 8 \
--hidden_dims 128 \
--pi_05_config pi05_libero_finetuned_two_moka_pots \
--pi_05_ckpt_dir  /data/hf_cache/models/pi05_libero_ep5_mokapots_4k/ \
--use_vlm_embedding 1 \
