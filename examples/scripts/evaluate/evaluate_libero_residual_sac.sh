#!/bin/bash
#SBATCH --job-name=eval_residual_sac          # Job name
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gres=gpu:1                          # GPUs per node
#SBATCH --cpus-per-task=12                    # CPU cores per task
#SBATCH --mem=128G                            # Memory per node
#SBATCH --time=4:00:00                        # Walltime (hh:mm:ss)
#SBATCH --partition=general                   # Partition/queue name
#SBATCH --output=/data/user_data/skowshik/r_sac_best/logs/eval_residual_sac_%x_%j.out
#SBATCH --error=/data/user_data/skowshik/r_sac_best/logs/eval_residual_sac_%x_%j.err

# =============================================================================
# LIBERO: Evaluate Residual SAC checkpoint
# =============================================================================
#
# Usage:
#   sbatch examples/scripts/slurm/evaluate_libero_residual_sac.sh
#
# To point at a different checkpoint or output directory, edit the
# CHECKPOINT_DIR and OUTPUT_DIR variables below.
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
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# -------------------------------
# Paths — edit these
# -------------------------------
# Directory produced by training (contains checkpoint_* files)
CHECKPOINT_DIR=/data/user_data/skowshik/r_sac_best/logs/libero-residual-sac/r-sac-a-exec_pi5_2moka-pots-ckpt-vlm-embedding-bc-warmup_2026_02_16_18_46_11_0000--s-0/checkpoint75000/

# Where to write summary.csv and diagnostic videos
OUTPUT_DIR=/data/user_data/skowshik/libero-residual-sac-eval/r-sac-a-exec_pi5_2moka-pots-ckpt-vlm-embedding-bc-warmup_2026_02_16_18_46_11_0000--s-0/checkpoint75000/

mkdir -p "$OUTPUT_DIR"

# -------------------------------
# Launch evaluation
# -------------------------------
python -m examples.evaluation.evaluate_residual_sac \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --num_evals     50 \
    --diagnostic_freq 5 \
    \
    --env libero \
    --seed 0 \
    --resize_image 100 \
    --query_freq 10 \
    --chunk_len 10 \
    \
    --pi_05_config pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k \
    --pi_05_ckpt_dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml-v1/4000/ \
    \
    --algo residual_sac \
    --residual_alpha 0.5 \
    --predict_a_exec 0 \
    \
    --hidden_dims 512 \
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
    \
    --tau 0.05 \
    --use_vlm_embedding 1
