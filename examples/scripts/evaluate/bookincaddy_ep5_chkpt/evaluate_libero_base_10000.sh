#!/bin/bash
#SBATCH --job-name=eval_libero_base_bookcaddy_10k # Job name
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --gres=gpu:1                            # GPUs per node
#SBATCH --cpus-per-task=12                      # CPU cores per task
#SBATCH --mem=128G                              # Memory per node
#SBATCH --time=4:00:00                          # Walltime (hh:mm:ss)
#SBATCH --partition=general                     # Partition/queue name
#SBATCH --output=/data/user_data/skowshik/r_sac_best/logs/eval_libero_base_%x_%j.out
#SBATCH --error=/data/user_data/skowshik/r_sac_best/logs/eval_libero_base_%x_%j.err

# =============================================================================
# LIBERO: Evaluate base policy (Pi-0.5) — Book Caddy ep5, checkpoint 10000
# =============================================================================
#
# Usage:
#   sbatch examples/scripts/evaluate/bookincaddy_ep5_chkpt/evaluate_libero_base_10000.sh
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
# Copy norm_stats
# -------------------------------
CKPT_DIR="/data/hf_cache/models/pi05_libero_lora_vision_fullft_action_placebookincaddy_task_ep5_bs32_v2_icml/pi05_libero_lora_vision_fullft_action_placebookincaddy_task_ep5_bs32_v2_icml-v1/10000/"
mkdir -p "${CKPT_DIR}/assets/libero"
cp "${CKPT_DIR}/assets/physical-intelligence/libero/norm_stats.json" "${CKPT_DIR}/assets/libero/norm_stats.json"

# -------------------------------
# Paths — edit these
# -------------------------------
OUTPUT_DIR=/data/user_data/skowshik/libero-base-eval/pi05_libero_custom_low_mem_ep5_bookcaddy_discrete_state_input_False_10k-500_horizon/

mkdir -p "$OUTPUT_DIR"

# -------------------------------
# Launch evaluation
# -------------------------------
python -m examples.evaluation.evaluate_base \
    --output_dir    "$OUTPUT_DIR" \
    --num_evals     50 \
    --max_env_steps 500 \
    \
    --env libero \
    --seed 0 \
    --resize_image 100 \
    --query_freq 10 \
    --chunk_len 10 \
    \
    --pi_05_config pi05_libero_custom_low_mem_ep5_bookcaddy_discrete_state_input_False_10k \
    --pi_05_ckpt_dir /data/hf_cache/models/pi05_libero_lora_vision_fullft_action_placebookincaddy_task_ep5_bs32_v2_icml/pi05_libero_lora_vision_fullft_action_placebookincaddy_task_ep5_bs32_v2_icml-v1/10000/ \
    \
    --task_suite_name libero_10 \
    --task_id 5
