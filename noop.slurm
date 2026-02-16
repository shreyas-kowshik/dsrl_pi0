#!/bin/bash
#SBATCH --job-name=dsrl-pi0-residual-parl
#SBATCH --partition=general          # partition name
#SBATCH --qos=normal            # qos
#SBATCH --gres=gpu:A6000:1              # request 2 GPUs
#SBATCH --cpus-per-task=16          # number of CPUs (increased)
#SBATCH --time=48:00:00          # time (1 day)
#SBATCH --mem=128G                   # 128GB total RAM (increased from 48GB)
#SBATCH --exclude=babel-v9-20,babel-t9-16,babel-w9-16,babel-w9-18
#SBATCH --output=/data/user_data/skowshik/rlvr-debug/outcome_%j.out      # Standard output file (%j is job ID)
#SBATCH --error=/data/user_data/skowshik/rlvr-debug/outcome_%j.out       # Standard error file (%j is job ID)

conda run -n openpi python noop.py