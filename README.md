<div align="center">

# DSRL for π₀: Diffusion Steering via Reinforcement Learning

## [[website](https://diffusion-steering.github.io)]      [[paper](https://arxiv.org/abs/2506.15799)]

</div>


## Overview
This repository provides the official implementation for our paper: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (CoRL 2025).

Specifically, it contains a JAX-based implementation of DSRL (Diffusion Steering via Reinforcement Learning) for steering a pre-trained generalist policy, [π₀](https://github.com/Physical-Intelligence/openpi), across various environments, including:

- **Simulation:** Libero, Aloha  
- **Real Robot:** Franka

If you find this repository useful for your research, please cite:

```
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## Installation
1. Create a conda environment:
```
conda create -n dsrl_pi0 python=3.11.11
conda activate dsrl_pi0
```

2. Clone this repo with all submodules
```
git clone git@github.com:nakamotoo/dsrl_pi0.git --recurse-submodules
cd dsrl_pi0
```

3. Install all packages and dependencies
```
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.0"

# install openpi
pip install -e openpi
pip install -e openpi/packages/openpi-client

# install Libero
pip install -e LIBERO
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu # needed for libero

# install LIBERO-PRO (OOD evaluation benchmark)
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git LIBERO-PRO
echo "# LIBERO-PRO package" > LIBERO-PRO/libero/__init__.py  # required for editable install
pip install -e LIBERO-PRO
```

## Training (Simulation)
Libero
```
bash examples/scripts/run_libero.sh
```
Aloha
```
bash examples/scripts/run_aloha.sh
```
### Training Logs
We provide sample W&B runs and logs: https://wandb.ai/mitsuhiko/DSRL_pi0_public

## Training (Real)
For real-world experiments, we use the remote hosting feature from pi0 (see [here](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)) which enables us to host the pi0 model on a higher-spec remote server, in case the robot's client machine is not powerful enough. 

0. Setup Franka robot and install DROID package [[link](https://github.com/droid-dataset/droid.git)]

1. [On the remote server] Host pi0 droid model on your remote server
```
cd openpi && python scripts/serve_policy.py --env=DROID
```
2. [On your robot client machine] Run DSRL
```
bash examples/scripts/run_real.sh
```

## Visualize Diagnostics
```
python3 -m examples.visualize_diagnostics \
    --diagnostics_dir /data/user_data/skowshik/libero-residual-sac-eval/r-sac-a-exec_pi5_2moka-pots-ckpt-vlm-embedding-bc-warmup_2026_02_16_18_46_11_0000--s-0/checkpoint25000/diagnostics \
    --port 8502

```

## LIBERO-PRO: OOD Evaluation with Position Displacement

[LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) extends the LIBERO benchmark with five generalization dimensions: object appearance, **position displacement**, language paraphrasing, task redefinition, and environment swap. This section describes how to evaluate using the **position displacement** perturbation.

### How position displacement works

Position displacement shifts the initial placement regions of task objects in the BDDL scene definition. The perturbation is specified via `--use_swap` (swaps two objects' start positions using `LIBERO-PRO/libero_ood/ood_spatial_relation.yaml`) or through combined perturbation modes. When `setup_libero_pro_env` is called, it:

1. Reads `LIBERO-PRO/evaluation_config.yaml` for OOD config paths.
2. Applies the enabled perturbation(s) to every `.bddl` file in the task suite directory, writing perturbed files to a `*_temp` sibling directory.
3. Generates new init states via `LIBERO-PRO/notebooks/generate_init_states.py`.
4. Returns the perturbed suite name (e.g. `libero_10_swap`) for the benchmark lookup.

The `LIBERO-PRO/evaluation_config.yaml` controls which OOD YAML files are used for each perturbation type.

### Running position-displacement evaluation

The path below re-uses the same checkpoint and Pi-0.5 config as the standard LIBERO evaluation but loads tasks from the LIBERO-PRO position-displaced benchmark (`--use_swap 1`):

```bash
bash examples/scripts/evaluate/evaluate_libero_residual_sac.sh \
    --task_suite_name libero_10 \
    --task_id 8 \
    --use_swap 1
```

Or directly:

```bash
python -m examples.evaluation.evaluate_residual_sac \
    --checkpoint_dir /path/to/checkpoint \
    --output_dir    /path/to/output \
    --num_evals 50 \
    --env libero \
    --task_suite_name libero_10 \
    --task_id 8 \
    --use_swap 1 \
    --pi_05_config pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k \
    --pi_05_ckpt_dir /path/to/pi05_ckpt \
    --algo residual_sac \
    --residual_alpha 0.5 \
    --hidden_dims 512 \
    --use_vlm_embedding 1
```

### Configuration file

`LIBERO-PRO/evaluation_config.yaml` sets the paths for each OOD config:

```yaml
bddl_files_path: "./LIBERO-PRO/libero/libero/bddl_files/"
script_path:     "./LIBERO-PRO/notebooks/generate_init_states.py"
init_file_dir:   "./LIBERO-PRO/libero/libero/init_files/"

use_environment: false
use_swap:        false   # set to true for position displacement
use_object:      false
use_language:    false
use_task:        false

ood_task_configs:
  swap:        "./LIBERO-PRO/libero_ood/ood_spatial_relation.yaml"
  object:      "./LIBERO-PRO/libero_ood/ood_object.yaml"
  language:    "./LIBERO-PRO/libero_ood/ood_language.yaml"
  task:        "./LIBERO-PRO/libero_ood/ood_task.yaml"
  environment: "./LIBERO-PRO/libero_ood/ood_environment.yaml"
```

Pass a custom config path with `--eval_config_path`.

## Credits
This repository is built upon [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR) repositories. 
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at nakamoto\[at\]berkeley\[dot\]edu 



```
# Via shell script (auto-discovers OUTPUT_DIR)
python -m examples.visualize_rollouts \
    --sh_path examples/scripts/evaluate/evaluate_libero_pro_base_vision_pre_trained.sh \
    --port 8502

# Or point directly at the output directory
python -m examples.visualize_rollouts \
    --output_dir /data/user_data/skowshik/libero-base-eval/pi05_libero_custom_low_mem_ep5_bookcaddy_discrete_state_input_False_4k-500_horizon/videos/ \
    --port 8505

```

# Assets movement
```
path="/data/hf_cache/models/pi05_libero_lora_vision_fullft_action_placebookincaddy_task_ep5_bs32_v2_icml/pi05_libero_lora_vision_fullft_action_placebookincaddy_task_ep5_bs32_v2_icml-v1/4000/" && mkdir -p "$path/assets/libero" && cp "$path/assets/physical-intelligence/libero/norm_stats.json" "$path/assets/libero/norm_stats.json"

```
