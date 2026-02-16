# Parameter Reference: `run_libero_residual_parl.sh`

Residual PA-RL (Policy-Agnostic RL) with Pi-0.5 on LIBERO.

---

## SLURM Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--job-name` | `residual_parl_pi0` | Name of the SLURM job, used in log filenames and scheduler listings. |
| `--nodes` | `1` | Number of compute nodes to allocate. |
| `--gres=gpu` | `1` | Number of GPUs requested per node. |
| `--cpus-per-task` | `12` | CPU cores allocated to the task (for data loading, env stepping, etc.). |
| `--mem` | `128G` | Total RAM per node. |
| `--time` | `48:00:00` | Maximum wallclock time before the job is killed. |
| `--partition` | `general` | SLURM partition/queue to submit the job to. |
| `--output` | `.../%x_%j.out` | Path for stdout logs. `%x` = job name, `%j` = job ID. |
| `--error` | `.../%x_%j.err` | Path for stderr logs. `%x` = job name, `%j` = job ID. |

---

## Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `proj_name` | `libero-residual-parl` | Project identifier used for WandB and experiment log paths. |
| `device_id` | `0` | GPU device index used for CUDA, MuJoCo EGL rendering, and `CUDA_VISIBLE_DEVICES`. |
| `DISPLAY` | `:0` | X11 display variable (needed by some rendering backends). |
| `MUJOCO_GL` | `egl` | MuJoCo rendering backend; EGL enables headless GPU rendering. |
| `PYOPENGL_PLATFORM` | `egl` | PyOpenGL platform matching the MuJoCo GL backend. |
| `MUJOCO_EGL_DEVICE_ID` | `$device_id` | Which GPU EGL should use for offscreen rendering. |
| `OPENPI_DATA_HOME` | `/data/hf_cache/pi-models/openpi` | Root directory where OpenPi model weights/data are cached. |
| `EXP` | `/data/.../logs/$proj_name` | Base directory for experiment logs and checkpoints. |
| `CUDA_VISIBLE_DEVICES` | `$device_id` | Restricts PyTorch/JAX to the specified GPU. |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Disables JAX/XLA GPU memory pre-allocation (allocates on demand). |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.9` | Maximum fraction of GPU memory JAX is allowed to use. |

---

## Training Script Parameters

### Algorithm & Environment

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--algorithm` | `residual_sac` | `residual_parl` | Algorithm type string used for WandB run naming and logging. |
| `--algo` | `residual_sac` | `residual_parl` | Algorithm selector that determines which learner class is instantiated. Options: `sac`, `residual_sac`, `q_weighted_pg`, `residual_grpo`, `residual_parl`. |
| `--env` | `libero` | `libero` | Simulation environment to use. Options include `libero`, `aloha_cube`, `cartpole`. |
| `--prefix` | `''` | `residual_parl_pi05-mokaPots-4k-vlm-a-exec` | String prefix prepended to the WandB run name for easy filtering. |
| `--wandb_project` | `residual_sac_sim` | `libero-residual-parl` | WandB project name under which runs are grouped. |

### Core Training

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--batch_size` | `16` | `64` | Mini-batch size sampled from the replay buffer per gradient step. |
| `--discount` | `0.999` | `0.999` | Discount factor (gamma) for future rewards in the Bellman equation. Values close to 1.0 encourage long-horizon credit assignment. |
| `--seed` | `42` | `0` | Random seed for reproducibility across env resets, network init, and sampling. |
| `--max_steps` | `1000000` | `2500000` | Total number of environment interaction steps before training terminates. |
| `--start_online_updates` | `1000` | `500` | Number of environment steps to collect (random exploration) before starting gradient updates. |
| `--multi_grad_step` | `1` | `1` | Number of gradient update steps per environment step (update-to-data ratio). |
| `--tau` | `0.005` | `0.05` | Soft target network update coefficient. Controls how quickly the target Q-network tracks the online Q-network via Polyak averaging: `target = tau * online + (1 - tau) * target`. |

### Logging & Evaluation

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--eval_interval` | `5000` | `25000` | Number of env steps between evaluation rollouts. |
| `--log_interval` | `1000` | `500` | Number of env steps between WandB/console metric logging. |
| `--checkpoint_interval` | `-1` | `100000` | Number of env steps between saving model checkpoints. `-1` disables periodic checkpointing. |
| `--eval_episodes` | `10` | `50` | Number of full episodes rolled out per evaluation cycle. |

### Network Architecture

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--encoder_type` | `small` | `small` | Vision encoder architecture for processing image observations. `small` uses a lightweight CNN. |
| `--hidden_dims` | `(256, 256, 256)` | `512` | Hidden layer dimensions for actor and critic MLPs. Accepts one or more integers. A single value creates a single hidden layer of that width. |
| `--resize_image` | `-1` | `100` | Target size (pixels) for resizing input images before the encoder. `-1` means no resizing. |

### Pi-0.5 Base Policy

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--pi_05_config` | `''` | `pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k` | Configuration name for the Pi-0.5 base policy model (determines architecture and input settings). |
| `--pi_05_ckpt_dir` | `''` | `...putbothmokapots.../4000/` | Path to the fine-tuned Pi-0.5 checkpoint directory to load as the frozen base policy. |
| `--query_freq` | `-1` | `10` | How often (in env steps) to query the Pi-0.5 base policy for a new action chunk. Should match `chunk_len`. |

### Residual Policy

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--residual_alpha` | `0.1` | `0.5` | Scaling factor applied to the residual action before adding to the base policy action: `a_exec = a_base + alpha * a_residual`. Controls the magnitude of corrections the residual policy can make. |
| `--chunk_len` | `10` | `10` | Action chunk length matching the Pi-0.5 prediction horizon. The residual policy produces corrections for this many timesteps at once. |
| `--action_magnitude` | `1.0` | `1.0` | Clipping bound for the magnitude of residual actions. Constrains how large corrections can be. |
| `--use_zero_residual_initially` | `1` | `1` | Whether to initialize the first trajectory's residual to zero (1=yes). Ensures the very first rollout uses pure base policy actions for a clean baseline. |
| `--predict_a_exec` | `0` | `1` | When 1, the actor directly predicts the executed action `a_exec` rather than a residual delta. Changes the actor's output interpretation. |

### Critic & Actor Updates

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--use_huber_loss` | `0` | `0` | Whether to use Huber loss (robust to outliers) instead of MSE for the critic TD error. 1=yes, 0=no. |
| `--num_critic_updates` | `2` | `20` | Number of critic gradient updates per training batch. More updates give a better Q-function estimate before updating the actor. |
| `--num_actor_updates` | `4` | `10` | Number of actor gradient updates per training batch. The actor is distilled from PARL-refined target actions via MSE. |

### BC (Behavioral Cloning) Regularization

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--bc_reg_coeff` | `0.0` | `0.0` | Coefficient for BC regularization loss added to the actor objective. Penalizes deviation from base policy actions. `0.0` disables it. |
| `--bc_on_success_only` | `0` | `0` | When 1, BC regularization loss is computed only on transitions from successful episodes. 0=use all transitions. |

### Success Buffer

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--success_buffer_ratio` | `0.0` | `0.0` | Fraction of each actor training batch drawn from the success-only replay buffer. `0.0` disables success buffer sampling. |
| `--success_buffer_min_size` | `100` | `100` | Minimum number of transitions in the success buffer before it is used for sampling. Prevents training on too few examples. |

### PARL (Policy-Agnostic RL) Sampling & Refinement

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--parl_num_samples` | `16` | `16` | **N**: Number of action candidates sampled from the actor (+ base policy) during Best-of-N selection. More samples improve coverage of the action space. |
| `--parl_num_elites` | `4` | `8` | **K**: Number of top-scoring actions (by Q-value) kept as elite candidates for gradient refinement. |
| `--parl_num_grad_steps` | `5` | `30` | Number of gradient ascent steps applied to elite actions to maximize Q-value. More steps yield better-refined targets but increase compute cost. |
| `--parl_step_size` | `0.01` | `0.001` | Learning rate (step size) for the gradient ascent on actions w.r.t. the Q-function. Smaller values give more stable refinement. |

### BC Warmup Phase

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--bc_warmup_steps` | `0` | `10000` | Number of initial gradient steps spent on BC warmup before switching to full RL. During warmup the actor is trained to clone the base policy while the critic is pre-trained. `0` disables warmup. |
| `--bc_warmup_num_critic_updates` | `10` | `8` | Number of critic updates per gradient step during the BC warmup phase. Aggressive critic training builds a good Q-function early. |
| `--bc_warmup_num_actor_updates` | `1` | `4` | Number of actor BC updates per gradient step during the warmup phase. Light actor training clones base policy behavior. |

### State Representation

| Parameter | Default | Script Value | Description |
|-----------|---------|--------------|-------------|
| `--use_vlm_embedding` | `0` | `1` | When 1, uses pre-computed VLM (Vision-Language Model) embeddings as state input instead of raw pixel observations. Provides a richer, pre-trained representation. |
