#! /usr/bin/env python
"""Filtered Behavior Cloning via Base Policy Distillation (JAX).

This script implements filtered behavior cloning:
1. Load a base policy (Pi-0.5) as agent_dp
2. Collect N trajectories using the base policy
3. Filter for successful trajectories only
4. Fine-tune agent_dp (JAX model) on the successful trajectories for K gradient steps
5. Repeat for multiple rounds

The base policy is a JAX PI0/PI05 model from the OpenPI codebase.
"""

import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib
import functools
import math

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import wandb
import orbax.checkpoint as ocp

from tqdm import tqdm
from openpi_client import image_tools

import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

# Monkey-patch the Policy class to fix tokenized_prompt dimension issue
# (same fix as in train_sim_residual.py)
_original_policy_infer = None

def _patched_infer(self, obs, **kwargs):
    original_input_transform = self._input_transform
    def fixed_input_transform(inputs):
        result = original_input_transform(inputs)
        if "tokenized_prompt" in result and result["tokenized_prompt"] is not None:
            arr = result["tokenized_prompt"]
            if hasattr(arr, 'ndim') and arr.ndim >= 2:
                result["tokenized_prompt"] = arr[0] if arr.shape[0] == 1 else arr
        if "tokenized_prompt_mask" in result and result["tokenized_prompt_mask"] is not None:
            arr = result["tokenized_prompt_mask"]
            if hasattr(arr, 'ndim') and arr.ndim >= 2:
                result["tokenized_prompt_mask"] = arr[0] if arr.shape[0] == 1 else arr
        return result
    self._input_transform = fixed_input_transform
    try:
        return _original_policy_infer(self, obs, **kwargs)
    finally:
        self._input_transform = original_input_transform

def patch_openpi_policy():
    global _original_policy_infer
    from openpi.policies import policy as _policy_module
    if _original_policy_infer is None:
        _original_policy_infer = _policy_module.Policy.infer
        _policy_module.Policy.infer = _patched_infer

patch_openpi_policy()


def _get_libero_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_pi_zero_input(obs, variant):
    """Convert raw observation to Pi-0/Pi-0.5 input format."""
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))
        obs_pi_zero = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )),
            "prompt": str(variant.task_description),
        }
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")
    return obs_pi_zero


def _make_pad_action(env_name, action_dim):
    """Create a single padding action for the given environment.

    For Libero: [0]*6 + [1] (6 zero dims + gripper=1).
    Other environments: raises NotImplementedError.
    """
    if env_name == 'libero':
        pad_action = np.zeros(action_dim, dtype=np.float32)
        pad_action[-1] = 1.0  # gripper closed
        return pad_action
    else:
        raise NotImplementedError(
            f"Action padding is only implemented for 'libero' environment, got: {env_name}. "
            f"Set --drop_short_actions 1 to drop samples instead of padding."
        )


def collect_trajectory_base_policy(variant, agent_dp, env):
    """Collect a single trajectory using the base policy.

    Stores per-timestep executed actions for building action_horizon-length targets.

    Returns dict with obs_pi_zero_list, executed_actions, rewards, is_success, etc.
    """
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    chunk_len = variant.chunk_len

    if 'libero' in variant.env:
        obs = env.reset()
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")

    obs_pi_zero_list = []
    executed_actions = []  # per-timestep executed actions
    rewards = []

    for t in range(max_timesteps):
        if t % query_frequency == 0:
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            infer_result = agent_dp.infer(obs_pi_zero)
            base_actions = infer_result["actions"][:chunk_len]

            obs_pi_zero_list.append(obs_pi_zero)

        action_t = np.clip(base_actions[t % query_frequency], -1.0, 1.0)
        executed_actions.append(action_t)

        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        else:
            raise NotImplementedError()

        rewards.append(reward)
        if done:
            break

    rewards = np.array(rewards)
    episode_return = np.sum(rewards)
    is_success = (reward == env_max_reward)

    return {
        'obs_pi_zero_list': obs_pi_zero_list,
        'executed_actions': np.array(executed_actions),
        'rewards': rewards,
        'is_success': is_success,
        'episode_return': episode_return,
        'env_steps': t + 1,
        'query_frequency': query_frequency,
    }


def collect_trajectories(variant, agent_dp, env, num_trajectories):
    """Collect N trajectories, return all + successful ones."""
    all_trajs = []
    success_trajs = []

    for traj_idx in range(num_trajectories):
        print(f"\n--- Collecting trajectory {traj_idx + 1}/{num_trajectories} ---")
        traj = collect_trajectory_base_policy(variant, agent_dp, env)
        all_trajs.append(traj)
        if traj['is_success']:
            success_trajs.append(traj)
        print(f"  Return: {traj['episode_return']:.2f}, Success: {traj['is_success']}, "
              f"Steps: {traj['env_steps']}")

    print(f"\nCollection done: {len(all_trajs)} total, {len(success_trajs)} successful "
          f"({100*len(success_trajs)/max(1,len(all_trajs)):.1f}%)")
    return all_trajs, success_trajs


def build_training_samples(success_trajs, action_horizon, env_name, action_dim,
                           drop_short_actions=True):
    """Build training samples from successful trajectories.

    For each query step in the trajectory, build a training sample with:
    - obs_pi_zero: the observation at that query step
    - actions: action_horizon-length action sequence starting from that timestep

    Actions are taken from the executed actions of the trajectory.

    Args:
        success_trajs: List of successful trajectory dicts.
        action_horizon: Target action sequence length (from model config).
        env_name: Environment name (for padding action format).
        action_dim: Action dimension for the environment.
        drop_short_actions: If True (default), drop samples where fewer than
            action_horizon steps remain in the trajectory. If False, pad
            remaining actions with environment-specific padding
            (Libero: [0]*6 + [1]).

    Returns:
        List of (obs_pi_zero, actions_array) tuples.
    """
    samples = []
    dropped = 0
    for traj in success_trajs:
        executed = traj['executed_actions']  # (T, action_dim)
        total_steps = len(executed)
        query_frequency = traj['query_frequency']

        for q_idx, obs_pz in enumerate(traj['obs_pi_zero_list']):
            t_start = q_idx * query_frequency
            t_end = min(t_start + action_horizon, total_steps)
            num_available = t_end - t_start

            if num_available < action_horizon:
                if drop_short_actions:
                    dropped += 1
                    continue
                # Pad with environment-specific padding action
                action_seq = executed[t_start:t_end]
                pad_action = _make_pad_action(env_name, action_dim)
                num_pad = action_horizon - num_available
                pad = np.tile(pad_action, (num_pad, 1))
                action_seq = np.concatenate([action_seq, pad], axis=0)
            else:
                action_seq = executed[t_start:t_end]

            samples.append((obs_pz, action_seq))

    if dropped > 0:
        print(f"  Dropped {dropped} samples with fewer than {action_horizon} remaining steps")

    return samples


def prepare_batch_jax(samples, agent_dp, batch_indices):
    """Prepare a batch of samples for the JAX PI0 model's compute_loss.

    Applies the policy's input transforms (LiberoInputs, Normalize, TokenizePrompt,
    PadStatesAndActions) to each sample, then stacks into JAX arrays.

    Returns:
        (observation: Observation, actions: jnp.array) ready for model.compute_loss()
    """
    from openpi.models.model import Observation

    batch_obs_dicts = []
    batch_actions = []

    for idx in batch_indices:
        obs_pz, actions = samples[idx]

        # Build input dict with actions included so transforms apply to both
        input_dict = dict(obs_pz)
        input_dict["actions"] = actions  # (action_horizon, env_action_dim)

        # Apply the policy's input transforms (LiberoInputs -> Normalize -> TokenizePrompt -> PadStatesAndActions)
        transformed = agent_dp._input_transform(input_dict)

        batch_obs_dicts.append(transformed)
        batch_actions.append(transformed["actions"])

    # Stack into batched JAX arrays
    image_keys = list(batch_obs_dicts[0]["image"].keys())
    batched_images = {}
    for key in image_keys:
        imgs = np.stack([d["image"][key] for d in batch_obs_dicts], axis=0)
        batched_images[key] = jnp.asarray(imgs)

    image_mask_keys = list(batch_obs_dicts[0]["image_mask"].keys())
    batched_image_masks = {}
    for key in image_mask_keys:
        masks = np.stack([np.asarray(d["image_mask"][key]) for d in batch_obs_dicts], axis=0)
        batched_image_masks[key] = jnp.asarray(masks, dtype=jnp.bool_)

    states = np.stack([np.asarray(d["state"]) for d in batch_obs_dicts], axis=0)
    batched_state = jnp.asarray(states, dtype=jnp.float32)

    prompts = np.stack([np.asarray(d["tokenized_prompt"]) for d in batch_obs_dicts], axis=0)
    batched_prompt = jnp.asarray(prompts, dtype=jnp.int32)

    prompt_masks = np.stack([np.asarray(d["tokenized_prompt_mask"]) for d in batch_obs_dicts], axis=0)
    batched_prompt_mask = jnp.asarray(prompt_masks, dtype=jnp.bool_)

    obs_dict = {
        "image": batched_images,
        "image_mask": batched_image_masks,
        "state": batched_state,
        "tokenized_prompt": batched_prompt,
        "tokenized_prompt_mask": batched_prompt_mask,
    }
    observation = Observation.from_dict(obs_dict)

    actions_np = np.stack(batch_actions, axis=0).astype(np.float32)
    actions_jax = jnp.asarray(actions_np)

    return observation, actions_jax


def create_train_step_fn(model, tx, trainable_filter):
    """Create a JIT-compiled training step function.

    Args:
        model: The PI0 NNX model.
        tx: optax optimizer.
        trainable_filter: NNX filter for trainable parameters.

    Returns:
        A function: (params, opt_state, rng, observation, actions) -> (new_params, new_opt_state, info)
    """
    graphdef, _ = nnx.split(model)

    def train_step(params, opt_state, rng, observation, actions):
        # Reconstruct model from graphdef + params
        model_local = nnx.merge(graphdef, params)
        model_local.train()

        def loss_fn(model_inner, rng, obs, acts):
            chunked_loss = model_inner.compute_loss(rng, obs, acts, train=True)
            return jnp.mean(chunked_loss)

        diff_state = nnx.DiffState(0, trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
            model_local, rng, observation, actions
        )

        # Filter to trainable params for optimizer
        trainable_params = params.filter(trainable_filter)
        updates, new_opt_state = tx.update(grads, opt_state, trainable_params)
        new_trainable = optax.apply_updates(trainable_params, updates)

        # Update model with new trainable params
        nnx.update(model_local, new_trainable)
        new_params = nnx.state(model_local)

        grad_norm = optax.global_norm(grads)
        info = {
            "loss": loss,
            "grad_norm": grad_norm,
        }

        return new_params, new_opt_state, info

    return jax.jit(train_step)


def rebuild_policy_inference(agent_dp, model):
    """Rebuild the JIT-compiled inference function after model params change.

    After training updates the model parameters, we need to re-freeze the
    module_jit'd sample_actions so inference uses the new weights.
    """
    from openpi.shared import nnx_utils

    agent_dp._model = model
    agent_dp._sample_actions = nnx_utils.module_jit(model.sample_actions)
    if hasattr(model, "get_prefix_rep"):
        agent_dp._get_prefix_rep = nnx_utils.module_jit(model.get_prefix_rep)


def perform_eval_base_policy(agent_dp, env, variant, step, wandb_logger):
    """Evaluate base policy and log metrics (no diagnostics)."""
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    chunk_len = variant.chunk_len

    episode_returns = []
    success_rates = []
    episode_lens = []

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            obs = env.reset()
        else:
            raise NotImplementedError()

        image_list = []
        rewards = []

        for t in range(max_timesteps):
            if variant.env == 'libero':
                curr_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

            if t % query_frequency == 0:
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                infer_result = agent_dp.infer(obs_pi_zero)
                base_actions = infer_result["actions"][:chunk_len]

            action_t = np.clip(base_actions[t % query_frequency], -1.0, 1.0)

            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)

            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        episode_lens.append(t + 1)
        rewards_arr = np.array(rewards)
        episode_return = np.sum(rewards_arr)
        episode_returns.append(episode_return)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)

        print(f'Eval rollout {rollout_id}: return={episode_return:.2f}, success={is_success}')
        if len(image_list) > 0:
            video = np.stack(image_list)
            if video.ndim == 4:
                video = video.transpose(0, 3, 1, 2)
            wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=step)

    success_rate = np.mean(np.array(success_rates, dtype=float))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)

    wandb_logger.log({'evaluation/success_rate': success_rate}, step=step)
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=step)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=step)

    for r in range(env_max_reward + 1):
        more_or_equal_r = sum(1 for ret in episode_returns if ret >= r)
        rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': rate}, step=step)

    print(f'\n[Eval @ step {step}] Success rate: {success_rate:.3f}, '
          f'Avg return: {avg_return:.2f}, Avg episode len: {avg_episode_len:.1f}\n')

    return success_rate


def save_jax_checkpoint(params, ckpt_dir):
    """Save JAX model params to a checkpoint directory using orbax."""
    os.makedirs(ckpt_dir, exist_ok=True)
    params_dir = os.path.join(ckpt_dir, "params")
    pure_dict = params.to_pure_dict()
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(params_dir, {"params": pure_dict})
    print(f"  Saved JAX checkpoint to {ckpt_dir}")


def main_base_policy_distillation(variant):
    """Main function for filtered behavior cloning / base policy distillation."""

    # Prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('Writing to output dir:', outputdir)

    # Environment setup
    if variant.env == 'libero':
        from libero.libero import benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_10"]()
        if variant.libero_task:
            task_names = task_suite.get_task_names()
            matching = [i for i, name in enumerate(task_names) if name == variant.libero_task]
            assert len(matching) == 1, f"Task '{variant.libero_task}' not found. Available: {task_names}"
            task_id = matching[0]
        else:
            task_id = 8
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 224, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 500
        variant.action_dim = 7
        print("Libero environment initialized with task:", task_description)
    else:
        raise NotImplementedError(f"Only 'libero' environment is supported, got: {variant.env}")

    # WandB setup
    group_name = variant.prefix + '_' + variant.launch_group_id
    import tempfile
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != '', variant, variant.wandb_project,
        experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name
    )

    # Load base policy (Pi-0.5) — JAX path
    from openpi.training import config as openpi_config
    from openpi.training import optimizer as openpi_optimizer
    from openpi.policies import policy_config
    from openpi.shared import download

    config = openpi_config.get_config(variant.pi_05_config)
    checkpoint_dir = download.maybe_download(variant.pi_05_ckpt_dir)
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print(f"Loaded Pi-0.5 policy from {checkpoint_dir}")

    # Verify this is a JAX model
    if agent_dp._is_pytorch_model:
        raise RuntimeError(
            "This script requires a JAX model. The loaded checkpoint appears to be PyTorch. "
            "Ensure the checkpoint directory contains JAX params (not model.safetensors)."
        )

    print(f"JAX devices: {jax.devices()}")

    # =========================================================================
    # Phase 0: Evaluate the base policy before any training
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 0: Evaluating base policy before training")
    print("=" * 60)
    perform_eval_base_policy(agent_dp, eval_env, variant, step=0, wandb_logger=wandb_logger)

    # =========================================================================
    # Set up optimizer from pi0.5 config
    # =========================================================================
    model = agent_dp._model
    model.eval()

    # Use trainable filter from config (respects LoRA / freeze settings)
    trainable_filter = config.trainable_filter if hasattr(config, 'trainable_filter') else nnx.Param

    # Create optimizer using the same settings as the pi0.5 training config
    tx = openpi_optimizer.create_optimizer(config.optimizer, config.lr_schedule)
    print(f"Optimizer from config: {config.optimizer}")
    print(f"LR schedule from config: {config.lr_schedule}")

    # Initialize optimizer state from current model params
    params = nnx.state(model)
    trainable_params = params.filter(trainable_filter)
    opt_state = tx.init(trainable_params)
    print(f"Optimizer initialized with trainable parameters.")

    # Get action_horizon from model for building training samples
    action_horizon = model.action_horizon
    print(f"Model action_horizon: {action_horizon}")

    # Whether to drop samples with fewer than action_horizon remaining steps
    drop_short_actions = variant.get('drop_short_actions', True)

    # Create JIT-compiled train step
    jit_train_step = create_train_step_fn(model, tx, trainable_filter)
    print("JIT train step function created.")

    # =========================================================================
    # Main loop: alternate between collecting trajectories and training
    # =========================================================================
    global_train_step = 0
    rng = jax.random.PRNGKey(variant.seed)

    for round_num in range(variant.num_rounds):
        print("\n" + "=" * 60)
        print(f"ROUND {round_num + 1}/{variant.num_rounds}")
        print("=" * 60)

        # =====================================================================
        # Phase 1: Collect N trajectories
        # =====================================================================
        print(f"\nPHASE 1: Collecting {variant.num_collect_trajectories} trajectories...")
        all_trajs, success_trajs = collect_trajectories(
            variant, agent_dp, env, variant.num_collect_trajectories
        )

        total_trajs = len(all_trajs)
        num_success = len(success_trajs)
        wandb_logger.log({
            'collection/total_trajectories': total_trajs,
            'collection/successful_trajectories': num_success,
            'collection/success_rate': num_success / total_trajs if total_trajs > 0 else 0,
            'collection/round': round_num,
        }, step=global_train_step)

        if num_success == 0:
            print("WARNING: No successful trajectories collected. Skipping training for this round.")
            continue

        # =====================================================================
        # Phase 2: Train agent_dp on successful trajectories
        # =====================================================================
        print(f"\nPHASE 2: Training on {num_success} successful trajectories "
              f"for {variant.num_train_steps_per_round} steps...")

        samples = build_training_samples(
            success_trajs, action_horizon,
            env_name=variant.env, action_dim=variant.action_dim,
            drop_short_actions=drop_short_actions,
        )
        num_samples = len(samples)
        print(f"  Total training samples (query steps): {num_samples}")

        if num_samples == 0:
            print("WARNING: No valid training samples after filtering. Skipping training.")
            continue

        if num_samples < variant.batch_size:
            print(f"  WARNING: Only {num_samples} samples, batch_size={variant.batch_size}. "
                  f"Sampling with replacement.")

        # Set model to train mode and recreate JIT step (graphdef may differ between train/eval)
        model.train()
        jit_train_step = create_train_step_fn(model, tx, trainable_filter)

        train_losses = []
        for train_step_idx in tqdm(range(variant.num_train_steps_per_round), desc='Training'):
            # Sample batch indices
            if num_samples >= variant.batch_size:
                batch_indices = np.random.choice(num_samples, size=variant.batch_size, replace=False)
            else:
                batch_indices = np.random.choice(num_samples, size=variant.batch_size, replace=True)

            # Prepare batch
            observation, actions = prepare_batch_jax(samples, agent_dp, batch_indices)

            # Training step
            rng, step_rng = jax.random.split(rng)
            params, opt_state, info = jit_train_step(params, opt_state, step_rng, observation, actions)

            loss_val = float(info["loss"])
            grad_norm_val = float(info["grad_norm"])
            train_losses.append(loss_val)
            global_train_step += 1

            # Logging
            if train_step_idx % variant.log_interval == 0:
                avg_loss = np.mean(train_losses[-min(variant.log_interval, len(train_losses)):])
                wandb_logger.log({
                    'training/loss': loss_val,
                    'training/avg_loss': avg_loss,
                    'training/grad_norm': grad_norm_val,
                    'training/round': round_num,
                }, step=global_train_step)
                print(f"  Step {train_step_idx}/{variant.num_train_steps_per_round}: "
                      f"loss={loss_val:.6f}, avg_loss={avg_loss:.6f}, grad_norm={grad_norm_val:.4f}")

        # Update the model with trained params
        nnx.update(model, params)
        model.eval()

        # Rebuild JIT'd inference so agent_dp.infer() uses new weights
        rebuild_policy_inference(agent_dp, model)

        avg_round_loss = np.mean(train_losses) if train_losses else 0
        print(f"\nRound {round_num + 1} training complete. Avg loss: {avg_round_loss:.6f}")
        wandb_logger.log({'training/round_avg_loss': avg_round_loss}, step=global_train_step)

        # =====================================================================
        # Phase 3: Evaluate after training
        # =====================================================================
        print(f"\nPHASE 3: Evaluating after round {round_num + 1}...")
        success_rate = perform_eval_base_policy(
            agent_dp, eval_env, variant, step=global_train_step, wandb_logger=wandb_logger
        )

        # Save checkpoint
        if variant.checkpoint_interval > 0 and (round_num + 1) % variant.checkpoint_interval == 0:
            ckpt_dir = os.path.join(outputdir, f"checkpoint_round_{round_num + 1}")
            save_jax_checkpoint(params, ckpt_dir)

    # Save final checkpoint
    final_ckpt_dir = os.path.join(outputdir, "checkpoint_final")
    save_jax_checkpoint(params, final_ckpt_dir)

    print("\n" + "=" * 60)
    print("FILTERED BEHAVIOR CLONING COMPLETE")
    print("=" * 60)
