#!/usr/bin/env python
"""Standalone evaluation script for Residual SAC checkpoints.

Loads a trained Residual SAC checkpoint, runs num_evals rollouts, generates
diagnostic plots for a subset of successful and failed trajectories, and writes
a summary CSV to output_dir.

Usage:
    python -m examples.evaluation.evaluate_residual_sac \\
        --checkpoint_dir /path/to/checkpoint_dir \\
        --output_dir /path/to/output \\
        --num_evals 50 \\
        --diagnostic_freq 5 \\
        --env libero \\
        --pi_05_config pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k \\
        --pi_05_ckpt_dir /path/to/pi05_ckpt \\
        --residual_alpha 0.5 \\
        --query_freq 10 \\
        --chunk_len 10 \\
        --hidden_dims 512 \\
        --use_vlm_embedding 1
"""

import os

# Tell XLA to use Triton GEMM — improves steps/sec by ~30% on some GPUs
_xla_flags = os.environ.get('XLA_FLAGS', '')
_xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = _xla_flags

import sys
import argparse
import csv
import glob as glob_module
import pathlib
import re
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from tqdm import tqdm

import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from jaxrl2.agents.pixel_sac.pixel_sac_residual_learner import PixelSACResidualLearner
from jaxrl2.agents.pixel_sac.pixel_ppo_residual_learner import PixelPPOResidualLearner
from jaxrl2.agents.pixel_sac.pixel_parl_residual_learner import PixelPARLResidualLearner
from jaxrl2.agents.pixel_sac.pixel_gradq_residual_learner import PixelGradQResidualLearner
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim, AttrDict
from jaxrl2.utils.launch_util import parse_training_args

from examples.train_sim_residual import (
    DummyEnvResidual,
    _get_libero_env,
    patch_openpi_policy,
)
from examples.train_utils_sim_residual import (
    obs_to_img,
    obs_to_pi_zero_input,
    obs_to_qpos,
)
from examples.diagnostics.data_collector import EvalTrajectoryData
from examples.diagnostics.run_diagnostics import generate_all_diagnostics

_home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(_home_dir, 'jax_compilation_cache'))

# Apply the openpi Policy monkey-patch (fixes tokenized_prompt batch-dim issue)
patch_openpi_policy()


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_eval_trajectory(variant, agent, env, agent_dp, rng):
    """Collect a single evaluation trajectory with full diagnostic data.

    Mirrors perform_control_eval_residual but for a single rollout and without
    any WandB logging.

    Args:
        variant: Training/eval configuration AttrDict.
        agent: Residual SAC agent (already loaded from checkpoint).
        env: Environment instance.
        agent_dp: Frozen base policy (Pi-0.5).
        rng: JAX PRNGKey.

    Returns:
        traj_data (EvalTrajectoryData): Structured trajectory data for diagnostics.
        stats (dict): Per-trajectory scalar statistics.
        rng: Updated PRNGKey.
    """
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    residual_alpha = float(agent._residual_alpha)
    chunk_len = variant.chunk_len
    predict_a_exec = variant.get('predict_a_exec', False)
    use_vlm_embedding = variant.get('use_vlm_embedding', False)

    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    elif variant.env == 'cartpole':
        obs = env.reset()
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")

    image_list = []
    rewards = []
    all_delta_norms = []
    all_base_norms = []
    all_clipping_rates = []

    # Diagnostics storage (per query-step)
    diag_obs_dicts = []
    diag_base_actions = []
    diag_delta_actions = []
    diag_a_exec = []
    episode_terminated = False
    eval_info = {}

    for t in tqdm(range(max_timesteps), desc='eval steps', leave=False):
        curr_image = obs_to_img(obs, variant)

        if t % query_frequency == 0:
            qpos = obs_to_qpos(obs, variant)

            # 1. Query frozen Pi-0.5
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            infer_result = agent_dp.infer(obs_pi_zero, return_vlm_embedding=use_vlm_embedding)
            base_actions = infer_result["actions"][:chunk_len]  # (chunk_len, action_dim)

            # 2. Build SAC observation dict
            if use_vlm_embedding:
                vlm_hidden_state = infer_result["vlm_embedding"][0]
                if vlm_hidden_state.ndim == 3 and vlm_hidden_state.shape[0] == 1:
                    vlm_hidden_state = vlm_hidden_state[0]
                vlm_hidden_state = np.mean(vlm_hidden_state, axis=0)  # (W,)
                obs_dict = {
                    'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    'vlm_embedding': vlm_hidden_state[np.newaxis, ..., np.newaxis],
                    'base_action': base_actions[np.newaxis, ..., np.newaxis],
                }
                if variant.add_states:
                    obs_dict['state'] = qpos[np.newaxis, ..., np.newaxis]
            elif variant.add_states:
                obs_dict = {
                    'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    'state': qpos[np.newaxis, ..., np.newaxis],
                    'base_action': base_actions[np.newaxis, ..., np.newaxis],
                }
            else:
                obs_dict = {
                    'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    'base_action': base_actions[np.newaxis, ..., np.newaxis],
                }

            rng, _ = jax.random.split(rng)

            # 3. Deterministic agent action (eval mode: use mean)
            actions_flat = agent.eval_actions(obs_dict)
            raw_actions = np.reshape(actions_flat, (query_frequency, variant.action_dim))

            # NaN guard
            if not np.all(np.isfinite(raw_actions)):
                print(f'[WARNING] NaN/Inf in eval actions at t={t}, replacing with zeros')
                raw_actions = np.nan_to_num(raw_actions, nan=0.0, posinf=0.0, neginf=0.0)

            if predict_a_exec:
                actions = np.clip(raw_actions, -1.0, 1.0)
                delta_actions = actions - base_actions[:query_frequency]
            else:
                delta_actions = raw_actions
                actions = np.clip(
                    base_actions[:query_frequency] + residual_alpha * delta_actions, -1.0, 1.0
                )

            # Track statistics
            delta_norm = np.linalg.norm(delta_actions.flatten())
            base_norm = np.linalg.norm(base_actions.flatten())
            clipping_rate = float(np.mean(np.abs(actions) > 0.999))
            all_delta_norms.append(delta_norm)
            all_base_norms.append(base_norm)
            all_clipping_rates.append(clipping_rate)

            # Diagnostics storage
            diag_obs_dicts.append({k: np.copy(v) for k, v in obs_dict.items()})
            diag_base_actions.append(np.copy(base_actions))
            diag_delta_actions.append(np.copy(delta_actions))
            diag_a_exec.append(np.copy(actions))

        action_t = actions[t % query_frequency]

        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
        elif variant.env == 'cartpole':
            obs, reward, done, eval_info = env.step(action_t)

        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            episode_terminated = True
            break

    # ------------------------------------------------------------------
    # Final observation (for Q-value bootstrapping in diagnostics)
    # ------------------------------------------------------------------
    final_image = obs_to_img(obs, variant)
    final_qpos = obs_to_qpos(obs, variant)
    last_base = (
        diag_base_actions[-1]
        if diag_base_actions
        else np.zeros((chunk_len, variant.action_dim))
    )

    if use_vlm_embedding:
        obs_pi_zero_final = obs_to_pi_zero_input(obs, variant)
        final_infer = agent_dp.infer(obs_pi_zero_final, return_vlm_embedding=True)
        vlm_hs = final_infer["vlm_embedding"][0]
        if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
            vlm_hs = vlm_hs[0]
        vlm_hs = np.mean(vlm_hs, axis=0)
        final_obs_dict = {
            'pixels': final_image[np.newaxis, ..., np.newaxis],
            'vlm_embedding': vlm_hs[np.newaxis, ..., np.newaxis],
            'base_action': last_base[np.newaxis, ..., np.newaxis],
        }
        if variant.add_states:
            final_obs_dict['state'] = final_qpos[np.newaxis, ..., np.newaxis]
    elif variant.add_states:
        final_obs_dict = {
            'pixels': final_image[np.newaxis, ..., np.newaxis],
            'state': final_qpos[np.newaxis, ..., np.newaxis],
            'base_action': last_base[np.newaxis, ..., np.newaxis],
        }
    else:
        final_obs_dict = {
            'pixels': final_image[np.newaxis, ..., np.newaxis],
            'base_action': last_base[np.newaxis, ..., np.newaxis],
        }
    diag_obs_dicts.append({k: np.copy(v) for k, v in final_obs_dict.items()})

    # ------------------------------------------------------------------
    # Compute per-episode stats
    # ------------------------------------------------------------------
    rewards_arr = np.array(rewards)
    episode_return = float(np.sum(rewards_arr))
    episode_len = t + 1
    truncated_episode = not episode_terminated or (
        t + 1 >= max_timesteps and not episode_terminated
    )

    if variant.env == 'cartpole':
        is_success = bool(eval_info.get('success', False))
    else:
        is_success = bool(reward == env_max_reward)

    traj_data = EvalTrajectoryData(
        obs_dicts=diag_obs_dicts,
        base_actions=np.array(diag_base_actions),
        delta_actions=np.array(diag_delta_actions),
        a_exec=np.array(diag_a_exec),
        rewards=rewards_arr,
        terminated=episode_terminated and not truncated_episode,
        truncated=truncated_episode,
        is_success=is_success,
        images=image_list,
        episode_return=episode_return,
        query_frequency=query_frequency,
    )

    stats = {
        'is_success': int(is_success),
        'episode_return': episode_return,
        'episode_len': episode_len,
        'delta_norm_mean': float(np.mean(all_delta_norms)) if all_delta_norms else 0.0,
        'base_norm_mean': float(np.mean(all_base_norms)) if all_base_norms else 0.0,
        'clipping_rate_mean': float(np.mean(all_clipping_rates)) if all_clipping_rates else 0.0,
    }

    return traj_data, stats, rng


# ---------------------------------------------------------------------------
# Agent creation (mirrors train_sim_residual.main_residual)
# ---------------------------------------------------------------------------

def _create_agent(variant, sample_obs, sample_action, kwargs):
    """Instantiate the correct Residual RL agent based on variant.algo."""
    algo = variant.get('algo', 'residual_sac')
    kwargs = dict(kwargs)  # shallow copy to avoid mutating caller's dict

    if algo == 'sac':
        sac_accepted = {
            'actor_lr', 'critic_lr', 'temp_lr', 'decay_steps',
            'hidden_dims', 'cnn_features', 'cnn_strides', 'cnn_padding',
            'latent_dim', 'discount', 'tau', 'critic_reduction', 'dropout_rate',
            'encoder_type', 'encoder_norm', 'color_jitter',
            'use_spatial_softmax', 'softmax_temperature', 'aug_next',
            'use_bottleneck', 'init_temperature', 'num_qs', 'target_entropy',
            'action_magnitude', 'num_cameras', 'learn_std', 'fixed_log_std',
        }
        sac_kwargs = {k: v for k, v in kwargs.items() if k in sac_accepted}
        agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **sac_kwargs)
        agent._residual_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        agent.algo = 'sac'
        agent.predict_a_exec = True
        agent.query_frequency = variant.query_freq
        agent._num_critic_updates = variant.get('num_critic_updates', 1)
        agent._num_actor_updates = variant.get('num_actor_updates', 1)
        agent.update_critic = agent.update
        agent.update_actor = lambda batch: {}
        agent.update_actor_bc = lambda batch: {}

    elif algo == 'residual_sac':
        kwargs['use_huber_loss'] = variant.get('use_huber_loss', False)
        kwargs['huber_delta'] = variant.get('huber_delta', 1.0)
        kwargs['max_grad_norm'] = variant.get('max_grad_norm', 1.0)
        kwargs['num_critic_updates'] = variant.get('num_critic_updates', 1)
        kwargs['num_actor_updates'] = variant.get('num_actor_updates', 1)
        kwargs['bc_reg_coeff'] = variant.get('bc_reg_coeff', 0.0)
        kwargs['bc_on_success_only'] = variant.get('bc_on_success_only', False)
        kwargs['predict_a_exec'] = variant.get('predict_a_exec', False)
        kwargs['log_std_min'] = variant.get('log_std_min', -5.0)
        kwargs['log_std_max'] = variant.get('log_std_max', 2.0)
        kwargs['learn_std'] = variant.get('learn_std', True)
        kwargs['use_vlm_embedding'] = variant.get('use_vlm_embedding', False)
        agent = PixelSACResidualLearner(variant.seed, sample_obs, sample_action, **kwargs)

    elif algo in ('q_weighted_pg', 'residual_grpo'):
        ppo_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ('temp_lr', 'init_temperature', 'backup_entropy',
                         'clip_temp', 'clip_min_temp', 'clip_max_temp', 'target_entropy')
        }
        ppo_kwargs['algo'] = algo
        ppo_kwargs['grpo_num_samples'] = variant.get('grpo_num_samples', 8)
        ppo_kwargs['clip_epsilon'] = variant.get('clip_epsilon', 0.2)
        ppo_kwargs['clip_min_epsilon_multiplier'] = variant.get('clip_min_epsilon_multiplier', 1.0)
        ppo_kwargs['clip_max_epsilon_multiplier'] = variant.get('clip_max_epsilon_multiplier', 1.0)
        ppo_kwargs['entropy_coeff'] = variant.get('entropy_coeff', 1e-3)
        ppo_kwargs['advantage_critic_reduction'] = variant.get('advantage_critic_reduction', 'mean')
        ppo_kwargs['adv_clip_min'] = variant.get('adv_clip_min', None)
        ppo_kwargs['adv_clip_max'] = variant.get('adv_clip_max', None)
        ppo_kwargs['log_ratio_clip'] = variant.get('log_ratio_clip', 20.0)
        ppo_kwargs['log_prob_clip'] = variant.get('log_prob_clip', 50.0)
        ppo_kwargs['max_grad_norm'] = variant.get('max_grad_norm', 1.0)
        ppo_kwargs['use_huber_loss'] = variant.get('use_huber_loss', False)
        ppo_kwargs['huber_delta'] = variant.get('huber_delta', 1.0)
        ppo_kwargs['num_critic_updates'] = variant.get('num_critic_updates', 2)
        ppo_kwargs['num_actor_updates'] = variant.get('num_actor_updates', 4)
        ppo_kwargs['bc_reg_coeff'] = variant.get('bc_reg_coeff', 0.0)
        ppo_kwargs['bc_on_success_only'] = variant.get('bc_on_success_only', False)
        ppo_kwargs['on_policy_ppo'] = variant.get('on_policy_ppo', False)
        ppo_kwargs['normalize_advantages'] = variant.get('normalize_advantages', False)
        ppo_kwargs['log_std_min'] = variant.get('log_std_min', -5.0)
        ppo_kwargs['log_std_max'] = variant.get('log_std_max', 2.0)
        ppo_kwargs['predict_a_exec'] = variant.get('predict_a_exec', False)
        ppo_kwargs['learn_std'] = variant.get('learn_std', True)
        ppo_kwargs['use_vlm_embedding'] = variant.get('use_vlm_embedding', False)
        agent = PixelPPOResidualLearner(variant.seed, sample_obs, sample_action, **ppo_kwargs)

    elif algo == 'residual_parl':
        parl_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ('temp_lr', 'init_temperature', 'backup_entropy',
                         'clip_temp', 'clip_min_temp', 'clip_max_temp', 'target_entropy')
        }
        parl_kwargs['parl_num_samples'] = variant.get('parl_num_samples', 16)
        parl_kwargs['parl_num_elites'] = variant.get('parl_num_elites', 4)
        parl_kwargs['parl_num_grad_steps'] = variant.get('parl_num_grad_steps', 5)
        parl_kwargs['parl_step_size'] = variant.get('parl_step_size', 0.01)
        parl_kwargs['max_grad_norm'] = variant.get('max_grad_norm', 1.0)
        parl_kwargs['use_huber_loss'] = variant.get('use_huber_loss', False)
        parl_kwargs['huber_delta'] = variant.get('huber_delta', 1.0)
        parl_kwargs['num_critic_updates'] = variant.get('num_critic_updates', 2)
        parl_kwargs['num_actor_updates'] = variant.get('num_actor_updates', 4)
        parl_kwargs['predict_a_exec'] = variant.get('predict_a_exec', False)
        parl_kwargs['log_std_min'] = variant.get('log_std_min', -5.0)
        parl_kwargs['log_std_max'] = variant.get('log_std_max', 2.0)
        parl_kwargs['learn_std'] = variant.get('learn_std', True)
        parl_kwargs['use_vlm_embedding'] = variant.get('use_vlm_embedding', False)
        agent = PixelPARLResidualLearner(variant.seed, sample_obs, sample_action, **parl_kwargs)

    elif algo == 'residual_gradq':
        gradq_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ('temp_lr', 'init_temperature', 'backup_entropy',
                         'clip_temp', 'clip_min_temp', 'clip_max_temp', 'target_entropy')
        }
        gradq_kwargs['gradq_num_grad_steps'] = variant.get('parl_num_grad_steps', 5)
        gradq_kwargs['gradq_step_size'] = variant.get('parl_step_size', 0.01)
        gradq_kwargs['max_grad_norm'] = variant.get('max_grad_norm', 1.0)
        gradq_kwargs['use_huber_loss'] = variant.get('use_huber_loss', False)
        gradq_kwargs['huber_delta'] = variant.get('huber_delta', 1.0)
        gradq_kwargs['num_critic_updates'] = variant.get('num_critic_updates', 2)
        gradq_kwargs['num_actor_updates'] = variant.get('num_actor_updates', 4)
        gradq_kwargs['predict_a_exec'] = variant.get('predict_a_exec', False)
        gradq_kwargs['log_std_min'] = variant.get('log_std_min', -5.0)
        gradq_kwargs['log_std_max'] = variant.get('log_std_max', 2.0)
        gradq_kwargs['learn_std'] = variant.get('learn_std', True)
        gradq_kwargs['use_vlm_embedding'] = variant.get('use_vlm_embedding', False)
        agent = PixelGradQResidualLearner(variant.seed, sample_obs, sample_action, **gradq_kwargs)

    else:
        raise ValueError(f'Unknown algorithm: {algo}')

    return agent


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(variant):
    """Load checkpoint and run evaluation rollouts."""

    # Prevent TensorFlow from grabbing GPUs
    tf.config.set_visible_devices([], 'GPU')

    output_dir = variant.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # Point variant.outputdir here so diagnostics land in output_dir/diagnostics/
    variant.outputdir = output_dir

    checkpoint_dir = variant.checkpoint_dir
    num_evals = variant.num_evals
    diagnostic_freq = variant.diagnostic_freq

    print('=' * 60)
    print('RESIDUAL SAC EVALUATION')
    print('=' * 60)
    print(f'  checkpoint_dir : {checkpoint_dir}')
    print(f'  output_dir     : {output_dir}')
    print(f'  num_evals      : {num_evals}')
    print(f'  diagnostic_freq: {diagnostic_freq}')
    print(f'  env            : {variant.env}')
    print(f'  algo           : {variant.get("algo", "residual_sac")}')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------
    if variant.env == 'libero':
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict['libero_10']()
        if variant.libero_task:
            task_names = task_suite.get_task_names()
            matching = [i for i, name in enumerate(task_names) if name == variant.libero_task]
            if len(matching) != 1:
                raise ValueError(
                    f"Task '{variant.libero_task}' not found in libero_10. "
                    f"Available: {task_names}"
                )
            task_id = matching[0]
        else:
            task_id = 8  # KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 224, variant.seed)
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 500
        print(f'LIBERO task: {task_description}')

    elif variant.env == 'aloha_cube':
        import gymnasium
        from gymnasium.envs.registration import register

        register(
            id='gym_aloha/AlohaTransferCube-v0',
            entry_point='gym_aloha.env:AlohaEnv',
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={'obs_type': 'pixels', 'task': 'transfer_cube'},
        )
        env = gymnasium.make(
            'gym_aloha/AlohaTransferCube-v0',
            obs_type='pixels_agent_pos',
            render_mode='rgb_array',
        )
        variant.task_description = 'Transfer cube'
        variant.env_max_reward = 4
        variant.max_timesteps = 500

    elif variant.env == 'cartpole':
        from envs.cartpole_env import CartPoleEnv

        render_size = variant.resize_image if variant.resize_image > 0 else 100
        env = CartPoleEnv(
            render_size=render_size,
            horizon=variant.get('cartpole_horizon', 100),
        )
        env.seed(variant.seed)
        variant.task_description = 'Balance the pole upright'
        variant.env_max_reward = 0
        variant.max_timesteps = variant.get('cartpole_horizon', 100)

    else:
        raise NotImplementedError(f'Unknown env: {variant.env}')

    # ------------------------------------------------------------------
    # Frozen base policy (Pi-0.5)
    # ------------------------------------------------------------------
    if variant.env == 'cartpole':
        from envs.zero_base_policy import ZeroBasePolicy

        # DummyEnvResidual sets variant.action_dim as a side-effect
        dummy_env = DummyEnvResidual(variant)
        agent_dp = ZeroBasePolicy(
            action_dim=variant.action_dim,
            chunk_len=variant.chunk_len,
            vlm_embedding_dim=variant.get('vlm_embedding_dim', 2048),
            vlm_seq_len=variant.get('vlm_seq_len', 16),
        )
        print(f'Using ZeroBasePolicy for CartPole (action_dim={variant.action_dim})')
    else:
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config
        from openpi.shared import download

        if variant.env == 'libero':
            config = openpi_config.get_config(variant.pi_05_config)
            pi_ckpt_dir = download.maybe_download(variant.pi_05_ckpt_dir)
        elif variant.env == 'aloha_cube':
            config = openpi_config.get_config('pi0_aloha_sim')
            pi_ckpt_dir = download.maybe_download(
                's3://openpi-assets/checkpoints/pi0_aloha_sim'
            )
        else:
            raise NotImplementedError()

        agent_dp = policy_config.create_trained_policy(config, pi_ckpt_dir)
        print(f'Loaded frozen Pi-0.5 policy from {pi_ckpt_dir}')

        # DummyEnvResidual sets variant.action_dim as a side-effect
        dummy_env = DummyEnvResidual(variant)

    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print(f'Obs shapes : {[(k, v.shape) for k, v in sample_obs.items()]}')
    print(f'Action shape: {sample_action.shape}')

    # ------------------------------------------------------------------
    # Agent instantiation
    # ------------------------------------------------------------------
    kwargs = dict(variant['train_kwargs'])
    kwargs.pop('cosine_decay', None)  # cosine LR decay is training-only
    kwargs['residual_alpha'] = variant.residual_alpha

    agent = _create_agent(variant, sample_obs, sample_action, kwargs)
    print(f'Agent created: {variant.get("algo", "residual_sac")}')

    # ------------------------------------------------------------------
    # Checkpoint loading — hard failure if unsuccessful
    # ------------------------------------------------------------------
    # Accepts two layouts:
    #   (a) Run directory: contains checkpoint_* entries inside it.
    #       --checkpoint_dir /run_dir/          → restores latest checkpoint
    #   (b) Specific checkpoint directory passed directly.
    #       --checkpoint_dir /run_dir/checkpoint75000/  → restores that step
    ckpt_path = pathlib.Path(checkpoint_dir)
    if not ckpt_path.exists():
        print(f'[ERROR] Checkpoint directory does not exist: {checkpoint_dir}')
        sys.exit(1)

    # Decide which path to pass to restore_checkpoint and what step to report.
    ckpt_entries = sorted(glob_module.glob(str(ckpt_path / 'checkpoint_*'))) + \
                   sorted(glob_module.glob(str(ckpt_path / 'checkpoint[0-9]*')))

    if ckpt_entries:
        # Layout (a): run directory containing checkpoint_* files/dirs
        restore_path = checkpoint_dir
        last_name = pathlib.Path(ckpt_entries[-1]).name
        m = re.search(r'checkpoint[_]?(\d+)', last_name)
        ckpt_step = int(m.group(1)) if m else -1
        print(f'Run directory with checkpoints: {[pathlib.Path(e).name for e in ckpt_entries]}')
    else:
        # Layout (b): the path itself is a specific checkpoint directory
        restore_path = checkpoint_dir
        m = re.search(r'checkpoint[_]?(\d+)', ckpt_path.name)
        ckpt_step = int(m.group(1)) if m else -1
        print(f'Using checkpoint directory directly: {ckpt_path.name}')

    print(f'Restoring from: {restore_path}  (step {ckpt_step})')

    try:
        agent.restore_checkpoint(restore_path)
    except Exception as exc:
        print(f'[ERROR] Failed to load checkpoint from {restore_path}:')
        traceback.print_exc()
        sys.exit(1)
    print(f'Checkpoint loaded successfully (step {ckpt_step}).')

    # ------------------------------------------------------------------
    # Evaluation rollouts
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(variant.seed + 789)

    all_traj_data = []
    all_stats = []

    print(f'\nRunning {num_evals} evaluation trajectories...\n')
    for rollout_id in range(num_evals):
        print(f'--- Rollout {rollout_id + 1}/{num_evals} ---')
        traj_data, stats, rng = collect_eval_trajectory(
            variant, agent, env, agent_dp, rng
        )
        all_traj_data.append(traj_data)
        all_stats.append(stats)
        print(
            f'  return={stats["episode_return"]:.3f}  '
            f'len={stats["episode_len"]}  '
            f'success={bool(stats["is_success"])}'
        )

    # ------------------------------------------------------------------
    # Separate trajectories into success / fail buckets
    # ------------------------------------------------------------------
    success_trajs = [(i, d, s) for i, (d, s) in enumerate(zip(all_traj_data, all_stats))
                     if s['is_success']]
    fail_trajs = [(i, d, s) for i, (d, s) in enumerate(zip(all_traj_data, all_stats))
                  if not s['is_success']]

    n_success_diag = min(diagnostic_freq, len(success_trajs))
    n_fail_diag = min(diagnostic_freq, len(fail_trajs))

    print(f'\nSuccess: {len(success_trajs)}/{num_evals}  '
          f'Fail: {len(fail_trajs)}/{num_evals}')
    print(f'Diagnostic trajectories: {n_success_diag} success + {n_fail_diag} fail')

    diag_traj_data = (
        [d for _, d, _ in success_trajs[:n_success_diag]]
        + [d for _, d, _ in fail_trajs[:n_fail_diag]]
    )

    # ------------------------------------------------------------------
    # Diagnostics generation
    # ------------------------------------------------------------------
    if diagnostic_freq != 0 and diag_traj_data:
        print(f'\nGenerating diagnostics for {len(diag_traj_data)} trajectories...')
        try:
            generate_all_diagnostics(agent, diag_traj_data, step_i=ckpt_step, variant=variant)
        except Exception as exc:
            print(f'[Diagnostics] Failed: {exc}')
            traceback.print_exc()
    else:
        print('\nDiagnostics skipped (diagnostic_freq=0 or no trajectories).')

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------
    summary_path = os.path.join(output_dir, 'summary.csv')
    fieldnames = [
        'rollout_id', 'is_success', 'episode_return', 'episode_len',
        'delta_norm_mean', 'base_norm_mean', 'clipping_rate_mean',
    ]

    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, stats in enumerate(all_stats):
            writer.writerow({'rollout_id': idx, **stats})

        # Aggregate summary row
        writer.writerow({
            'rollout_id': 'AGGREGATE',
            'is_success': f'{np.mean([s["is_success"] for s in all_stats]):.4f}',
            'episode_return': f'{np.mean([s["episode_return"] for s in all_stats]):.4f}',
            'episode_len': f'{np.mean([s["episode_len"] for s in all_stats]):.2f}',
            'delta_norm_mean': f'{np.mean([s["delta_norm_mean"] for s in all_stats]):.4f}',
            'base_norm_mean': f'{np.mean([s["base_norm_mean"] for s in all_stats]):.4f}',
            'clipping_rate_mean': f'{np.mean([s["clipping_rate_mean"] for s in all_stats]):.4f}',
        })

    # Print aggregate summary
    success_rate = float(np.mean([s['is_success'] for s in all_stats]))
    avg_return = float(np.mean([s['episode_return'] for s in all_stats]))
    avg_len = float(np.mean([s['episode_len'] for s in all_stats]))

    print('\n' + '=' * 60)
    print('EVALUATION SUMMARY')
    print('=' * 60)
    print(f'  Checkpoint step  : {ckpt_step}')
    print(f'  Num trajectories : {num_evals}')
    print(f'  Success rate     : {success_rate:.4f}  ({int(success_rate*num_evals)}/{num_evals})')
    print(f'  Avg return       : {avg_return:.4f}')
    print(f'  Avg episode len  : {avg_len:.2f}')
    print(f'  Summary CSV      : {summary_path}')
    print(f'  Diagnostics dir  : {os.path.join(output_dir, "diagnostics")}')
    print('=' * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a Residual SAC checkpoint.'
    )

    # ------------------------------------------------------------------
    # Evaluation-specific arguments (not in training launch script)
    # ------------------------------------------------------------------
    parser.add_argument(
        '--checkpoint_dir', required=True, type=str,
        help='Path to the checkpoint directory to load.'
    )
    parser.add_argument(
        '--output_dir', required=True, type=str,
        help='Directory to write summary.csv and diagnostics to.'
    )
    parser.add_argument(
        '--num_evals', default=50, type=int,
        help='Number of evaluation trajectories to run.'
    )
    parser.add_argument(
        '--diagnostic_freq', default=5, type=int,
        help=(
            'Number of success and fail trajectories to run diagnostics for '
            '(up to this many from each bucket). 0 disables diagnostics.'
        )
    )

    # ------------------------------------------------------------------
    # Training parameters (mirrors launch_train_sim_residual.py)
    # ------------------------------------------------------------------
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--env', default='libero', type=str)
    parser.add_argument('--add_states', default=1, type=int)
    parser.add_argument('--resize_image', default=-1, type=int)
    parser.add_argument('--query_freq', default=10, type=int)
    parser.add_argument('--chunk_len', default=10, type=int)
    parser.add_argument('--pi_05_config', default='', type=str)
    parser.add_argument('--pi_05_ckpt_dir', default='', type=str)
    parser.add_argument('--libero_task', default='', type=str)
    parser.add_argument('--cartpole_horizon', default=100, type=int)

    # Residual
    parser.add_argument('--residual_alpha', default=0.5, type=float)
    parser.add_argument('--predict_a_exec', default=0, type=int)
    parser.add_argument('--use_zero_residual_initially', default=1, type=int)

    # Algorithm
    parser.add_argument('--algo', default='residual_sac', type=str,
                        choices=['sac', 'residual_sac', 'q_weighted_pg',
                                 'residual_grpo', 'residual_parl', 'residual_gradq'])

    # PPO/GRPO
    parser.add_argument('--grpo_num_samples', default=8, type=int)
    parser.add_argument('--clip_epsilon', default=0.2, type=float)
    parser.add_argument('--clip_min_epsilon_multiplier', default=1.0, type=float)
    parser.add_argument('--clip_max_epsilon_multiplier', default=1.0, type=float)
    parser.add_argument('--entropy_coeff', default=1e-3, type=float)
    parser.add_argument('--advantage_critic_reduction', default='mean', type=str)
    parser.add_argument('--adv_clip_min', default=None, type=float)
    parser.add_argument('--adv_clip_max', default=None, type=float)
    parser.add_argument('--log_ratio_clip', default=20.0, type=float)
    parser.add_argument('--log_prob_clip', default=50.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--use_huber_loss', default=0, type=int)
    parser.add_argument('--huber_delta', default=1.0, type=float)
    parser.add_argument('--num_critic_updates', default=2, type=int)
    parser.add_argument('--num_actor_updates', default=4, type=int)
    parser.add_argument('--on_policy_ppo', default=0, type=int)
    parser.add_argument('--normalize_advantages', default=0, type=int)

    # BC / success buffer
    parser.add_argument('--bc_reg_coeff', default=0.0, type=float)
    parser.add_argument('--bc_on_success_only', default=0, type=int)
    parser.add_argument('--success_buffer_ratio', default=0.0, type=float)
    parser.add_argument('--success_buffer_min_size', default=100, type=int)
    parser.add_argument('--bc_warmup_steps', default=0, type=int)
    parser.add_argument('--bc_warmup_num_critic_updates', default=10, type=int)
    parser.add_argument('--bc_warmup_num_actor_updates', default=1, type=int)

    # Policy std
    parser.add_argument('--learn_std', default=1, type=int)
    parser.add_argument('--log_std_min', default=-5.0, type=float)
    parser.add_argument('--log_std_max', default=2.0, type=float)

    # VLM embedding
    parser.add_argument('--use_vlm_embedding', default=0, type=int)
    parser.add_argument('--vlm_embedding_dim', default=2048, type=int)
    parser.add_argument('--vlm_seq_len', default=16, type=int)

    # PARL
    parser.add_argument('--parl_num_samples', default=16, type=int)
    parser.add_argument('--parl_num_elites', default=4, type=int)
    parser.add_argument('--parl_num_grad_steps', default=5, type=int)
    parser.add_argument('--parl_step_size', default=0.01, type=float)

    # Reward shaping
    parser.add_argument('--reward_type', default='sparse', type=str,
                        choices=['sparse', 'dense'])

    # ------------------------------------------------------------------
    # Train kwargs (architecture / optimiser defaults — must match training)
    # ------------------------------------------------------------------
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(256, 256, 256),
        cnn_features=(64, 64, 64, 64),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding='VALID',
        latent_dim=200,
        discount=0.999,
        tau=0.005,
        critic_reduction='mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        backup_entropy=False,
        critic_pop_base_actions=True,
        clip_temp=True,
        clip_min_temp=0.01,
        clip_max_temp=2.0,
    )

    variant, _ = parse_training_args(train_args_dict, parser)

    # Convert integer flags to booleans
    variant['use_zero_residual_initially'] = bool(variant.get('use_zero_residual_initially', 1))
    variant['predict_a_exec'] = bool(variant.get('predict_a_exec', 0))
    variant['backup_entropy'] = bool(variant.get('backup_entropy', 0))
    variant['critic_pop_base_actions'] = bool(variant.get('critic_pop_base_actions', 1))
    variant['clip_temp'] = bool(variant.get('clip_temp', 1))
    variant['use_huber_loss'] = bool(variant.get('use_huber_loss', 0))
    variant['bc_on_success_only'] = bool(variant.get('bc_on_success_only', 0))
    variant['on_policy_ppo'] = bool(variant.get('on_policy_ppo', 0))
    variant['normalize_advantages'] = bool(variant.get('normalize_advantages', 0))
    variant['learn_std'] = bool(variant.get('learn_std', 1))
    variant['use_vlm_embedding'] = bool(variant.get('use_vlm_embedding', 0))
    variant['add_states'] = bool(variant.get('add_states', 1))

    run_evaluation(variant)
    sys.exit(0)
