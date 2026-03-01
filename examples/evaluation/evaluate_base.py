#!/usr/bin/env python
"""Standalone evaluation script for the base policy (Pi-0.5) only.

Loads a frozen Pi-0.5 policy, runs num_evals rollouts using only the base
policy actions (no residual), and writes a summary CSV to output_dir.

Usage:
    python -m examples.evaluation.evaluate_base \\
        --output_dir /path/to/output \\
        --num_evals 50 \\
        --env libero \\
        --pi_05_config pi05_libero_custom_low_mem_ep5_discrete_state_input_False_4k \\
        --pi_05_ckpt_dir /path/to/pi05_ckpt \\
        --query_freq 10 \\
        --chunk_len 10
"""

import os

# Tell XLA to use Triton GEMM — improves steps/sec by ~30% on some GPUs
_xla_flags = os.environ.get('XLA_FLAGS', '')
_xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = _xla_flags

import sys
import argparse
import csv

import jax
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio

import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from jaxrl2.utils.general_utils import AttrDict

from examples.train_sim_residual import (
    _get_libero_env,
    patch_openpi_policy,
)
from examples.train_utils_sim_residual import (
    obs_to_img,
    obs_to_pi_zero_input,
)

_home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(_home_dir, 'jax_compilation_cache'))

# Apply the openpi Policy monkey-patch (fixes tokenized_prompt batch-dim issue)
patch_openpi_policy()


# ---------------------------------------------------------------------------
# Video writing helper
# ---------------------------------------------------------------------------

def _write_mp4(frames, filepath, fps=20):
    """Write a list of (H, W, 3) uint8 RGB frames to an .mp4 file."""
    if not frames:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    writer = imageio.get_writer(
        filepath, fps=fps, codec='libx264',
        output_params=['-pix_fmt', 'yuv420p'],
    )
    for frame in frames:
        h, w = frame.shape[:2]
        h = h if h % 2 == 0 else h - 1
        w = w if w % 2 == 0 else w - 1
        writer.append_data(frame[:h, :w])
    writer.close()


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_eval_trajectory(variant, agent_dp, env, rng):
    """Collect a single evaluation trajectory using only the base policy.

    Args:
        variant: Eval configuration AttrDict.
        agent_dp: Frozen base policy (Pi-0.5).
        env: Environment instance.
        rng: JAX PRNGKey.

    Returns:
        stats (dict): Per-trajectory scalar statistics.
        rng: Updated PRNGKey.
    """
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    chunk_len = variant.chunk_len

    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")

    image_list = []
    rewards = []
    episode_terminated = False
    actions = None  # set on first query

    flip_horizontal = variant.get('flip_horizontal', False)

    for t in tqdm(range(max_timesteps), desc='eval steps', leave=False):
        curr_image = obs_to_img(obs, variant)
        image_list.append(curr_image)

        if t % query_frequency == 0:
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            if flip_horizontal:
                for key in ('observation/image', 'observation/wrist_image'):
                    if key in obs_pi_zero:
                        obs_pi_zero[key] = np.ascontiguousarray(obs_pi_zero[key][:, ::-1, :])
            infer_result = agent_dp.infer(obs_pi_zero)
            base_actions = infer_result["actions"][:chunk_len]  # (chunk_len, action_dim)
            actions = np.clip(base_actions, -1.0, 1.0)

        action_t = actions[t % query_frequency]

        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated

        rewards.append(reward)
        if done:
            episode_terminated = True
            break

    rng, _ = jax.random.split(rng)

    is_success = bool(reward == env_max_reward)
    episode_return = float(np.sum(rewards))
    episode_len = t + 1

    stats = {
        'is_success': int(is_success),
        'episode_return': episode_return,
        'episode_len': episode_len,
    }
    return stats, rng, image_list


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(variant):
    """Load base policy and run evaluation rollouts."""

    # Prevent TensorFlow from grabbing GPUs
    tf.config.set_visible_devices([], 'GPU')

    output_dir = variant.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_evals = variant.num_evals

    print('=' * 60)
    print('BASE POLICY EVALUATION')
    print('=' * 60)
    print(f'  output_dir       : {output_dir}')
    print(f'  num_evals        : {num_evals}')
    print(f'  env              : {variant.env}')
    print(f'  flip_horizontal  : {variant.get("flip_horizontal", False)}')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------
    if variant.env == 'libero':
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from examples.perturbation import setup_libero_pro_env

        task_suite_name = setup_libero_pro_env(variant.task_suite_name, variant)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        if variant.libero_task:
            task_names = task_suite.get_task_names()
            matching = [i for i, name in enumerate(task_names) if name == variant.libero_task]
            if len(matching) != 1:
                raise ValueError(
                    f"Task '{variant.libero_task}' not found in {task_suite_name}. "
                    f"Available: {task_names}"
                )
            task_id = matching[0]
        else:
            task_id = variant.task_id
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 224, variant.seed)
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = variant.max_env_steps
        print(f'LIBERO task: {task_description}')

    elif 'aloha' in variant.env:
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
        variant.max_timesteps = variant.max_env_steps

    else:
        raise NotImplementedError(f'Unknown env: {variant.env}')

    # ------------------------------------------------------------------
    # Frozen base policy (Pi-0.5)
    # ------------------------------------------------------------------
    from openpi.training import config as openpi_config
    from openpi.policies import policy_config
    from openpi.shared import download

    if variant.env == 'libero':
        config = openpi_config.get_config(variant.pi_05_config)
        pi_ckpt_dir = download.maybe_download(variant.pi_05_ckpt_dir)
    elif 'aloha' in variant.env:
        config = openpi_config.get_config('pi0_aloha_sim')
        pi_ckpt_dir = download.maybe_download(
            's3://openpi-assets/checkpoints/pi0_aloha_sim'
        )
    else:
        raise NotImplementedError()

    agent_dp = policy_config.create_trained_policy(config, pi_ckpt_dir)
    print(f'Loaded frozen Pi-0.5 policy from {pi_ckpt_dir}')

    # ------------------------------------------------------------------
    # Evaluation rollouts
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(variant.seed + 789)
    all_stats = []

    videos_dir = os.path.join(output_dir, 'videos')
    os.makedirs(videos_dir, exist_ok=True)

    print(f'\nRunning {num_evals} evaluation trajectories...\n')
    for rollout_id in range(num_evals):
        print(f'--- Rollout {rollout_id + 1}/{num_evals} ---')
        stats, rng, image_list = collect_eval_trajectory(variant, agent_dp, env, rng)
        all_stats.append(stats)
        print(
            f'  return={stats["episode_return"]:.3f}  '
            f'len={stats["episode_len"]}  '
            f'success={bool(stats["is_success"])}'
        )

        video_path = os.path.join(videos_dir, f'rollout_{rollout_id:04d}.mp4')
        _write_mp4(image_list, video_path)
        print(f'  video -> {video_path}')

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------
    summary_path = os.path.join(output_dir, 'summary.csv')
    fieldnames = ['rollout_id', 'is_success', 'episode_return', 'episode_len']

    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, stats in enumerate(all_stats):
            writer.writerow({'rollout_id': idx, **stats})

        writer.writerow({
            'rollout_id': 'AGGREGATE',
            'is_success': f'{np.mean([s["is_success"] for s in all_stats]):.4f}',
            'episode_return': f'{np.mean([s["episode_return"] for s in all_stats]):.4f}',
            'episode_len': f'{np.mean([s["episode_len"] for s in all_stats]):.2f}',
        })

    success_rate = float(np.mean([s['is_success'] for s in all_stats]))
    avg_return = float(np.mean([s['episode_return'] for s in all_stats]))
    avg_len = float(np.mean([s['episode_len'] for s in all_stats]))

    print('\n' + '=' * 60)
    print('EVALUATION SUMMARY')
    print('=' * 60)
    print(f'  Num trajectories : {num_evals}')
    print(f'  Success rate     : {success_rate:.4f}  ({int(success_rate*num_evals)}/{num_evals})')
    print(f'  Avg return       : {avg_return:.4f}')
    print(f'  Avg episode len  : {avg_len:.2f}')
    print(f'  Summary CSV      : {summary_path}')
    print('=' * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the base policy (Pi-0.5) without any residual.'
    )

    parser.add_argument(
        '--output_dir', required=True, type=str,
        help='Directory to write summary.csv to.'
    )
    parser.add_argument('--num_evals', default=50, type=int,
                        help='Number of evaluation trajectories to run.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--env', default='libero', type=str)
    parser.add_argument('--resize_image', default=-1, type=int)
    parser.add_argument('--query_freq', default=10, type=int)
    parser.add_argument('--chunk_len', default=10, type=int)
    parser.add_argument('--max_env_steps', default=500, type=int,
                        help='Maximum environment steps per episode (default: 500).')
    parser.add_argument('--pi_05_config', default='', type=str)
    parser.add_argument('--pi_05_ckpt_dir', default='', type=str)
    parser.add_argument('--libero_task', default='', type=str)

    # LIBERO-PRO
    parser.add_argument('--task_suite_name', default='libero_10', type=str)
    parser.add_argument('--task_id', default=8, type=int)
    parser.add_argument('--eval_config_path', default='LIBERO-PRO/evaluation_config.yaml', type=str)
    parser.add_argument('--use_swap', default=0, type=int)
    parser.add_argument('--use_object', default=0, type=int)
    parser.add_argument('--use_language', default=0, type=int)
    parser.add_argument('--use_task', default=0, type=int)
    parser.add_argument('--use_environment', default=0, type=int)
    parser.add_argument('--flip_horizontal', default=0, type=int,
                        help='Flip observation images horizontally before passing to the model (0/1).')

    args = parser.parse_args()
    variant = AttrDict(vars(args))

    # Convert integer flags to booleans
    variant['flip_horizontal'] = bool(variant.get('flip_horizontal', 0))
    variant['use_swap'] = bool(variant.get('use_swap', 0))
    variant['use_object'] = bool(variant.get('use_object', 0))
    variant['use_language'] = bool(variant.get('use_language', 0))
    variant['use_task'] = bool(variant.get('use_task', 0))
    variant['use_environment'] = bool(variant.get('use_environment', 0))

    run_evaluation(variant)
    sys.exit(0)
