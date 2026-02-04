#! /usr/bin/env python
"""Main training script for Residual SAC in simulation.

This script sets up the Residual SAC training pipeline where:
- A frozen base policy (Pi-0.5) produces base action chunks
- SAC predicts residual actions in environment action space
- Executed actions are: a_exec = clip(base_action + alpha * delta, -1, 1)
"""

import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib
import copy

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_residual_learner import PixelSACResidualLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim_residual import trajwise_alternating_training_loop_residual
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config

# Monkey-patch the Policy class to fix tokenized_prompt dimension issue
# The openpi TokenizePrompt transform adds a batch dimension, and then
# Policy.infer adds another batch dimension, resulting in shape [1,1,L]
# instead of [1,L]. This patch squeezes the extra dimension.
_original_policy_infer = None

def _patched_infer(self, obs, **kwargs):
    import jax.numpy as jnp
    
    # Store original _input_transform
    original_input_transform = self._input_transform
    
    def fixed_input_transform(inputs):
        result = original_input_transform(inputs)
        # Squeeze the extra batch dimension from tokenized_prompt if it exists
        if "tokenized_prompt" in result and result["tokenized_prompt"] is not None:
            arr = result["tokenized_prompt"]
            if hasattr(arr, 'ndim') and arr.ndim >= 2:
                # Remove the first dimension added by TokenizePrompt
                result["tokenized_prompt"] = arr[0] if arr.shape[0] == 1 else arr
        if "tokenized_prompt_mask" in result and result["tokenized_prompt_mask"] is not None:
            arr = result["tokenized_prompt_mask"]
            if hasattr(arr, 'ndim') and arr.ndim >= 2:
                result["tokenized_prompt_mask"] = arr[0] if arr.shape[0] == 1 else arr
        return result
    
    # Temporarily replace _input_transform
    self._input_transform = fixed_input_transform
    try:
        return _original_policy_infer(self, obs, **kwargs)
    finally:
        self._input_transform = original_input_transform

def patch_openpi_policy():
    """Apply the monkey-patch to fix tokenized_prompt dimensions."""
    global _original_policy_infer
    from openpi.policies import policy as _policy_module
    if _original_policy_infer is None:
        _original_policy_infer = _policy_module.Policy.infer
        _policy_module.Policy.infer = _patched_infer

# Apply the patch on import
patch_openpi_policy()
from openpi.policies import policy_config
from openpi.shared import download

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnvResidual(gym.ObservationWrapper):
    """Dummy environment for Residual SAC with base_action in observation space.
    
    Observation space includes:
        - pixels: (H, W, 3 * num_cameras, 1)
        - state: (state_dim, 1) [optional]
        - base_action: (chunk_len, action_dim, 1) - base action from frozen policy
        
    Action space:
        - (chunk_len, action_dim) - residual/delta actions in env action space
    """

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        
        # Determine dimensions based on environment
        if variant.env == 'libero':
            state_dim = 8
            action_dim = 7
        elif variant.env == 'aloha_cube':
            state_dim = 14
            action_dim = 14
        else:
            raise NotImplementedError(f"Unknown env: {variant.env}")
        
        chunk_len = variant.chunk_len  # e.g., 10 for Pi-0.5
        query_freq = variant.query_freq  # e.g., 5 or 10
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        
        if variant.add_states:
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        
        # Base action from frozen policy (clipped to [-1, 1])
        obs_dict['base_action'] = Box(
            low=-1.0, high=1.0, 
            shape=(chunk_len, action_dim, 1), 
            dtype=np.float32
        )
        
        self.observation_space = Dict(obs_dict)
        
        # Action space: residual actions (delta) for the full chunk
        # Shape: (query_freq, action_dim)
        self.action_space = Box(
            low=-1.0, high=1.0, 
            shape=(query_freq, action_dim), 
            dtype=np.float32
        )
        
        # Store action_dim for use in training
        variant.action_dim = action_dim


def main_residual(variant):
    """Main function for Residual SAC training."""
    
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    
    # Shard leading dimension (batch dimension) across all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # Prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
        
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
   
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    
    # Environment setup
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_10"]()
        task_id = 8  # KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 224, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 500
        print("Libero environment initialized with task description:", task_description)
    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 500
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")

    # WandB setup
    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != '', variant, variant.wandb_project, 
        experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name
    )

    # Create dummy env for observation/action space specs
    dummy_env = DummyEnvResidual(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    
    print('Residual SAC sample obs shapes:', [(k, v.shape) for k, v in sample_obs.items()])
    print('Residual SAC sample action shape:', sample_action.shape)
    
    # Load frozen base policy (Pi-0.5)
    if variant.env == 'libero':
        config = openpi_config.get_config(variant.pi_05_config)
        checkpoint_dir = download.maybe_download(variant.pi_05_ckpt_dir)
    elif variant.env == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()
    
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print(f"Loaded frozen Pi-0.5 policy from {checkpoint_dir}")
    
    # Add residual_alpha to kwargs for the learner
    kwargs['residual_alpha'] = variant.residual_alpha
    
    # Create Residual SAC agent
    agent = PixelSACResidualLearner(variant.seed, sample_obs, sample_action, **kwargs)
    print(f"Initialized Residual SAC with alpha={variant.residual_alpha}")

    # Replay buffer
    online_buffer_size = variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    
    # Start training
    trajwise_alternating_training_loop_residual(
        variant, agent, env, eval_env, online_replay_buffer, replay_buffer, 
        wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp
    )
