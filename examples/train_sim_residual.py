#! /usr/bin/env python
"""Main training script for Residual RL in simulation.

This script sets up the Residual RL training pipeline where:
- A frozen base policy (Pi-0.5) produces base action chunks
- Residual policy predicts residual actions in environment action space
- Executed actions are: a_exec = clip(base_action + alpha * delta, -1, 1)

Supported algorithms:
- 'sac': Standard SAC (no residual, no base policy)
- 'residual_sac': SAC-style (maximize Q - alpha * log_prob)
- 'q_weighted_pg': Q-weighted Policy Gradient with PPO clipping (raw Q as advantage)
- 'residual_grpo': GRPO with PPO clipping (Q - mean(Q) as advantage)
"""

import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib
import copy

import jax
import jax.numpy as jnp
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.agents.pixel_sac.pixel_sac_residual_learner import PixelSACResidualLearner
from jaxrl2.agents.pixel_sac.pixel_ppo_residual_learner import PixelPPOResidualLearner
from jaxrl2.agents.pixel_sac.pixel_parl_residual_learner import PixelPARLResidualLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gym
from gym.spaces import Dict, Box

# Heavy env/model imports are deferred to avoid requiring all deps for simple test envs
# libero, gym_aloha, openpi are imported conditionally inside main_residual()

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim_residual import trajwise_alternating_training_loop_residual
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
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
        self.use_vlm_embedding = variant.get('use_vlm_embedding', False)
        
        # Determine dimensions based on environment
        if variant.env == 'libero':
            state_dim = 8
            action_dim = 7
        elif variant.env == 'aloha_cube':
            state_dim = 14
            action_dim = 14
        elif variant.env == 'cartpole':
            state_dim = 4
            action_dim = 1
        else:
            raise NotImplementedError(f"Unknown env: {variant.env}")
        
        chunk_len = variant.chunk_len  # e.g., 10 for Pi-0.5
        query_freq = variant.query_freq  # e.g., 5 or 10
        obs_dict = {}
        
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if self.use_vlm_embedding:
            vlm_embedding_dim = variant.get('vlm_embedding_dim', 2048)
            obs_dict['vlm_embedding'] = Box(
                low=-np.inf, high=np.inf,
                shape=(vlm_embedding_dim, 1),
                dtype=np.float32
            )
        
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
        from libero.libero import benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
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
        import gymnasium
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gymnasium.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 500
    elif variant.env == 'cartpole':
        from envs.cartpole_env import CartPoleEnv
        render_size = variant.resize_image if variant.resize_image > 0 else 100
        env = CartPoleEnv(render_size=render_size, horizon=variant.get('cartpole_horizon', 100))
        env.seed(variant.seed)
        eval_env = CartPoleEnv(render_size=render_size, horizon=variant.get('cartpole_horizon', 100))
        eval_env.seed(variant.seed + 100)
        variant.env_max_reward = 0  # best reward is 0 (theta=0)
        variant.max_timesteps = variant.get('cartpole_horizon', 100)
        variant.task_description = 'Balance the pole upright'
        print("CartPole test environment initialized.")
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
    
    # Load frozen base policy (Pi-0.5 or zero policy for test envs)
    if variant.env == 'cartpole':
        from envs.zero_base_policy import ZeroBasePolicy
        agent_dp = ZeroBasePolicy(
            action_dim=variant.action_dim,
            chunk_len=variant.chunk_len,
            vlm_embedding_dim=variant.get('vlm_embedding_dim', 2048),
            vlm_seq_len=variant.get('vlm_seq_len', 16),
        )
        print(f"Using ZeroBasePolicy for CartPole (action_dim={variant.action_dim}, chunk_len={variant.chunk_len})")
    else:
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config
        from openpi.shared import download
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
    
    # Get algorithm selection
    algo = variant.get('algo', 'residual_sac')
    
    # Create Residual RL agent based on algorithm
    if algo == 'sac':
        # Plain SAC (no residual)
        # Whitelist only the params PixelSACLearner accepts
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
        sac_agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **sac_kwargs)
        # Wrap with interface expected by the training loop
        sac_agent._residual_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        sac_agent.algo = 'sac'
        sac_agent.predict_a_exec = True  # actor output IS the action (no residual composition)
        sac_agent.query_frequency = variant.query_freq
        sac_agent._num_critic_updates = variant.get('num_critic_updates', 1)
        sac_agent._num_actor_updates = variant.get('num_actor_updates', 1)
        # Provide update_critic / update_actor / update_actor_bc expected by the training loop.
        # PixelSACLearner.update() does critic+actor+temp in one fused jit call,
        # so we run the full update in update_critic and make update_actor a no-op.
        sac_agent.update_critic = sac_agent.update
        sac_agent.update_actor = lambda batch: {}  # already done in update_critic
        sac_agent.update_actor_bc = lambda batch: {}  # no BC for plain SAC
        agent = sac_agent
        print(f"Initialized plain SAC (no residual)")
    elif algo == 'residual_sac':
        # SAC learner
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
        kwargs['predict_a_exec'] = variant.get('predict_a_exec', False)
        kwargs['use_vlm_embedding'] = variant.get('use_vlm_embedding', False)
        agent = PixelSACResidualLearner(variant.seed, sample_obs, sample_action, **kwargs)
        print(f"Initialized Residual SAC with alpha={variant.residual_alpha}, predict_a_exec={variant.get('predict_a_exec', False)}")
    elif algo in ['q_weighted_pg', 'residual_grpo']:
        # PPO/GRPO learner
        ppo_kwargs = {k: v for k, v in kwargs.items() if k not in ['temp_lr', 'init_temperature', 'backup_entropy', 'clip_temp', 'clip_min_temp', 'clip_max_temp', 'target_entropy']}
        ppo_kwargs['algo'] = algo
        ppo_kwargs['grpo_num_samples'] = variant.get('grpo_num_samples', 8)
        ppo_kwargs['clip_epsilon'] = variant.get('clip_epsilon', 0.2)
        ppo_kwargs['clip_min_epsilon_multiplier'] = variant.get('clip_min_epsilon_multiplier', 1.0)
        ppo_kwargs['clip_max_epsilon_multiplier'] = variant.get('clip_max_epsilon_multiplier', 1.0)
        ppo_kwargs['entropy_coeff'] = variant.get('entropy_coeff', 1e-3)
        ppo_kwargs['advantage_critic_reduction'] = variant.get('advantage_critic_reduction', 'mean')
        ppo_kwargs['adv_clip_min'] = variant.get('adv_clip_min', None)
        ppo_kwargs['adv_clip_max'] = variant.get('adv_clip_max', None)
        # Stability parameters
        ppo_kwargs['log_ratio_clip'] = variant.get('log_ratio_clip', 20.0)
        ppo_kwargs['log_prob_clip'] = variant.get('log_prob_clip', 50.0)
        ppo_kwargs['max_grad_norm'] = variant.get('max_grad_norm', 1.0)
        ppo_kwargs['use_huber_loss'] = variant.get('use_huber_loss', False)
        ppo_kwargs['huber_delta'] = variant.get('huber_delta', 1.0)
        # Update ratio control
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
        print(f"Initialized Residual {algo.upper()} with alpha={variant.residual_alpha}, predict_a_exec={variant.get('predict_a_exec', False)}")
    elif algo == 'residual_parl':
        # PARL: Policy-Agnostic RL (Best-of-N + Grad-Q + BC distillation)
        parl_kwargs = {k: v for k, v in kwargs.items() if k not in ['temp_lr', 'init_temperature', 'backup_entropy', 'clip_temp', 'clip_min_temp', 'clip_max_temp', 'target_entropy']}
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
        print(f"Initialized Residual PARL with alpha={variant.residual_alpha}, "
              f"N={parl_kwargs['parl_num_samples']}, K={parl_kwargs['parl_num_elites']}, "
              f"grad_steps={parl_kwargs['parl_num_grad_steps']}, step_size={parl_kwargs['parl_step_size']}")
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Replay buffer
    online_buffer_size = variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    
    # Success replay buffer (stores only transitions from successful episodes)
    success_buffer_ratio = variant.get('success_buffer_ratio', 0.0)
    if success_buffer_ratio > 0.0:
        # Smaller capacity since only success data goes here
        success_buffer_size = max(10000, int(online_buffer_size * 0.5))
        success_replay_buffer = ReplayBuffer(
            dummy_env.observation_space, dummy_env.action_space, int(success_buffer_size)
        )
        success_replay_buffer.seed(variant.seed + 1)
        print(f'Created success replay buffer with capacity {success_buffer_size}')
    else:
        success_replay_buffer = None
    
    # Start training
    trajwise_alternating_training_loop_residual(
        variant, agent, env, eval_env, online_replay_buffer, replay_buffer, 
        wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp,
        success_replay_buffer=success_replay_buffer,
    )
