"""Residual Q-weighted PG / GRPO Learner for Pixel Observations.

This module implements a Residual GRPO/Q-weighted PG agent where:
- A frozen base policy (e.g., Pi-0.5) produces base action chunks
- The actor predicts residual actions in environment action space
- Executed actions are: a_exec = clip(base_action + alpha * delta, -1, 1)
- Critic learns Q(s, a_exec) via TD learning
- Actor uses PPO-clipped objective with:
  - Q-weighted PG: raw Q values as advantages
  - GRPO: Q - mean(Q) as advantages (group baseline subtraction)

Algorithms:
- 'q_weighted_pg': Uses raw Q as advantage (no baseline)
- 'residual_grpo': Uses Q - mean(Q) as advantage (GRPO baseline)
"""

import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union, Any

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state

from jaxrl2.agents.agent import Agent
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.pixel_sac.residual_actor_updater import update_actor_residual_ppo
from jaxrl2.agents.pixel_sac.residual_critic_updater import update_critic_residual
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any


@functools.partial(jax.jit, static_argnames=(
    'critic_reduction', 'color_jitter', 'aug_next', 'num_cameras',
    'query_frequency', 'action_dim', 'grpo_num_samples', 'advantage_critic_reduction',
    'use_grpo_baseline',
))
def _update_residual_ppo_jit(
    rng: PRNGKey,
    actor: TrainState,
    target_actor_params: Params,
    critic: TrainState,
    target_critic_params: Params,
    batch: DatasetDict,
    discount: float,
    tau: float,
    actor_tau: float,
    residual_alpha: float,
    critic_reduction: str,
    color_jitter: bool,
    aug_next: bool,
    num_cameras: int,
    query_frequency: int,
    action_dim: int,
    grpo_num_samples: int,
    clip_epsilon: float,
    clip_min_epsilon_multiplier: float,
    clip_max_epsilon_multiplier: float,
    entropy_coeff: float,
    advantage_critic_reduction: str,
    use_grpo_baseline: bool,
    adv_clip_min: Optional[float],
    adv_clip_max: Optional[float],
) -> Tuple[PRNGKey, TrainState, Params, TrainState, Params, Dict[str, float]]:
    """JIT-compiled update function for Residual PPO/GRPO."""
    
    # Data augmentation for pixels
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']
    
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:, :, :, i*3:(i+1)*3].set(
                        (color_transform(key, aug_pixels[:, :, :, i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                    )
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:, :, :, i*3:(i+1)*3].set(
                        (color_transform(key, aug_next_pixels[:, :, :, i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                    )
            else:
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})
    
    # Critic update (online TD learning, same as SAC)
    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    # Note: For critic update, we use current actor (not target) to sample next actions
    # PPO doesn't use temperature for critic backup
    temp_dummy = None
    new_critic, critic_info = update_critic_residual(
        key, actor, critic, target_critic, temp_dummy, batch,
        discount, residual_alpha, query_frequency,
        critic_reduction=critic_reduction, backup_entropy=False,
    )
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)
    
    # Actor update with PPO/GRPO
    key, rng = jax.random.split(rng)
    target_actor = actor.replace(params=target_actor_params)
    target_critic_for_actor = critic.replace(params=target_critic_params)
    new_actor, actor_info = update_actor_residual_ppo(
        key, actor, target_actor, target_critic_for_actor, batch,
        residual_alpha, query_frequency, action_dim,
        grpo_num_samples=grpo_num_samples,
        clip_epsilon=clip_epsilon,
        clip_min_epsilon_multiplier=clip_min_epsilon_multiplier,
        clip_max_epsilon_multiplier=clip_max_epsilon_multiplier,
        entropy_coeff=entropy_coeff,
        advantage_critic_reduction=advantage_critic_reduction,
        use_grpo_baseline=use_grpo_baseline,
        adv_clip_min=adv_clip_min,
        adv_clip_max=adv_clip_max,
    )
    
    # Target actor update
    new_target_actor_params = soft_target_update(new_actor.params, target_actor_params, actor_tau)

    return rng, new_actor, new_target_actor_params, new_critic, new_target_critic_params, {
        **critic_info,
        **actor_info,
    }


class PixelPPOResidualLearner(Agent):
    """Residual Q-weighted PG / GRPO Learner for pixel observations.
    
    This agent uses:
    - PPO-clipped objective for actor updates
    - TD learning for critic (same as SAC)
    - Target networks for both actor and critic
    - Q values as advantages (raw or with group baseline)
    """

    def __init__(
        self,
        seed: int,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Algorithm selection
        algo: str = 'residual_grpo',  # 'q_weighted_pg' or 'residual_grpo'
        # Actor
        actor_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256, 256),
        latent_dim: int = 50,
        dropout_rate: float = 0.0,
        encoder_type: str = 'resnet_small',
        encoder_norm: str = 'batch',
        use_bottleneck: bool = True,
        use_spatial_softmax: bool = False,
        softmax_temperature: float = 1.0,
        # Critic
        critic_lr: float = 3e-4,
        critic_pop_base_actions: bool = True,
        num_qs: int = 2,
        # Target networks
        tau: float = 0.005,
        actor_tau: float = 0.005,
        discount: float = 0.99,
        critic_reduction: str = 'min',
        # Residual
        residual_alpha: float = 1.0,
        action_magnitude: float = 0.1,
        # Data augmentation
        color_jitter: bool = True,
        aug_next: bool = True,
        num_cameras: int = 1,
        # PPO / GRPO parameters
        grpo_num_samples: int = 8,
        clip_epsilon: float = 0.2,
        clip_min_epsilon_multiplier: float = 1.0,
        clip_max_epsilon_multiplier: float = 1.0,
        entropy_coeff: float = 0.0,
        advantage_critic_reduction: str = 'mean',
        adv_clip_min: Optional[float] = None,
        adv_clip_max: Optional[float] = None,
        # Other
        decay_steps: Optional[int] = None,
        cnn_features: Sequence[int] = (32, 64, 128, 256),
        cnn_strides: Sequence[int] = (2, 2, 2, 2),
        cnn_padding: str = 'VALID',
    ):
        """Initialize Residual PPO/GRPO learner.
        
        Args:
            algo: 'q_weighted_pg' (raw Q) or 'residual_grpo' (Q - mean(Q))
            actor_tau: Target actor EMA coefficient
            grpo_num_samples: Number of samples per state for GRPO
            clip_epsilon: PPO clipping epsilon
            entropy_coeff: Entropy bonus coefficient (default 0)
            adv_clip_min: Optional lower bound for advantage clipping
            adv_clip_max: Optional upper bound for advantage clipping
        """
        # Validate algo
        assert algo in ['q_weighted_pg', 'residual_grpo'], \
            f"algo must be 'q_weighted_pg' or 'residual_grpo', got {algo}"
        
        self._residual_alpha = jnp.asarray(residual_alpha, dtype=jnp.float32)
        self.color_jitter = color_jitter
        self.aug_next = aug_next
        self.num_cameras = num_cameras
        
        # Infer query_frequency from actions shape (like SAC does)
        self.query_frequency = actions.shape[1]

        # Action dimensions: (query_frequency, action_dim) -> flattened
        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]
        self.action_dim_per_step = actions.shape[-1]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        # Select encoder
        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError(f'Encoder type not found: {encoder_type}')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])
        
        # Actor: outputs residual actions (delta) in [-action_magnitude, action_magnitude]
        policy_def = LearnedStdTanhNormalPolicy(
            hidden_dims, self.action_dim, 
            dropout_rate=dropout_rate, 
            low=-action_magnitude, 
            high=action_magnitude
        )

        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
            pop_base_actions=False
        )
        print(f"Residual PPO Actor: {actor_def}")
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None

        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
            batch_stats=actor_batch_stats,
        )

        # Critic: takes observations and executed actions
        critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        critic_def = PixelMultiplexer(
            encoder=encoder_def,
            network=critic_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
            pop_base_actions=critic_pop_base_actions
        )
        print(f"Residual PPO Critic: {critic_def}")
        
        actions_flat = actions.reshape(actions.shape[0], -1)
        critic_def_init = critic_def.init(critic_key, observations, actions_flat)

        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
            batch_stats=critic_batch_stats
        )
        target_critic_params = copy.deepcopy(critic_params)
        
        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        
        # Target actor for PPO/GRPO
        self._target_actor_params = copy.deepcopy(actor_params)
        
        # Algorithm selection
        self.algo = algo
        self.use_grpo_baseline = (algo == 'residual_grpo')
        
        # PPO/GRPO specific parameters
        self.actor_tau = actor_tau
        self.grpo_num_samples = grpo_num_samples
        self.clip_epsilon = clip_epsilon
        self.clip_min_epsilon_multiplier = clip_min_epsilon_multiplier
        self.clip_max_epsilon_multiplier = clip_max_epsilon_multiplier
        self.entropy_coeff = entropy_coeff
        self.advantage_critic_reduction = advantage_critic_reduction
        self.adv_clip_min = adv_clip_min
        self.adv_clip_max = adv_clip_max

        if algo == 'q_weighted_pg':
            assert grpo_num_samples >= 1
        elif algo == 'residual_grpo':
            assert grpo_num_samples >= 2


            
        print(f'Residual PPO Learner initialized with:')
        print(f'  algo: {self.algo}')
        print(f'  use_grpo_baseline: {self.use_grpo_baseline}')
        print(f'  residual_alpha: {self._residual_alpha}')
        print(f'  query_frequency: {self.query_frequency}')
        print(f'  action_dim: {self.action_dim}')
        print(f'  action_dim_per_step: {self.action_dim_per_step}')
        print(f'  action_chunk_shape: {self.action_chunk_shape}')
        print(f'  critic_reduction: {self.critic_reduction}')
        print(f'  actor_tau: {self.actor_tau}')
        print(f'  grpo_num_samples: {self.grpo_num_samples}')
        print(f'  clip_epsilon: {self.clip_epsilon}')
        print(f'  entropy_coeff: {self.entropy_coeff}')
        print(f'  advantage_critic_reduction: {self.advantage_critic_reduction}')
        print(f'  adv_clip_min: {self.adv_clip_min}')
        print(f'  adv_clip_max: {self.adv_clip_max}')

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        """Update actor, critic using PPO/GRPO.
        
        Args:
            batch: Batch of transitions with observations containing 'base_action'.
            
        Returns:
            Dictionary of training metrics.
        """
        new_rng, new_actor, new_target_actor, new_critic, new_target_critic, info = _update_residual_ppo_jit(
            self._rng,
            self._actor,
            self._target_actor_params,
            self._critic,
            self._target_critic_params,
            batch,
            self.discount,
            self.tau,
            self.actor_tau,
            self._residual_alpha,
            self.critic_reduction,
            self.color_jitter,
            self.aug_next,
            self.num_cameras,
            self.query_frequency,
            self.action_dim_per_step,
            self.grpo_num_samples,
            self.clip_epsilon,
            self.clip_min_epsilon_multiplier,
            self.clip_max_epsilon_multiplier,
            self.entropy_coeff,
            self.advantage_critic_reduction,
            self.use_grpo_baseline,
            self.adv_clip_min,
            self.adv_clip_max,
        )
        
        self._rng = new_rng
        self._actor = new_actor
        self._target_actor_params = new_target_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        
        # Add residual_alpha to info for logging
        info['residual/alpha'] = float(self._residual_alpha)
        info['algo'] = self.algo
        
        return info

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        """Perform evaluation visualization."""
        from examples.train_utils_sim_residual import make_multiple_value_reward_visulizations_residual
        make_multiple_value_reward_visulizations_residual(self, variant, i, eval_buffer, wandb_logger)

    def make_value_reward_visulization(self, variant, trajs):
        """Create value/reward visualization for trajectories."""
        num_traj = len(trajs['rewards'])
        traj_images = []

        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            next_observations = trajs['next_observations'][itraj]
            actions = trajs['actions'][itraj]  # These are delta actions
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            q_pred = []

            for t in range(0, len(actions)):
                action = actions[t][None]  # (1, query_frequency, action_dim)
                obs_pixels = observations['pixels'][t]
                base_action = observations['base_action'][t]  # (chunk_len, action_dim, 1)

                obs_dict = {'pixels': obs_pixels[None], 'base_action': base_action[None]}
                for k, v in observations.items():
                    if k not in ['pixels', 'base_action']:
                        obs_dict[k] = v[t][None]

                # Compose executed action for Q evaluation
                base_action_squeezed = base_action.squeeze(-1)  # (chunk_len, action_dim)
                a_exec = np.clip(
                    base_action_squeezed[:self.query_frequency] + self._residual_alpha * action.squeeze(0), 
                    -1.0, 1.0
                )
                a_exec_flat = a_exec.reshape(1, -1)

                q_value = get_value_residual(a_exec_flat, obs_dict, self._critic)
                q_pred.append(q_value)

            traj_images.append(make_visual_residual(q_pred, rewards, masks, observations['pixels']))
        
        print('Finished reward value visuals for residual PPO.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'target_actor_params': self._target_actor_params,
            'residual_alpha': self._residual_alpha,
            'algo': self.algo,
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        if 'residual_alpha' in output_dict:
            self._residual_alpha = jnp.asarray(output_dict['residual_alpha'], dtype=jnp.float32)
        if 'target_actor_params' in output_dict:
            self._target_actor_params = output_dict['target_actor_params']
        if 'algo' in output_dict:
            self.algo = output_dict['algo']
        print(f'Restored residual PPO checkpoint from {dir} (algo: {self.algo})')


@functools.partial(jax.jit)
def get_value_residual(action, observation, critic):
    """Get Q value for executed action."""
    input_collections = {'params': critic.params}
    q_pred = critic.apply_fn(input_collections, observation, action)
    return q_pred


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr


def make_visual_residual(q_estimates, rewards, masks, images):
    """Create visualization of Q values, rewards, and masks for a trajectory."""
    q_estimates_np = np.stack(q_estimates, 0).squeeze()
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates_np)])

    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    if len(q_estimates_np.shape) == 2:
        for i in range(q_estimates_np.shape[1]):
            axs[1].plot(q_estimates_np[:, i], linestyle='--', marker='o')
    else:
        axs[1].plot(q_estimates_np, linestyle='--', marker='o')
    axs[1].set_ylabel('Q values')
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('Rewards')
    axs[2].set_xlim([0, len(rewards)])
    
    axs[3].plot(masks, linestyle='--', marker='d')
    axs[3].set_ylabel('Masks')
    axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return out_image
