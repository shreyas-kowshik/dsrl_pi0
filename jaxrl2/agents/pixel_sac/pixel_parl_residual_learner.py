"""Residual PARL (Policy-Agnostic RL) Learner for Pixel Observations.

This module implements a Residual PARL agent where:
- A frozen base policy (e.g., Pi-0.5) produces base action chunks
- The actor predicts residual actions in environment action space
- Executed actions are: a_exec = clip(base_action + alpha * delta, -1, 1)
- Critic learns Q(s, a_exec) via TD learning (standard)
- Actor is trained by:
  1. Sampling N actions from the actor
  2. Evaluating with Q, keeping top-K elites
  3. Refining elites via gradient ascent on Q w.r.t. action
  4. Distilling the best refined action back into the actor via MSE (BC loss)

The actor NEVER differentiates through the critic. The critic only provides
stop-gradient targets, avoiding issues with tanh squashing, action clipping, etc.
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
from jaxrl2.agents.pixel_sac.residual_actor_updater import update_actor_residual_parl, update_actor_bc_residual
from jaxrl2.agents.pixel_sac.residual_critic_updater import update_critic_residual
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy, FixedStdTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any


# ============================================================
# JIT-compiled update functions
# ============================================================

@functools.partial(jax.jit, static_argnames=(
    'critic_reduction', 'color_jitter', 'aug_next', 'num_cameras',
    'query_frequency', 'use_huber_loss', 'predict_a_exec',
))
def _update_critic_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    batch: DatasetDict,
    discount: float,
    tau: float,
    residual_alpha: float,
    critic_reduction: str,
    color_jitter: bool,
    aug_next: bool,
    num_cameras: int,
    query_frequency: int,
    use_huber_loss: bool,
    huber_delta: float,
    predict_a_exec: bool = False,
) -> Tuple[PRNGKey, TrainState, Params, Dict[str, float]]:
    """JIT-compiled critic update (same as PPO/SAC — standard TD learning)."""
    
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
    
    # Critic update via TD learning
    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    temp_dummy = None
    new_critic, critic_info = update_critic_residual(
        key, actor, critic, target_critic, temp_dummy, batch,
        discount, residual_alpha, query_frequency,
        critic_reduction=critic_reduction, backup_entropy=False,
        use_huber_loss=use_huber_loss, huber_delta=huber_delta,
        predict_a_exec=predict_a_exec,
    )
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)
    
    return rng, new_critic, new_target_critic_params, critic_info


@functools.partial(jax.jit, static_argnames=(
    'color_jitter', 'num_cameras',
    'query_frequency', 'action_dim',
    'parl_num_samples', 'parl_num_elites', 'parl_num_grad_steps',
    'critic_reduction', 'predict_a_exec',
))
def _update_actor_parl_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: DatasetDict,
    residual_alpha: float,
    color_jitter: bool,
    num_cameras: int,
    query_frequency: int,
    action_dim: int,
    parl_num_samples: int,
    parl_num_elites: int,
    parl_num_grad_steps: int,
    parl_step_size: float,
    critic_reduction: str,
    predict_a_exec: bool = False,
) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
    """JIT-compiled PARL actor update (Best-of-N + Grad-Q + BC distillation)."""
    
    # Data augmentation for pixels
    aug_pixels = batch['observations']['pixels']
    
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
    
    # PARL actor update
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor_residual_parl(
        key, actor, critic, batch,
        residual_alpha, query_frequency, action_dim,
        parl_num_samples=parl_num_samples,
        parl_num_elites=parl_num_elites,
        parl_num_grad_steps=parl_num_grad_steps,
        parl_step_size=parl_step_size,
        critic_reduction=critic_reduction,
        predict_a_exec=predict_a_exec,
    )
    
    return rng, new_actor, actor_info


@functools.partial(jax.jit, static_argnames=(
    'color_jitter', 'num_cameras', 'query_frequency', 'predict_a_exec',
))
def _update_actor_bc_jit(
    rng: PRNGKey,
    actor: TrainState,
    batch: DatasetDict,
    color_jitter: bool,
    num_cameras: int,
    query_frequency: int,
    predict_a_exec: bool = False,
) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
    """JIT-compiled BC warmup actor update."""
    
    # Data augmentation for pixels
    aug_pixels = batch['observations']['pixels']
    
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
    
    # BC warmup actor update
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor_bc_residual(
        key, actor, batch, query_frequency,
        predict_a_exec=predict_a_exec,
    )
    
    return rng, new_actor, actor_info


# ============================================================
# PARL Learner
# ============================================================

class PixelPARLResidualLearner(Agent):
    """Residual PARL Learner for pixel observations.
    
    Policy-Agnostic RL: the actor is trained by distilling Q-optimized
    actions rather than by differentiating through the critic.
    
    Training loop:
    - Critic: standard TD learning (same as SAC/GRPO)
    - Actor: Best-of-N sampling → Q-gradient refinement → MSE distillation
    """

    def __init__(
        self,
        seed: int,
        observations: FrozenDict,
        actions: jnp.ndarray,
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
        discount: float = 0.99,
        critic_reduction: str = 'min',
        # Residual
        residual_alpha: float = 1.0,
        action_magnitude: float = 0.1,
        # Policy std bounds
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        learn_std: bool = True,
        fixed_log_std: float = -0.5,
        # Data augmentation
        color_jitter: bool = True,
        aug_next: bool = True,
        num_cameras: int = 1,
        # PARL-specific parameters
        parl_num_samples: int = 16,
        parl_num_elites: int = 4,
        parl_num_grad_steps: int = 5,
        parl_step_size: float = 0.01,
        # Stability
        max_grad_norm: float = 1.0,
        use_huber_loss: bool = False,
        huber_delta: float = 1.0,
        # Update ratio control
        num_critic_updates: int = 2,
        num_actor_updates: int = 4,
        # Action prediction mode
        predict_a_exec: bool = False,
        # Other
        decay_steps: Optional[int] = None,
        cnn_features: Sequence[int] = (32, 64, 128, 256),
        cnn_strides: Sequence[int] = (2, 2, 2, 2),
        cnn_padding: str = 'VALID',
    ):
        """Initialize Residual PARL learner.
        
        Args:
            parl_num_samples: N — number of action candidates sampled from actor.
            parl_num_elites: K — number of top-Q actions kept for refinement.
            parl_num_grad_steps: Number of gradient ascent steps on Q w.r.t. action.
            parl_step_size: Learning rate for gradient ascent on actions.
        """
        self._residual_alpha = jnp.asarray(residual_alpha, dtype=jnp.float32)
        self.color_jitter = color_jitter
        self.aug_next = aug_next
        self.num_cameras = num_cameras
        
        self.query_frequency = actions.shape[1]
        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]
        self.action_dim_per_step = actions.shape[-1]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.algo = 'residual_parl'
        
        # PARL parameters
        self.parl_num_samples = parl_num_samples
        self.parl_num_elites = parl_num_elites
        self.parl_num_grad_steps = parl_num_grad_steps
        self.parl_step_size = parl_step_size
        
        # Stability
        self.max_grad_norm = max_grad_norm
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        
        # Update ratios
        self.num_critic_updates = num_critic_updates
        self.num_actor_updates = num_actor_updates
        
        # Action prediction mode
        self.predict_a_exec = predict_a_exec
        
        if predict_a_exec:
            print(f'[WARNING] predict_a_exec=True: residual_alpha={residual_alpha} is IGNORED '
                  f'for action composition. Actor predicts a_exec directly.')

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        # ----- Encoder -----
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
            actor_lr_schedule = optax.cosine_decay_schedule(actor_lr, decay_steps)
            critic_lr_schedule = optax.cosine_decay_schedule(critic_lr, decay_steps)
        else:
            actor_lr_schedule = actor_lr
            critic_lr_schedule = critic_lr

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])
        
        # ----- Actor -----
        if learn_std:
            policy_def = LearnedStdTanhNormalPolicy(
                hidden_dims, self.action_dim, 
                dropout_rate=dropout_rate, 
                log_std_min=log_std_min,
                log_std_max=log_std_max,
                low=-action_magnitude, 
                high=action_magnitude
            )
        else:
            policy_def = FixedStdTanhNormalPolicy(
                hidden_dims, self.action_dim,
                dropout_rate=dropout_rate,
                fixed_log_std=fixed_log_std,
                low=-action_magnitude,
                high=action_magnitude,
            )

        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
            pop_base_actions=False
        )
        print(f"PARL Actor: {actor_def}")
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init.get('batch_stats', None)

        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=actor_lr_schedule),
        )
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=actor_optimizer,
            batch_stats=actor_batch_stats,
        )

        # ----- Critic -----
        critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        critic_def = PixelMultiplexer(
            encoder=encoder_def,
            network=critic_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
            pop_base_actions=critic_pop_base_actions
        )
        print(f"PARL Critic: {critic_def}")
        
        actions_flat = actions.reshape(actions.shape[0], -1)
        critic_def_init = critic_def.init(critic_key, observations, actions_flat)

        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init.get('batch_stats', None)
        
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=critic_lr_schedule),
        )
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=critic_optimizer,
            batch_stats=critic_batch_stats
        )
        target_critic_params = copy.deepcopy(critic_params)
        
        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params

        print(f'PARL Residual Learner initialized:')
        print(f'  residual_alpha: {self._residual_alpha}')
        print(f'  query_frequency: {self.query_frequency}')
        print(f'  action_dim: {self.action_dim}')
        print(f'  action_dim_per_step: {self.action_dim_per_step}')
        print(f'  critic_reduction: {self.critic_reduction}')
        print(f'  parl_num_samples (N): {self.parl_num_samples}')
        print(f'  parl_num_elites (K): {self.parl_num_elites}')
        print(f'  parl_num_grad_steps: {self.parl_num_grad_steps}')
        print(f'  parl_step_size: {self.parl_step_size}')
        print(f'  use_huber_loss: {self.use_huber_loss}')
        print(f'  huber_delta: {self.huber_delta}')
        print(f'  max_grad_norm: {self.max_grad_norm}')
        print(f'  num_critic_updates: {self.num_critic_updates}')
        print(f'  num_actor_updates: {self.num_actor_updates}')
        print(f'  predict_a_exec: {self.predict_a_exec}')
        print(f'  learn_std: {learn_std}')

    # ============================================================
    # Update methods
    # ============================================================

    def update_critic(self, batch: FrozenDict) -> Dict[str, float]:
        """Perform a single critic TD update."""
        new_rng, new_critic, new_target_critic, critic_info = _update_critic_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            batch,
            self.discount,
            self.tau,
            self._residual_alpha,
            self.critic_reduction,
            self.color_jitter,
            self.aug_next,
            self.num_cameras,
            self.query_frequency,
            self.use_huber_loss,
            self.huber_delta,
            self.predict_a_exec,
        )
        self._rng = new_rng
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        return critic_info

    def update_actor(self, batch: FrozenDict) -> Dict[str, float]:
        """Perform a single PARL actor update (Best-of-N + Grad-Q + BC distill)."""
        new_rng, new_actor, actor_info = _update_actor_parl_jit(
            self._rng,
            self._actor,
            self._critic,
            batch,
            self._residual_alpha,
            self.color_jitter,
            self.num_cameras,
            self.query_frequency,
            self.action_dim_per_step,
            self.parl_num_samples,
            self.parl_num_elites,
            self.parl_num_grad_steps,
            self.parl_step_size,
            self.critic_reduction,
            self.predict_a_exec,
        )
        self._rng = new_rng
        self._actor = new_actor
        return actor_info

    def update_actor_bc(self, batch: FrozenDict) -> Dict[str, float]:
        """Perform a single BC warmup actor update."""
        new_rng, new_actor, actor_info = _update_actor_bc_jit(
            self._rng,
            self._actor,
            batch,
            self.color_jitter,
            self.num_cameras,
            self.query_frequency,
            self.predict_a_exec,
        )
        self._rng = new_rng
        self._actor = new_actor
        return actor_info

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        """Perform one critic + one actor update (backward compat)."""
        critic_info = self.update_critic(batch)
        actor_info = self.update_actor(batch)
        
        all_info = {**critic_info, **actor_info}
        all_info['residual/alpha'] = float(self._residual_alpha)
        all_info['algo'] = self.algo
        
        return all_info

    # ============================================================
    # Evaluation & checkpointing
    # ============================================================

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
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            q_pred = []

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]
                base_action = observations['base_action'][t]

                obs_dict = {'pixels': obs_pixels[None], 'base_action': base_action[None]}
                for k, v in observations.items():
                    if k not in ['pixels', 'base_action']:
                        obs_dict[k] = v[t][None]

                base_action_squeezed = base_action.squeeze(-1)
                if self.predict_a_exec:
                    a_exec = np.clip(action.squeeze(0), -1.0, 1.0)
                else:
                    a_exec = np.clip(
                        base_action_squeezed[:self.query_frequency] + self._residual_alpha * action.squeeze(0), 
                        -1.0, 1.0
                    )
                a_exec_flat = a_exec.reshape(1, -1)

                q_value = get_value_residual(a_exec_flat, obs_dict, self._critic)
                q_pred.append(q_value)

            traj_images.append(make_visual_residual(q_pred, rewards, masks, observations['pixels']))
        
        print('Finished reward value visuals for PARL.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        return {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'residual_alpha': self._residual_alpha,
            'algo': self.algo,
            'predict_a_exec': self.predict_a_exec,
        }

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        if 'residual_alpha' in output_dict:
            self._residual_alpha = jnp.asarray(output_dict['residual_alpha'], dtype=jnp.float32)
        if 'algo' in output_dict:
            self.algo = output_dict['algo']
        if 'predict_a_exec' in output_dict:
            self.predict_a_exec = bool(output_dict['predict_a_exec'])
        print(f'Restored PARL checkpoint from {dir} (algo: {self.algo}, predict_a_exec: {self.predict_a_exec})')


# ============================================================
# Utility functions
# ============================================================

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
    images = images[..., -1]
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
