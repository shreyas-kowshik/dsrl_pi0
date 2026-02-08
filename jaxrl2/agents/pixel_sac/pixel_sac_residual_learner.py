"""Residual SAC Learner for Pixel Observations.

This module implements a Residual SAC agent where:
- A frozen base policy (e.g., Pi-0.5) produces base action chunks
- SAC predicts residual actions in environment action space
- Executed actions are: a_exec = clip(base_action + alpha * delta, -1, 1)
- Critic learns Q(s, a_exec)
- Actor optimizes residuals via the executed action

The observation space includes 'base_action' from the frozen policy.
"""

import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import chex
import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.pixel_sac.residual_actor_updater import update_actor_residual, update_actor_bc_residual
from jaxrl2.agents.pixel_sac.residual_critic_updater import update_critic_residual
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature
from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any


@functools.partial(
    jax.jit,
    static_argnames=(
        'critic_reduction', 'color_jitter', 'aug_next', 'num_cameras',
        'backup_entropy', 'query_frequency', 'use_huber_loss', 'predict_a_exec',
    ),
)
def _update_critic_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    temp: TrainState,
    batch: DatasetDict,
    discount: float,
    tau: float,
    residual_alpha: float,
    critic_reduction: str,
    color_jitter: bool,
    aug_next: bool,
    num_cameras: int,
    backup_entropy: bool,
    query_frequency: int,
    use_huber_loss: bool = False,
    huber_delta: float = 1.0,
    predict_a_exec: bool = False,
) -> Tuple[PRNGKey, TrainState, Params, Dict[str, float]]:
    """JIT-compiled critic update for Residual SAC."""
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

    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic_residual(
        key,
        actor,
        critic,
        target_critic,
        temp,
        batch,
        discount,
        residual_alpha,
        query_frequency,
        critic_reduction=critic_reduction,
        backup_entropy=backup_entropy,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta,
        predict_a_exec=predict_a_exec,
    )
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)

    return rng, new_critic, new_target_critic_params, critic_info


@functools.partial(
    jax.jit,
    static_argnames=(
        'critic_reduction', 'color_jitter', 'num_cameras', 'query_frequency',
        'bc_on_success_only', 'bc_flag', 'predict_a_exec',
    ),
)
def _update_actor_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    temp: TrainState,
    batch: DatasetDict,
    residual_alpha: float,
    critic_reduction: str,
    color_jitter: bool,
    num_cameras: int,
    query_frequency: int,
    target_entropy: float,
    bc_flag: bool,
    bc_reg_coeff: float,
    bc_on_success_only: bool,
    predict_a_exec: bool = False,
) -> Tuple[PRNGKey, TrainState, TrainState, Dict[str, float]]:
    """JIT-compiled actor + temperature update for Residual SAC."""
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

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor_residual(
        key,
        actor,
        critic,
        temp,
        batch,
        residual_alpha,
        query_frequency,
        critic_reduction=critic_reduction,
        bc_flag=bc_flag,
        bc_reg_coeff=bc_reg_coeff,
        bc_on_success_only=bc_on_success_only,
        predict_a_exec=predict_a_exec,
    )

    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    return rng, new_actor, new_temp, {**actor_info, **alpha_info}


@functools.partial(
    jax.jit,
    static_argnames=(
        'color_jitter', 'num_cameras', 'query_frequency', 'predict_a_exec',
    ),
)
def _update_actor_bc_jit(
    rng: PRNGKey,
    actor: TrainState,
    batch: DatasetDict,
    color_jitter: bool,
    num_cameras: int,
    query_frequency: int,
    predict_a_exec: bool = False,
) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
    """JIT-compiled BC warmup actor update for SAC."""
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

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor_bc_residual(
        key, actor, batch, query_frequency,
        predict_a_exec=predict_a_exec,
    )

    return rng, new_actor, actor_info


class PixelSACResidualLearner(Agent):
    """Residual SAC Learner for pixel observations with base action conditioning.
    
    This agent learns residual actions on top of a frozen base policy.
    The observation space must include 'base_action' from the frozen policy.
    """

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 residual_alpha: float = 0.1,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 critic_reduction: str = 'mean',
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='group',
                 color_jitter=True,
                 use_spatial_softmax=True,
                 softmax_temperature=1,
                 aug_next=True,
                 use_bottleneck=True,
                 init_temperature: float = 1.0,
                 num_qs: int = 2,
                 target_entropy: float = None,
                 action_magnitude: float = 1.0,
                 num_cameras: int = 1,
                 backup_entropy: bool = False,
                 critic_pop_base_actions: bool = False,
                 clip_temp: bool = True,
                 clip_min_temp: float = 0.01,
                 clip_max_temp: float = 2.0,
                 use_huber_loss: bool = False,
                 huber_delta: float = 1.0,
                 max_grad_norm: float = 1.0,
                 num_critic_updates: int = 1,
                 num_actor_updates: int = 1,
                 bc_reg_coeff: float = 0.0,
                 bc_on_success_only: bool = False,
                 predict_a_exec: bool = False,
                 log_std_min: float = -5.0,
                 log_std_max: float = 2.0,
                 ):
        """Initialize Residual SAC Learner.
        
        Args:
            seed: Random seed.
            observations: Sample observation dict (must include 'base_action').
            actions: Sample action array with shape (batch, query_frequency, action_dim).
            residual_alpha: Scaling factor for residual actions. a_exec = base + alpha * delta.
            actor_lr: Actor learning rate.
            critic_lr: Critic learning rate.
            temp_lr: Temperature learning rate.
            decay_steps: If provided, use cosine decay for actor lr.
            hidden_dims: MLP hidden dimensions.
            cnn_features: CNN feature dimensions (for small encoder).
            cnn_strides: CNN strides (for small encoder).
            cnn_padding: CNN padding (for small encoder).
            latent_dim: Latent dimension for encoder output.
            discount: Discount factor.
            tau: Target network update rate.
            critic_reduction: How to reduce Q ensemble ('min' or 'mean').
            dropout_rate: Dropout rate for policy.
            encoder_type: Type of visual encoder.
            encoder_norm: Normalization for encoder.
            color_jitter: Whether to use color jitter augmentation.
            use_spatial_softmax: Whether to use spatial softmax in encoder.
            softmax_temperature: Temperature for spatial softmax.
            aug_next: Whether to augment next observations.
            use_bottleneck: Whether to use bottleneck in encoder.
            init_temperature: Initial SAC temperature.
            num_qs: Number of Q functions in ensemble.
            target_entropy: Target entropy for temperature tuning.
            action_magnitude: Action bounds for policy output.
            num_cameras: Number of cameras (for multi-view augmentation)
            backup_entropy: Whether to backup entropy in critic updates.
        """
        
        # Validate predict_a_exec configuration
        if predict_a_exec:
            print(f'[WARNING] predict_a_exec=True: residual_alpha={residual_alpha} is IGNORED '
                  f'for action composition. Actor predicts a_exec directly.')
            assert residual_alpha is not None, \
                'residual_alpha must still be provided (used only for logging diagnostics)'
        
        self._residual_alpha = jnp.asarray(residual_alpha, dtype=jnp.float32)
        self.aug_next = aug_next
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras
        self.query_frequency = actions.shape[1]

        # Action dimensions: (query_frequency, action_dim) -> flattened
        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.critic_backup_entropy = backup_entropy
        self.algo = 'residual_sac'
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.max_grad_norm = max_grad_norm
        self.num_critic_updates = num_critic_updates
        self.num_actor_updates = num_actor_updates
        self.bc_reg_coeff = bc_reg_coeff
        self.bc_on_success_only = bc_on_success_only
        self.predict_a_exec = predict_a_exec

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # Select encoder
        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            print('using impala small')
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
            log_std_min=log_std_min,
            log_std_max=log_std_max,
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
        print(f"Residual SAC Actor: {actor_def}")
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=actor_lr),
        )

        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=actor_optimizer,
            batch_stats=actor_batch_stats,
            
        )

        # Critic: takes observations and executed actions (not residuals)
        # Executed actions are flattened to the same dimensionality as residuals, but represent different semantics.
        critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        critic_def = PixelMultiplexer(
            encoder=encoder_def,
            network=critic_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
            pop_base_actions=critic_pop_base_actions # TODO Make this a parameter? Add to Variant?
        )
        print(f"Residual SAC Critic: {critic_def}")
        
        # Initialize critic with executed actions (same shape as residuals after flattening)
        # The actions passed here should be flattened: (batch, query_frequency * action_dim)
        actions_flat = actions.reshape(actions.shape[0], -1)
        critic_def_init = critic_def.init(critic_key, observations, actions_flat)
        self._critic_init_params = critic_def_init['params']

        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=critic_lr),
        )
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=critic_optimizer,
            batch_stats=critic_batch_stats
        )
        target_critic_params = copy.deepcopy(critic_params)
        
        # Temperature
        temp_def = Temperature(init_temperature, clip_temp, clip_min_temp, clip_max_temp)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
            batch_stats=None
        )

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        
        if target_entropy is None or target_entropy == 'auto':
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)
            
        print(f'Residual SAC initialized with:')
        print(f'  residual_alpha: {self._residual_alpha}')
        print(f'  action_dim: {self.action_dim}')
        print(f'  action_chunk_shape: {self.action_chunk_shape}')
        print(f'  target_entropy: {self.target_entropy}')
        print(f'  critic_reduction: {self.critic_reduction}')
        print(f'  use_huber_loss: {self.use_huber_loss}')
        print(f'  huber_delta: {self.huber_delta}')
        print(f'  max_grad_norm: {self.max_grad_norm}')
        print(f'  num_critic_updates: {self.num_critic_updates}')
        print(f'  num_actor_updates: {self.num_actor_updates}')
        print(f'  bc_reg_coeff: {self.bc_reg_coeff}')
        print(f'  bc_on_success_only: {self.bc_on_success_only}')
        print(f'  predict_a_exec: {self.predict_a_exec}')
        print(f'  log_std_min: {log_std_min}')
        print(f'  log_std_max: {log_std_max}')

    def update_critic(self, batch: FrozenDict) -> Dict[str, float]:
        """Perform a single critic update.
        
        Args:
            batch: Batch of transitions with observations containing 'base_action'.
            
        Returns:
            Dictionary of critic training metrics.
        """
        new_rng, new_critic, new_target_critic, critic_info = _update_critic_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._temp,
            batch,
            self.discount,
            self.tau,
            self._residual_alpha,
            self.critic_reduction,
            self.color_jitter,
            self.aug_next,
            self.num_cameras,
            self.critic_backup_entropy,
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
        """Perform a single actor + temperature update.
        
        Args:
            batch: Batch of transitions with observations containing 'base_action'.
            
        Returns:
            Dictionary of actor training metrics.
        """
        new_rng, new_actor, new_temp, actor_info = _update_actor_jit(
            self._rng,
            self._actor,
            self._critic,
            self._temp,
            batch,
            self._residual_alpha,
            self.critic_reduction,
            self.color_jitter,
            self.num_cameras,
            self.query_frequency,
            self.target_entropy,
            bool(self.bc_reg_coeff > 0.0 ),
            self.bc_reg_coeff,
            self.bc_on_success_only,
            self.predict_a_exec,
        )
        self._rng = new_rng
        self._actor = new_actor
        self._temp = new_temp
        return actor_info

    def update_actor_bc(self, batch: FrozenDict) -> Dict[str, float]:
        """Perform a single BC warmup actor update.
        
        Distills base policy actions into the residual policy via MSE.
        Used during BC warmup phase before RL training begins.
        
        Args:
            batch: Batch of transitions with observations containing 'base_action'.
            
        Returns:
            Dictionary of BC actor training metrics.
        """
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
        """Perform one critic update and one actor update (for backward compatibility).
        
        For proper UTD control, use update_critic() and update_actor() separately
        in the training loop with freshly sampled batches.
        
        Args:
            batch: Batch of transitions with observations containing 'base_action'.
            
        Returns:
            Dictionary of training metrics.
        """
        critic_info = self.update_critic(batch)
        actor_info = self.update_actor(batch)

        all_info = {**critic_info, **actor_info}
        all_info['residual/alpha'] = float(self._residual_alpha)
        all_info['algo'] = self.algo

        return all_info

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
                if self.predict_a_exec:
                    # Stored actions ARE a_exec
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
        
        print('Finished reward value visuals for residual SAC.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'temp': self._temp,
            'residual_alpha': self._residual_alpha,
            'algo': self.algo,
            'predict_a_exec': self.predict_a_exec,
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._temp = output_dict['temp']
        if 'residual_alpha' in output_dict:
            self._residual_alpha = jnp.asarray(output_dict['residual_alpha'], dtype=jnp.float32)
        if 'algo' in output_dict:
            self.algo = output_dict['algo']
        if 'predict_a_exec' in output_dict:
            self.predict_a_exec = bool(output_dict['predict_a_exec'])
        print(f'Restored residual SAC checkpoint from {dir} (algo: {self.algo}, predict_a_exec: {self.predict_a_exec})')


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
