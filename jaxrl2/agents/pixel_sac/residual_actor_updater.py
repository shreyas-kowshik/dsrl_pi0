"""Residual SAC Actor Updater.

This module implements the actor update for Residual SAC where:
- The observation contains base_action from a frozen base policy (e.g., Pi-0.5)
- The actor outputs residual (delta) actions
- The executed action is a_exec = clip(base_action + alpha * delta, -1, 1)
- Actor is optimized to maximize Q(s, a_exec) - alpha_temp * log_prob(delta)
"""

from typing import Dict, Tuple
import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

def _flat(x):
    return x.reshape(x.shape[0], -1)

def _cosine_sim(a, b, eps=1e-8):
    a = _flat(a)
    b = _flat(b)
    an = jnp.linalg.norm(a, axis=-1)
    bn = jnp.linalg.norm(b, axis=-1)
    return (jnp.sum(a * b, axis=-1) / (an * bn + eps))


def update_actor_residual(
        key: PRNGKey, 
        actor: TrainState, 
        critic: TrainState,
        temp: TrainState, 
        batch: DatasetDict, 
        residual_alpha: float,
        query_frequency: int,
        cross_norm: bool = False, 
        critic_reduction: str = 'min') -> Tuple[TrainState, Dict[str, float]]:
    """Update actor for Residual SAC.
    
    Args:
        key: PRNG key for sampling.
        actor: Actor TrainState.
        critic: Critic TrainState.
        temp: Temperature TrainState.
        batch: Batch of transitions containing:
            - observations['base_action']: (B, chunk_len, action_dim, 1) base actions
        residual_alpha: Scaling factor for residual actions.
        cross_norm: Whether to use cross normalization (for batch norm).
        critic_reduction: How to reduce across ensemble ('min' or 'mean').
        
    Returns:
        Updated actor TrainState and info dict.
    """
    
    key, key_act = jax.random.split(key, num=2)
    
    # Extract base actions from observations (remove trailing dimension)
    base_action_raw = batch['observations']['base_action']

    # chex.assert_rank(base_action_raw, 4)
    # chex.assert_equal(base_action_raw.shape[-1], 1)

    base_action = jnp.squeeze(base_action_raw, axis=-1)  # (B, T, A)
    B, T, A = base_action.shape

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Dict[str, float], Dict]]:
        # Forward pass through actor
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn(
                {'params': actor_params, 'batch_stats': actor.batch_stats}, 
                batch['observations'], 
                mutable=['batch_stats']
            )
            # if cross_norm:
            #     next_dist = actor.apply_fn(
            #         {'params': actor_params, 'batch_stats': actor.batch_stats}, 
            #         batch['next_observations'], 
            #         mutable=['batch_stats']
            #     )
            # else:
            #     next_dist = actor.apply_fn(
            #         {'params': actor_params, 'batch_stats': actor.batch_stats}, 
            #         batch['next_observations']
            #     )
            # if type(next_dist) == tuple:
            #     next_dist, new_model_state = next_dist
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'])
            # next_dist = actor.apply_fn({'params': actor_params}, batch['next_observations'])
            new_model_state = {}
        
        # For logging: get distribution parameters
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)
        
        # Sample delta actions from the policy
        delta_actions, log_probs = dist.sample_and_log_prob(seed=key_act)  # (B, chunk_len * action_dim)
        # chex.assert_rank(delta_actions, 2)
        # chex.assert_shape(delta_actions, (B, query_frequency * A))

        # chex.assert_rank(log_probs, 1)
        # chex.assert_shape(log_probs, (B,))
        
        # Reshape delta actions to chunk shape: (B, chunk_len * action_dim) -> (B, chunk_len, action_dim)
        delta_actions_chunked = delta_actions.reshape(B, query_frequency, A)
        
        # Compose executed action
        a_exec = jnp.clip(base_action[:, :query_frequency, :] + residual_alpha * delta_actions_chunked, -1.0, 1.0)
        a_exec_flat = a_exec.reshape(a_exec.shape[0], -1)  # (B, chunk_len * action_dim)
        
        # Evaluate Q on composed action
        if hasattr(critic, 'batch_stats') and critic.batch_stats is not None:
            qs, _ = critic.apply_fn(
                {'params': critic.params, 'batch_stats': critic.batch_stats}, 
                batch['observations'],
                a_exec_flat, 
                mutable=['batch_stats']
            )
        else:    
            qs = critic.apply_fn({'params': critic.params}, batch['observations'], a_exec_flat)
        
        # chex.assert_rank(qs, 2)     # (N, B)
        # chex.assert_equal(qs.shape[1], B)

        if critic_reduction == 'min':
            q = qs.min(axis=0)
        elif critic_reduction == 'mean':
            q = qs.mean(axis=0)
        else:
            raise ValueError(f"Invalid critic reduction: {critic_reduction}")
        
        # Actor loss: maximize Q, minimize entropy cost
        alpha_val = temp.apply_fn({'params': temp.params})
        actor_loss = (alpha_val * log_probs - q).mean()
        

        # Compute residual-specific metrics
        delta_norm = jnp.linalg.norm(delta_actions, axis=-1)
        
        # Clipping rate for actor's proposed actions
        clipping_rate = (jnp.abs(a_exec) >= 1.0).mean()

        # --- Collapse / dominance diagnostics ---
        delta_flat = delta_actions  # (B, query_frequency*A)
        base_flat = base_action[:, :query_frequency, :].reshape(base_action.shape[0], -1)  # (B, query_frequency*A)
        a_exec_flat = a_exec.reshape(a_exec.shape[0], -1)          # (B, query_frequency*A)

        delta_norm = jnp.linalg.norm(delta_flat, axis=-1)          # (B,)
        base_norm = jnp.linalg.norm(base_flat, axis=-1)            # (B,)
        exec_norm  = jnp.linalg.norm(a_exec_flat, axis=-1)         # (B,)

        # "Effective" residual magnitude after scaling
        eff_delta_norm = jnp.abs(residual_alpha) * delta_norm      # (B,)

        # Ratio tells you collapse (<~0.05) vs dominance (>~1)
        ratio_eff_delta_to_base = eff_delta_norm / (base_norm + 1e-8)

        # How much did execution actually change relative to base?
        delta_exec_flat = a_exec_flat - base_flat
        delta_exec_norm = jnp.linalg.norm(delta_exec_flat, axis=-1)  # (B,)
        ratio_exec_change_to_base = delta_exec_norm / (base_norm + 1e-8)

        # Fraction of near-zero residual dims (collapse indicator)
        near_zero_frac = (jnp.abs(delta_flat) < 1e-3).mean()

        # Action clipping saturation (dominance / constraint binding)
        clip_frac = (jnp.abs(a_exec) >= 0.999).mean()

        # Geometry: are residuals aligned with base or fighting it?
        cos_base_delta = _cosine_sim(base_action[:, :query_frequency, :], delta_actions_chunked)  # (B,)
        cos_base_exec_change = _cosine_sim(base_action[:, :query_frequency, :], (a_exec - base_action[:, :query_frequency, :]))  # (B,)

        things_to_log = {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'q_pi_in_actor': q.mean(),
            'mean_pi_norm': mean_dist_norm.mean(),
            'std_pi_norm': std_dist_norm.mean(),
            'mean_pi_avg': mean_dist.mean(),
            'mean_pi_max': mean_dist.max(),
            'mean_pi_min': mean_dist.min(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_max': std_diag_dist.max(),
            'std_pi_min': std_diag_dist.min(),
            # Residual-specific metrics from actor
            'actor/delta_norm_mean': delta_norm.mean(),
            'actor/delta_norm_std': delta_norm.std(),
            'actor/delta_mean': delta_actions.mean(),
            'actor/delta_std': delta_actions.std(),
            'actor/a_exec_mean': a_exec.mean(),
            'actor/a_exec_std': a_exec.std(),
            'actor/clipping_rate': clipping_rate,
            'actor/effective_residual_norm': (residual_alpha * delta_norm).mean(),
            'collapse/delta_norm_mean': delta_norm.mean(),
            'collapse/delta_norm_std': delta_norm.std(),
            'collapse/base_norm_mean': base_norm.mean(),
            'collapse/exec_norm_mean': exec_norm.mean(),
            'collapse/eff_delta_norm_mean': eff_delta_norm.mean(),
            'collapse/ratio_eff_delta_to_base_mean': ratio_eff_delta_to_base.mean(),
            'collapse/ratio_exec_change_to_base_mean': ratio_exec_change_to_base.mean(),
            'collapse/near_zero_frac': near_zero_frac,
            'collapse/clip_frac': clip_frac,
            'collapse/cos_base_delta_mean': cos_base_delta.mean(),
            'collapse/cos_base_delta_std': cos_base_delta.std(),
            'collapse/cos_base_exec_change_mean': cos_base_exec_change.mean(),
            'collapse/alpha_temp': alpha_val,
            'collapse/alpha_logp_mean': (alpha_val * log_probs).mean(),

        }
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
