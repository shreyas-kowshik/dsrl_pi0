"""Residual SAC Critic Updater.

This module implements the critic update for Residual SAC where:
- The observation contains base_action from a frozen base policy (e.g., Pi-0.5)
- The stored action is the residual (delta) action
- The critic evaluates Q(s, a_exec) where a_exec = clip(base_action + alpha * delta, -1, 1)
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


def update_critic_residual(
        key: PRNGKey, 
        actor: TrainState, 
        critic: TrainState,
        target_critic: TrainState, 
        temp: TrainState, 
        batch: DatasetDict,
        discount: float, 
        residual_alpha: float,
        query_frequency: int,
        backup_entropy: bool = False,
        critic_reduction: str = 'min',) -> Tuple[TrainState, Dict[str, float]]:
    """Update critic for Residual SAC.
    
    Args:
        key: PRNG key for sampling.
        actor: Actor TrainState.
        critic: Critic TrainState.
        target_critic: Target critic TrainState.
        temp: Temperature TrainState.
        batch: Batch of transitions containing:
            - observations['base_action']: (B, chunk_len, action_dim, 1) base actions
            - actions: (B, chunk_len, action_dim) residual/delta actions
            - next_observations['base_action']: (B, chunk_len, action_dim, 1) next base actions
        discount: Discount factor.
        residual_alpha: Scaling factor for residual actions.
        backup_entropy: Whether to include entropy in target Q.
        critic_reduction: How to reduce across ensemble ('min' or 'mean').
        
    Returns:
        Updated critic TrainState and info dict.
    """
    
    # Extract base actions from observations (remove trailing dimension)
    base_action_raw = batch['observations']['base_action']
    next_base_action_raw = batch['next_observations']['base_action']

    # Assert last dim exists and is singleton (so squeeze is safe and meaningful)
    # chex.assert_rank(base_action_raw, 4)
    # chex.assert_rank(next_base_action_raw, 4)
    # chex.assert_equal(base_action_raw.shape[-1], 1)
    # chex.assert_equal(next_base_action_raw.shape[-1], 1)

    base_action = jnp.squeeze(base_action_raw, axis=-1)          # (B, T, A)
    next_base_action = jnp.squeeze(next_base_action_raw, axis=-1)  # (B, T, A)

    B, T, A = base_action.shape
    # chex.assert_shape(next_base_action, (B, T, A))

    
    # Stored actions are delta/residual actions
    delta_action = batch['actions']  # (B, query_frequency, action_dim)
    # chex.assert_shape(delta_action, (B, T, A))
    # chex.assert_tree_all_finite(delta_action)
    
    # Compose executed action for current transition
    a_exec = jnp.clip(base_action[:, :query_frequency, : ] + residual_alpha * delta_action, -1.0, 1.0)
    
    # Flatten action chunks for critic input: (B, query_frequency, action_dim) -> (B, query_frequency * action_dim)
    a_exec_flat = a_exec.reshape(a_exec.shape[0], -1)
    
    # Sample next delta actions from actor
    key, sample_key = jax.random.split(key)
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_delta_actions, next_log_probs = dist.sample_and_log_prob(seed=sample_key)
    
    # # Reshape next delta actions: (B, chunk_len * action_dim) -> (B, chunk_len, action_dim)
    # chex.assert_rank(next_delta_actions, 2)
    # chex.assert_shape(next_delta_actions, (B, T * A))

    # # Log probs should be per-sample scalar (B,)
    # chex.assert_rank(next_log_probs, 1)
    # chex.assert_shape(next_log_probs, (B,))
    next_delta_actions_chunked = next_delta_actions.reshape(B, query_frequency, A)
    
    # Compose next executed action
    next_a_exec = jnp.clip(next_base_action[:, :query_frequency, :] + residual_alpha * next_delta_actions_chunked, -1.0, 1.0)
    next_a_exec_flat = next_a_exec.reshape(next_a_exec.shape[0], -1)
    
    # Compute target Q values
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], next_a_exec_flat)
    # # Expect critic ensemble output shape (N, B)
    # chex.assert_rank(next_qs, 2)
    # chex.assert_equal(next_qs.shape[1], B)

    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplementedError(f"Unknown critic_reduction: {critic_reduction}")

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    if backup_entropy:
        target_q -= batch["discount"] * batch['masks'] * temp.apply_fn(
            {'params': temp.params}) * next_log_probs
    
    target_q = jax.lax.stop_gradient(target_q)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, batch['observations'], a_exec_flat)
        # --- Q(base) vs Q(exec) diagnostics ---
        # base_flat = base_action[:, :query_frequency, :].reshape(base_action.shape[0], -1)

        # qs_base = critic.apply_fn({'params': critic_params}, batch['observations'], base_flat)

        # Reduce ensemble the same way as elsewhere
        if critic_reduction == 'min':
            q_exec = qs.min(axis=0)       # (B,)
            # q_base = qs_base.min(axis=0)  # (B,)
        elif critic_reduction == 'mean':
            q_exec = qs.mean(axis=0)
            # q_base = qs_base.mean(axis=0)

        # q_adv = q_exec - q_base  # (B,) "advantage" of executing residual vs base

        # chex.assert_rank(qs, 2)
        # chex.assert_equal(qs.shape[1], B)
        critic_loss = ((qs - target_q)**2).mean()
        
        # Compute logging metrics
        delta_norm = jnp.linalg.norm(delta_action.reshape(delta_action.shape[0], -1), axis=-1)
        base_action_norm = jnp.linalg.norm(base_action[:, :query_frequency, :].reshape(base_action.shape[0], -1), axis=-1)
        a_exec_norm = jnp.linalg.norm(a_exec_flat, axis=-1)
        
        # Clipping rate: fraction of action dimensions that hit the bounds
        clipping_rate = (jnp.abs(a_exec) >= 1.0).mean()
        
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': qs.mean(),
            'q_std': qs.std(),
            'q_min': qs.min(),
            'q_max': qs.max(),
            'target_actor_entropy': -next_log_probs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'target_q_std': target_q.std(),
            # Residual-specific metrics
            'residual/delta_norm_mean': delta_norm.mean(),
            'residual/delta_norm_std': delta_norm.std(),
            'residual/delta_mean': delta_action.mean(),
            'residual/delta_std': delta_action.std(),
            'residual/delta_min': delta_action.min(),
            'residual/delta_max': delta_action.max(),
            'residual/base_action_norm_mean': base_action_norm.mean(),
            'residual/a_exec_norm_mean': a_exec_norm.mean(),
            'residual/clipping_rate': clipping_rate,
            'residual/a_exec_mean': a_exec.mean(),
            'residual/a_exec_std': a_exec.std(),
            'residual/a_exec_min': a_exec.min(),
            'residual/a_exec_max': a_exec.max(),
            'collapse/q_exec_mean': q_exec.mean(),
            # 'collapse/q_base_mean': q_base.mean(),
            # 'collapse/q_adv_mean': q_adv.mean(),
            # 'collapse/q_adv_std': q_adv.std(),
            # 'collapse/q_adv_pos_frac': (q_adv > 0.0).mean(),
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
