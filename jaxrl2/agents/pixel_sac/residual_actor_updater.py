"""Residual Actor Updaters for SAC and Q-weighted PG / GRPO.

This module implements actor updates for Residual RL where:
- The observation contains base_action from a frozen base policy (e.g., Pi-0.5)
- The actor outputs residual (delta) actions
- The executed action is a_exec = clip(base_action + alpha * delta, -1, 1)

Two update functions:
- update_actor_residual: SAC-style (maximize Q - alpha * log_prob)
- update_actor_residual_ppo: Q-weighted PG / GRPO with PPO clipping
"""

from typing import Dict, Tuple, Optional
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
        critic_reduction: str = 'min',
        bc_flag: bool = False,
        bc_reg_coeff: float = 0.0,
        bc_on_success_only: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:
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
        rl_loss = (alpha_val * log_probs - q).mean()
        
        # BC regularization loss (stored-action NLL)
        bc_loss_val = jnp.array(0.0)
        bc_info = {}
        if bc_flag:
            bc_loss_val, bc_info = compute_bc_loss_residual(
                dist, batch, query_frequency, bc_on_success_only
            )
        
        actor_loss = rl_loss + bc_reg_coeff * bc_loss_val
        

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
            'actor/rl_loss': rl_loss,
            'bc/reg_coeff': bc_reg_coeff,
            'bc/weighted_loss': bc_reg_coeff * bc_loss_val,
            **bc_info,
        }
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info


def compute_bc_loss_residual(
        dist,
        batch: DatasetDict,
        query_frequency: int,
        bc_on_success_only: bool = False,
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Compute BC regularization loss (negative log-likelihood of stored actions).
    
    Encourages the actor to reproduce the stored delta (residual) actions from
    the replay buffer, which are the actions that led to the observed transitions.
    When bc_on_success_only=True, the loss is computed only on transitions from
    successful episodes (using the success_flag in the batch).
    
    This function is meant to be called INSIDE actor_loss_fn, using the `dist`
    already constructed from the candidate actor_params being differentiated.
    
    Args:
        dist: The current policy distribution (from actor forward pass on batch obs).
        batch: Batch of transitions containing:
            - actions: (B, query_frequency, action_dim) stored delta actions
            - success_flag: (B,) binary flag (1.0 = success episode, 0.0 = failure)
        query_frequency: Chunk length.
        bc_on_success_only: If True, only compute BC loss on success transitions.
        
    Returns:
        bc_loss: Scalar BC loss (mean NLL, possibly masked).
        info: Dict with BC diagnostics.
    """
    # Stored delta actions: (B, query_frequency, action_dim) -> flatten to (B, query_freq * action_dim)
    stored_actions = batch['actions']  # (B, query_frequency, action_dim)
    B = stored_actions.shape[0]
    stored_actions_flat = stored_actions.reshape(B, -1)  # (B, query_freq * action_dim)
    
    # NLL of stored actions under current policy
    log_probs = dist.log_prob(stored_actions_flat)  # (B,)
    nll = -log_probs  # (B,)
    
    if bc_on_success_only:
        # Mask: only include transitions from successful episodes
        success_mask = batch['success_flag']  # (B,)
        num_success = jnp.sum(success_mask) + 1e-8  # avoid div by zero
        bc_loss = jnp.sum(nll * success_mask) / num_success
        bc_frac = jnp.mean(success_mask)
    else:
        bc_loss = jnp.mean(nll)
        bc_frac = 1.0
    
    info = {
        'bc/loss': bc_loss,
        'bc/nll_mean': jnp.mean(nll),
        'bc/nll_max': jnp.max(nll),
        'bc/nll_min': jnp.min(nll),
        'bc/log_prob_mean': jnp.mean(log_probs),
        'bc/success_frac_in_batch': bc_frac,
    }
    
    return bc_loss, info


def _nan_to_num_tree(tree):
    """Apply nan_to_num to all leaves in a pytree."""
    return jax.tree_util.tree_map(
        lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6), 
        tree
    )


def update_actor_residual_ppo(
        key: PRNGKey,
        actor: TrainState,
        target_critic: TrainState,
        batch: DatasetDict,
        residual_alpha: float,
        query_frequency: int,
        action_dim: int,
        grpo_num_samples: int = 8,
        clip_epsilon: float = 0.2,
        clip_min_epsilon_multiplier: float = 1.0,
        clip_max_epsilon_multiplier: float = 1.0,
        entropy_coeff: float = 1e-3,
        advantage_critic_reduction: str = 'mean',
        use_grpo_baseline: bool = True,
        adv_clip_min: Optional[float] = None,
        adv_clip_max: Optional[float] = None,
        log_ratio_clip: float = 20.0,
        log_prob_clip: float = 50.0,
        bc_flag: bool = False,
        bc_reg_coeff: float = 0.0,
        bc_on_success_only: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:
    """Update actor for Residual Q-weighted PG / GRPO.
    
    Uses stop_gradient on current actor for old_log_probs (no target actor).
    Includes numerical stability fixes: log_ratio clamping, log_prob clamping,
    and NaN guards on gradients.
    
    Args:
        key: PRNG key.
        actor: Current actor TrainState.
        target_critic: Target critic TrainState (for Q values).
        batch: Batch of transitions.
        residual_alpha: Scaling factor for residual actions.
        query_frequency: Query frequency (chunk length).
        action_dim: Action dimension per step.
        grpo_num_samples: G, number of samples per state.
        clip_epsilon: PPO clip epsilon.
        clip_min_epsilon_multiplier: Multiplier for lower clip bound.
        clip_max_epsilon_multiplier: Multiplier for upper clip bound.
        entropy_coeff: Entropy bonus coefficient (default 1e-3 for stability).
        advantage_critic_reduction: 'min' or 'mean' for Q ensemble reduction.
        use_grpo_baseline: If True, use Q - mean(Q) (GRPO). If False, use raw Q.
        adv_clip_min: Optional lower bound for advantage clipping.
        adv_clip_max: Optional upper bound for advantage clipping.
        log_ratio_clip: Clamp log_ratio to [-clip, clip] before exp (default 20.0).
        log_prob_clip: Clamp log_probs to [-clip, clip] (default 50.0).
        
    Returns:
        Updated actor TrainState and info dict.
    """
    key, sample_key = jax.random.split(key)
    
    # Extract base actions from observations
    base_action_raw = batch['observations']['base_action']
    base_action = jnp.squeeze(base_action_raw, axis=-1)  # (B, T, A)
    B, T, A = base_action.shape
    action_dim_flat = query_frequency * action_dim
    G = grpo_num_samples
    
    chex.assert_shape(base_action, (B, T, A))
    chex.assert_equal(action_dim_flat, query_frequency * action_dim)
    
    # Step 1: Sample G delta actions from current actor with stop_gradient
    # (No target actor - use current actor frozen for "old" policy)
    if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
        frozen_dist = actor.apply_fn(
            {'params': jax.lax.stop_gradient(actor.params), 
             'batch_stats': actor.batch_stats},
            batch['observations']
        )
        if isinstance(frozen_dist, tuple):
            frozen_dist = frozen_dist[0]
    else:
        frozen_dist = actor.apply_fn(
            {'params': jax.lax.stop_gradient(actor.params)}, 
            batch['observations']
        )
    
    sample_keys = jax.random.split(sample_key, G)
    
    def sample_one(k):
        delta, lp = frozen_dist.sample_and_log_prob(seed=k)
        # Clamp log_probs for numerical stability
        lp = jnp.clip(lp, -log_prob_clip, log_prob_clip)
        return delta, lp
    
    delta_actions, old_log_probs = jax.vmap(sample_one)(sample_keys)  # (G, B, action_dim_flat), (G, B)
    
    # Stop gradient on old_log_probs (they are from frozen policy)
    old_log_probs = jax.lax.stop_gradient(old_log_probs)
    delta_actions = jax.lax.stop_gradient(delta_actions)
    
    chex.assert_shape(delta_actions, (G, B, action_dim_flat))
    chex.assert_shape(old_log_probs, (G, B))
    
    # Step 2: Compute Q values for all samples
    def compute_q_for_sample(delta_flat):
        delta_chunked = delta_flat.reshape(B, query_frequency, action_dim)
        a_exec = jnp.clip(base_action[:, :query_frequency, :] + residual_alpha * delta_chunked, -1.0, 1.0)
        a_exec_flat = a_exec.reshape(B, query_frequency * action_dim)
        qs = target_critic.apply_fn({'params': target_critic.params}, batch['observations'], a_exec_flat)
        if advantage_critic_reduction == 'min':
            return qs.min(axis=0)
        else:
            return qs.mean(axis=0)
    
    q_values = jax.vmap(compute_q_for_sample)(delta_actions)  # (G, B)
    
    chex.assert_shape(q_values, (G, B))
    
    # Step 3: Compute advantages
    if use_grpo_baseline:
        # GRPO: subtract group mean
        q_mean = q_values.mean(axis=0, keepdims=True)  # (1, B)
        advantages = q_values - q_mean  # (G, B)
    else:
        # Q-weighted PG: use raw Q
        advantages = q_values  # (G, B)
    
    chex.assert_shape(advantages, (G, B))
    
    # Optional advantage clipping
    if adv_clip_min is not None or adv_clip_max is not None:
        advantages = jnp.clip(advantages, adv_clip_min, adv_clip_max)
    
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Dict[str, float], Dict]]:
        # Get current distribution
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn(
                {'params': actor_params, 'batch_stats': actor.batch_stats},
                batch['observations'],
                mutable=['batch_stats']
            )
            if isinstance(dist, tuple):
                dist = dist[0]
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'])
            new_model_state = {}
        
        # Compute current log probs for all G samples
        def compute_lp(delta_flat):
            lp = dist.log_prob(delta_flat)
            # Clamp log_probs for numerical stability
            return jnp.clip(lp, -log_prob_clip, log_prob_clip)
        
        log_probs = jax.vmap(compute_lp)(delta_actions)  # (G, B)
        
        # Flatten for PPO loss
        log_probs_flat = log_probs.reshape(G * B)
        old_log_probs_flat = old_log_probs.reshape(G * B)
        advantages_flat = advantages.reshape(G * B)
        
        # PPO clipped loss with log_ratio clamping for numerical stability
        log_ratio = log_probs_flat - old_log_probs_flat
        # Clamp log_ratio before exp to prevent Inf
        log_ratio_clamped = jnp.clip(log_ratio, -log_ratio_clip, log_ratio_clip)
        ratio = jnp.exp(log_ratio_clamped)
        
        lower_bound = 1.0 - clip_epsilon * clip_min_epsilon_multiplier
        upper_bound = 1.0 + clip_epsilon * clip_max_epsilon_multiplier
        clipped_ratio = jnp.clip(ratio, lower_bound, upper_bound)
        
        pg_loss = -jnp.mean(jnp.minimum(ratio * advantages_flat, clipped_ratio * advantages_flat))
        
        # Entropy from base distribution
        base_entropy = dist.distribution.entropy()  # (B,)
        mean_entropy = base_entropy.mean()
        entropy_loss = -entropy_coeff * mean_entropy
        
        # BC regularization loss (stored-action NLL)
        bc_loss_val = jnp.array(0.0)
        bc_info = {}
        if bc_flag:
            bc_loss_val, bc_info = compute_bc_loss_residual(
                dist, batch, query_frequency, bc_on_success_only
            )
        
        actor_loss = pg_loss + entropy_loss + bc_reg_coeff * bc_loss_val
        
        # Logging - get distribution parameters
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        log_std_dist = jnp.log(std_diag_dist + 1e-8)
        
        # PPO stats
        approx_kl = ((ratio - 1) - log_ratio_clamped).mean()
        ratio_clipped_upper = jnp.mean(ratio > upper_bound)
        ratio_clipped_lower = jnp.mean(ratio < lower_bound)
        log_ratio_clipped_frac = jnp.mean(jnp.abs(log_ratio) > log_ratio_clip)
        
        # Sample diagnostics from first sample
        delta_sample = delta_actions[0]
        delta_chunked = delta_sample.reshape(B, query_frequency, action_dim)
        a_exec = jnp.clip(base_action[:, :query_frequency, :] + residual_alpha * delta_chunked, -1.0, 1.0)
        
        delta_norm = jnp.linalg.norm(delta_sample, axis=-1)
        base_flat = base_action[:, :query_frequency, :].reshape(B, query_frequency * action_dim)
        base_norm = jnp.linalg.norm(base_flat, axis=-1)
        eff_delta_norm = jnp.abs(residual_alpha) * delta_norm
        clipping_rate = (jnp.abs(a_exec) >= 1.0).mean()
        
        info = {
            'actor_loss': actor_loss,
            'pg_loss': pg_loss,
            'entropy_loss': entropy_loss,
            'entropy': mean_entropy,
            'q_pi_in_actor': q_values.mean(),
            'q_pi_std': q_values.std(),
            'advantages_mean': advantages.mean(),
            'advantages_std': advantages.std(),
            'log_probs_mean': log_probs.mean(),
            'log_probs_min': log_probs.min(),
            'log_probs_max': log_probs.max(),
            'old_log_probs_mean': old_log_probs.mean(),
            'mean_pi_norm': jnp.linalg.norm(mean_dist, axis=-1).mean(),
            'std_pi_norm': jnp.linalg.norm(std_diag_dist, axis=-1).mean(),
            'mean_pi_avg': mean_dist.mean(),
            # Extended std logging for stability monitoring
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_min': std_diag_dist.min(),
            'std_pi_max': std_diag_dist.max(),
            'log_std_mean': log_std_dist.mean(),
            'log_std_min': log_std_dist.min(),
            'log_std_max': log_std_dist.max(),
            # PPO stats
            'ppo/ratio_mean': ratio.mean(),
            'ppo/ratio_std': ratio.std(),
            'ppo/ratio_min': ratio.min(),
            'ppo/ratio_max': ratio.max(),
            'ppo/approx_kl': approx_kl,
            'ppo/ratio_clipped_upper': ratio_clipped_upper,
            'ppo/ratio_clipped_lower': ratio_clipped_lower,
            'ppo/log_ratio_mean': log_ratio.mean(),
            'ppo/log_ratio_abs_max': jnp.abs(log_ratio).max(),
            'ppo/log_ratio_clipped_frac': log_ratio_clipped_frac,
            # Residual diagnostics
            'actor/delta_norm_mean': delta_norm.mean(),
            'actor/clipping_rate': clipping_rate,
            'actor/effective_residual_norm': eff_delta_norm.mean(),
            'collapse/ratio_eff_delta_to_base_mean': (eff_delta_norm / (base_norm + 1e-8)).mean(),
            # GRPO stats
            'grpo/num_samples': float(G),
            'grpo/q_group_std': jnp.std(q_values, axis=0).mean(),
            # BC regularization
            'actor/rl_loss': pg_loss + entropy_loss,
            'bc/reg_coeff': bc_reg_coeff,
            'bc/weighted_loss': bc_reg_coeff * bc_loss_val,
            **bc_info,
        }
        
        return actor_loss, (info, new_model_state)
    
    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    # NaN guard: replace NaN/Inf in gradients with zeros
    grads = _nan_to_num_tree(grads)
    
    if 'batch_stats' in new_model_state and new_model_state.get('batch_stats'):
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)
    
    return new_actor, info
