"""Residual Actor Updaters for SAC and Q-weighted PG / GRPO.

This module implements actor updates for Residual RL where:
- The observation contains base_action from a frozen base policy (e.g., Pi-0.5)
- The actor outputs residual (delta) actions
- The executed action is a_exec = clip(base_action + alpha * delta, -1, 1)

Update functions:
- update_actor_residual: SAC-style (maximize Q - alpha * log_prob)
- update_actor_residual_ppo: Q-weighted PG / GRPO style (maximize E_g[ Q_g * log pi(a_g | s) ] + entropy bonus)
- update_actor_residual_ppo_onpolicy: On-policy PPO with stored old_log_probs
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


def _soft_clip(x, lo=-1.0, hi=1.0, margin=0.05):
    """Differentiable soft-clip using tanh at boundaries.
    
    Inside [lo+margin, hi-margin] this is identity.
    Outside, it smoothly saturates toward lo/hi via tanh.
    Gradient is always non-zero, unlike jnp.clip.
    """
    mid = (hi + lo) / 2.0
    half_range = (hi - lo) / 2.0
    # Normalize to [-1, 1] range
    x_norm = (x - mid) / half_range
    # Apply tanh-based soft saturation
    return mid + half_range * jnp.tanh(x_norm)


def _safe_clip_for_log_prob(x, eps=1e-6):
    """Clamp actions to (-1+eps, 1-eps) to avoid atanh(±1) = ±inf in log_prob.
    
    This is used when computing log_prob of stored/clipped actions that may
    sit exactly at ±1.0 due to hard clipping during collection.
    """
    return jnp.clip(x, -1.0 + eps, 1.0 - eps)


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
        predict_a_exec: bool = False,
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
        predict_a_exec: If True, actor predicts a_exec directly (not delta).
        
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
        
        # Sample actions from the policy
        actions_sampled, log_probs = dist.sample_and_log_prob(seed=key_act)  # (B, chunk_len * action_dim)
        # chex.assert_rank(actions_sampled, 2)
        # chex.assert_shape(actions_sampled, (B, query_frequency * A))

        # chex.assert_rank(log_probs, 1)
        # chex.assert_shape(log_probs, (B,))
        
        # Reshape to chunk shape: (B, chunk_len * action_dim) -> (B, chunk_len, action_dim)
        actions_chunked = actions_sampled.reshape(B, query_frequency, A)
        
        if predict_a_exec:
            # Actor predicts a_exec directly.
            # TanhNormal output is already in (-1, 1) — no clipping needed.
            # Using jnp.clip or _soft_clip here would either kill gradients
            # or apply a second squashing function (double-tanh).
            a_exec = actions_chunked
            delta_actions_chunked = (a_exec - base_action[:, :query_frequency, :])
        else:
            # Original: actor predicts delta, compose a_exec
            delta_actions_chunked = actions_chunked
            # Use _soft_clip to preserve gradients when base + alpha*delta hits bounds
            a_exec = _soft_clip(base_action[:, :query_frequency, :] + residual_alpha * actions_chunked, -1.0, 1.0)
        
        a_exec_flat = a_exec.reshape(a_exec.shape[0], -1)  # (B, chunk_len * action_dim)
        delta_actions = delta_actions_chunked.reshape(B, query_frequency * A)  # for logging
        
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
        
        # Actor loss: maximize Q + alpha * entropy.
        # We use the FULL TanhNormal log_prob (from sample_and_log_prob) which
        # includes the Jacobian correction -log(1-tanh^2(z)). This term is
        # critical: it provides gradient to the mean that pushes it away from
        # the tanh saturation boundaries. Without it, the mean is free to grow
        # unboundedly, causing all actions to saturate at ±1.
        #
        # The Jacobian correction is computed from the pre-tanh sample z (not
        # via atanh), so it's numerically stable. We clip the final log_prob
        # to [-50, 50] as a safety net for extreme cases.
        alpha_val = temp.apply_fn({'params': temp.params})
        log_probs_clipped = jnp.clip(log_probs, -50.0, 50.0)
        base_entropy = dist.distribution.entropy()  # (B,) — for diagnostics
        rl_loss = (-q + alpha_val * log_probs_clipped).mean()
        
        # BC regularization loss (stored-action NLL)
        bc_loss_val = jnp.array(0.0)
        bc_info = {}
        if bc_flag:
            bc_loss_val, bc_info = compute_bc_loss_residual(
                dist, batch, query_frequency, bc_on_success_only, key=key_act
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
        # When predict_a_exec, delta IS the actual exec change, no scaling needed
        eff_delta_norm = jnp.where(
            predict_a_exec, delta_norm, jnp.abs(residual_alpha) * delta_norm
        )  # (B,)

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
            'entropy': (-log_probs_clipped).mean(),  # TanhNormal entropy — used by temperature auto-tuning
            'base_gaussian_entropy': base_entropy.mean(),  # Gaussian entropy — diagnostic only
            'tanh_log_prob_mean': log_probs.mean(),  # raw (unclipped) for diagnostics
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
            'collapse/alpha_entropy_mean': (alpha_val * (-log_probs_clipped)).mean(),
            'actor/rl_loss': rl_loss,
            'bc/reg_coeff': bc_reg_coeff,
            'bc/weighted_loss': bc_reg_coeff * bc_loss_val,
            **bc_info,
        }
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    # NaN detection logging (no masking — let it crash so we can diagnose)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    has_nan_grad = jnp.any(jnp.array([jnp.any(jnp.isnan(g)) for g in grad_leaves]))
    has_inf_grad = jnp.any(jnp.array([jnp.any(jnp.isinf(g)) for g in grad_leaves]))
    info['debug/actor_grad_has_nan'] = has_nan_grad.astype(jnp.float32)
    info['debug/actor_grad_has_inf'] = has_inf_grad.astype(jnp.float32)
    info['debug/actor_loss_is_nan'] = jnp.isnan(info['actor_loss']).astype(jnp.float32)
    
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
        key: PRNGKey = None,
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Compute BC regularization loss (MSE between policy sample and stored actions).
    
    Uses sample-based MSE instead of mode-based MSE to provide gradient signal
    to both mean AND std via the reparameterization trick. Using mode() would
    make the loss independent of std, causing std to collapse to zero.
    
    Uses MSE instead of NLL to avoid numerical issues with atanh(±1) = ±∞
    when stored actions are near the action bounds.
    
    This function is meant to be called INSIDE actor_loss_fn, using the `dist`
    already constructed from the candidate actor_params being differentiated.
    
    Args:
        dist: The current policy distribution (from actor forward pass on batch obs).
        batch: Batch of transitions containing:
            - actions: (B, query_frequency, action_dim) stored delta actions
            - success_flag: (B,) binary flag (1.0 = success episode, 0.0 = failure)
        query_frequency: Chunk length.
        bc_on_success_only: If True, only compute BC loss on success transitions.
        key: PRNG key for sampling. Required for reparameterized sample.
        
    Returns:
        bc_loss: Scalar BC loss (MSE, possibly masked).
        info: Dict with BC diagnostics.
    """
    # Stored delta actions: (B, query_frequency, action_dim) -> flatten to (B, query_freq * action_dim)
    stored_actions = batch['actions']  # (B, query_frequency, action_dim)
    B = stored_actions.shape[0]
    stored_actions_flat = stored_actions.reshape(B, -1)  # (B, query_freq * action_dim)
    
    # Sample from policy (reparameterized) for gradient to both mean and std.
    # Using mode() would only give gradient to mean, causing std collapse.
    policy_sample = dist.sample(seed=key)  # (B, action_dim_flat)
    # Also compute mode for diagnostics only (not in loss)
    policy_mode = dist.mode()  # (B, action_dim_flat)
    
    # Per-sample MSE: mean over action dimensions, keep batch dim
    mse_per_sample = jnp.mean((policy_sample - stored_actions_flat) ** 2, axis=-1)  # (B,)
    
    if bc_on_success_only:
        # Mask: only include transitions from successful episodes
        success_mask = batch['success_flag']  # (B,)
        num_success = jnp.sum(success_mask) + 1e-8  # avoid div by zero
        bc_loss = jnp.sum(mse_per_sample * success_mask) / num_success
        bc_frac = jnp.mean(success_mask)
    else:
        bc_loss = jnp.mean(mse_per_sample)
        bc_frac = 1.0
    
    info = {
        'bc/loss': bc_loss,
        'bc/mse_mean': jnp.mean(mse_per_sample),
        'bc/mse_max': jnp.max(mse_per_sample),
        'bc/mse_min': jnp.min(mse_per_sample),
        'bc/sample_mean': jnp.mean(policy_sample),
        'bc/sample_std': jnp.std(policy_sample),
        'bc/mode_mean': jnp.mean(policy_mode),
        'bc/mode_std': jnp.std(policy_mode),
        'bc/stored_action_mean': jnp.mean(stored_actions_flat),
        'bc/stored_action_std': jnp.std(stored_actions_flat),
        'bc/success_frac_in_batch': bc_frac,
    }
    
    return bc_loss, info


def _nan_to_num_tree(tree):
    """Apply nan_to_num to all leaves in a pytree.
    
    Safety net only — this should ideally never be triggered.
    Uses posinf=0, neginf=0 to avoid injecting large values that
    cause secondary parameter explosions.
    """
    return jax.tree_util.tree_map(
        lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 
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
        normalize_advantages: bool = False,
        adv_clip_min: Optional[float] = None,
        adv_clip_max: Optional[float] = None,
        log_ratio_clip: float = 20.0,
        log_prob_clip: float = 50.0,
        bc_flag: bool = False,
        bc_reg_coeff: float = 0.0,
        bc_on_success_only: bool = False,
        predict_a_exec: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:
    """Update actor for Residual Q-weighted PG / GRPO.
    
    Clean advantage-weighted regression (NOT PPO):
        L = -E_g[ A_g * log pi(a_g | s) ] - entropy_coeff * H[pi]
    
    Since actions are sampled fresh from the current policy each call
    (no environment interaction between old/new), importance ratios are
    meaningless.  We directly weight log-probs by stop-gradient advantages.
    
    Args:
        key: PRNG key.
        actor: Current actor TrainState.
        target_critic: Target critic TrainState (for Q values).
        batch: Batch of transitions.
        residual_alpha: Scaling factor for residual actions.
        query_frequency: Query frequency (chunk length).
        action_dim: Action dimension per step.
        grpo_num_samples: G, number of samples per state.
        clip_epsilon: Unused (kept for caller compatibility).
        clip_min_epsilon_multiplier: Unused (kept for caller compatibility).
        clip_max_epsilon_multiplier: Unused (kept for caller compatibility).
        entropy_coeff: Entropy bonus coefficient (default 1e-3 for stability).
        advantage_critic_reduction: 'min' or 'mean' for Q ensemble reduction.
        use_grpo_baseline: If True, use Q - mean(Q) (GRPO). If False, use raw Q.
        adv_clip_min: Optional lower bound for advantage clipping.
        adv_clip_max: Optional upper bound for advantage clipping.
        log_ratio_clip: Unused (kept for caller compatibility).
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
        delta = frozen_dist.sample(seed=k)
        return delta
    
    delta_actions = jax.vmap(sample_one)(sample_keys)  # (G, B, action_dim_flat)
    
    # Stop gradient — these are proposal actions from the frozen policy
    delta_actions = jax.lax.stop_gradient(delta_actions)
    
    chex.assert_shape(delta_actions, (G, B, action_dim_flat))
    
    # Step 2: Compute Q values for all samples
    # Note: these Q values are used as stop_gradient'd advantages, so jnp.clip is fine here.
    # The clip here is for valid critic evaluation, not for gradient flow.
    def compute_q_for_sample(action_flat):
        action_chunked = action_flat.reshape(B, query_frequency, action_dim)
        if predict_a_exec:
            a_exec = jnp.clip(action_chunked, -1.0, 1.0)
        else:
            a_exec = jnp.clip(base_action[:, :query_frequency, :] + residual_alpha * action_chunked, -1.0, 1.0)
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
    
    # Optional advantage normalization
    if normalize_advantages:
        adv_std = jnp.std(advantages) + 1e-8
        adv_mean = jnp.mean(advantages)
        advantages = (advantages - adv_mean) / adv_std
    
    # Optional advantage clipping
    if adv_clip_min is not None or adv_clip_max is not None:
        advantages = jnp.clip(advantages, adv_clip_min, adv_clip_max)
    
    # Stop gradient on advantages (they are from frozen critic, should not backprop)
    advantages = jax.lax.stop_gradient(advantages)
    
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
        
        # Compute current log probs for all G proposal actions
        # Clamp actions to (-1+eps, 1-eps) before log_prob to avoid atanh(±1)=±inf
        def compute_lp(delta_flat):
            safe_delta = _safe_clip_for_log_prob(delta_flat)
            lp = dist.log_prob(safe_delta)
            return jnp.clip(lp, -log_prob_clip, log_prob_clip)
        
        log_probs = jax.vmap(compute_lp)(delta_actions)  # (G, B)
        
        # Flatten for advantage-weighted loss
        log_probs_flat = log_probs.reshape(G * B)       # (G*B,)
        advantages_flat = advantages.reshape(G * B)      # (G*B,)
        
        # Q-weighted policy gradient: L = -E[ A * log pi(a|s) ]
        # No importance ratios — actions are from current policy (stop-gradient'd).
        pg_loss = -(advantages_flat * log_probs_flat).mean()
        
        # Entropy bonus: average over G samples for lower variance
        # (we already have G forward passes, so this is nearly free)
        entropy_per_sample = -log_probs  # (G, B) — TanhNormal entropy estimate
        mean_entropy = entropy_per_sample.mean()
        entropy_loss = -entropy_coeff * mean_entropy
        
        # BC regularization loss
        bc_loss_val = jnp.array(0.0)
        bc_info = {}
        if bc_flag:
            bc_loss_val, bc_info = compute_bc_loss_residual(
                dist, batch, query_frequency, bc_on_success_only, key=sample_key
            )
        
        actor_loss = pg_loss + entropy_loss + bc_reg_coeff * bc_loss_val
        
        # Logging - get distribution parameters
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        log_std_dist = jnp.log(std_diag_dist + 1e-8)
        
        # Sample diagnostics from first sample
        delta_sample = delta_actions[0]
        delta_chunked = delta_sample.reshape(B, query_frequency, action_dim)
        if predict_a_exec:
            a_exec = jnp.clip(delta_chunked, -1.0, 1.0)
            delta_for_logging = a_exec - base_action[:, :query_frequency, :]
            delta_norm = jnp.linalg.norm(delta_for_logging.reshape(B, -1), axis=-1)
        else:
            a_exec = jnp.clip(base_action[:, :query_frequency, :] + residual_alpha * delta_chunked, -1.0, 1.0)
            delta_norm = jnp.linalg.norm(delta_sample, axis=-1)
        base_flat = base_action[:, :query_frequency, :].reshape(B, query_frequency * action_dim)
        base_norm = jnp.linalg.norm(base_flat, axis=-1)
        # When predict_a_exec, delta IS the actual exec change, no scaling needed
        eff_delta_norm = jnp.where(
            predict_a_exec, delta_norm, jnp.abs(residual_alpha) * delta_norm
        )
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


def update_actor_residual_ppo_onpolicy(
        key: PRNGKey,
        actor: TrainState,
        target_critic: TrainState,
        batch: DatasetDict,
        residual_alpha: float,
        query_frequency: int,
        action_dim: int,
        clip_epsilon: float = 0.2,
        clip_min_epsilon_multiplier: float = 1.0,
        clip_max_epsilon_multiplier: float = 1.0,
        entropy_coeff: float = 1e-3,
        advantage_critic_reduction: str = 'mean',
        normalize_advantages: bool = False,
        log_ratio_clip: float = 20.0,
        log_prob_clip: float = 50.0,
        bc_flag: bool = False,
        bc_reg_coeff: float = 0.0,
        bc_on_success_only: bool = False,
        predict_a_exec: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:
    """On-policy PPO actor update using stored old_log_probs from the replay buffer.
    
    Bandit-style: uses raw Q values as advantages (no GRPO baseline subtraction).
    The batch is sampled from the most recently collected trajectory, which means
    the actions in the batch were taken by the behavior policy whose log_probs are
    stored as batch['old_log_probs'].
    
    This gives proper importance weight ratios: pi_current(a|s) / pi_old(a|s).
    
    Args:
        key: PRNG key.
        actor: Current actor TrainState.
        target_critic: Target critic TrainState (for Q values).
        batch: Batch sampled from last trajectory, must contain:
            - observations['base_action']: base actions
            - actions: delta actions (residual) that were actually taken
            - old_log_probs: log prob of those actions under the behavior policy
        residual_alpha: Scaling factor for residual actions.
        query_frequency: Query frequency (chunk length).
        action_dim: Action dimension per step.
        clip_epsilon: PPO clip epsilon.
        entropy_coeff: Entropy bonus coefficient.
        advantage_critic_reduction: 'min' or 'mean' for Q ensemble reduction.
        normalize_advantages: Whether to normalize advantages.
        log_ratio_clip: Clamp log_ratio to [-clip, clip] before exp.
        log_prob_clip: Clamp log_probs to [-clip, clip].
        
    Returns:
        Updated actor TrainState and info dict.
    """
    # Extract base actions from observations
    base_action_raw = batch['observations']['base_action']
    base_action = jnp.squeeze(base_action_raw, axis=-1)  # (B, T, A)
    B, T, A = base_action.shape
    action_dim_flat = query_frequency * action_dim
    
    # The stored actions that were actually executed
    stored_actions = batch['actions']  # (B, query_frequency, action_dim)
    stored_actions_flat = stored_actions.reshape(B, action_dim_flat)  # (B, action_dim_flat)
    
    # Stored old log probs from behavior policy
    old_log_probs = batch['old_log_probs']  # (B,)
    old_log_probs = jnp.clip(old_log_probs, -log_prob_clip, log_prob_clip)
    
    # Compute Q values for the stored actions (the actions that were actually taken)
    stored_actions_chunked = stored_actions.reshape(B, query_frequency, action_dim)
    if predict_a_exec:
        # Stored actions ARE a_exec directly
        a_exec = jnp.clip(stored_actions_chunked, -1.0, 1.0)
    else:
        # Original: stored actions are delta, compose a_exec
        a_exec = jnp.clip(base_action[:, :query_frequency, :] + residual_alpha * stored_actions_chunked, -1.0, 1.0)
    a_exec_flat = a_exec.reshape(B, action_dim_flat)
    
    qs = target_critic.apply_fn({'params': target_critic.params}, batch['observations'], a_exec_flat)
    if advantage_critic_reduction == 'min':
        q_values = qs.min(axis=0)  # (B,)
    else:
        q_values = qs.mean(axis=0)  # (B,)
    
    # Bandit-style: use raw Q as advantage (no baseline subtraction)
    advantages = q_values  # (B,)
    
    # Optional advantage normalization
    if normalize_advantages:
        adv_std = jnp.std(advantages) + 1e-8
        adv_mean = jnp.mean(advantages)
        advantages = (advantages - adv_mean) / adv_std
    
    # Stop gradient on advantages and old_log_probs
    advantages = jax.lax.stop_gradient(advantages)
    old_log_probs = jax.lax.stop_gradient(old_log_probs)
    
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
        
        # Current log prob of the stored actions.
        # Clamp to (-1+eps, 1-eps) before log_prob to avoid atanh(±1)=±inf.
        # This is critical when predict_a_exec=True because stored actions
        # are hard-clipped a_exec values that can sit exactly at ±1.0.
        safe_stored = _safe_clip_for_log_prob(stored_actions_flat)
        log_probs = dist.log_prob(safe_stored)  # (B,)
        log_probs = jnp.clip(log_probs, -log_prob_clip, log_prob_clip)
        
        # PPO clipped loss with proper importance weights
        log_ratio = log_probs - old_log_probs
        log_ratio_clamped = jnp.clip(log_ratio, -log_ratio_clip, log_ratio_clip)
        ratio = jnp.exp(log_ratio_clamped)
        
        lower_bound = 1.0 - clip_epsilon * clip_min_epsilon_multiplier
        upper_bound = 1.0 + clip_epsilon * clip_max_epsilon_multiplier
        clipped_ratio = jnp.clip(ratio, lower_bound, upper_bound)
        
        pg_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
        
        # Entropy bonus: use TanhNormal log_prob (includes Jacobian correction)
        # to provide gradient that prevents the mean from saturating at boundaries.
        _, entropy_log_probs = dist.sample_and_log_prob(seed=key)
        entropy_log_probs = jnp.clip(entropy_log_probs, -50.0, 50.0)
        mean_entropy = (-entropy_log_probs).mean()  # TanhNormal entropy estimate
        entropy_loss = -entropy_coeff * mean_entropy
        
        # BC regularization loss
        bc_loss_val = jnp.array(0.0)
        bc_info = {}
        if bc_flag:
            bc_loss_val, bc_info = compute_bc_loss_residual(
                dist, batch, query_frequency, bc_on_success_only, key=key
            )
        
        actor_loss = pg_loss + entropy_loss + bc_reg_coeff * bc_loss_val
        
        # Logging
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        log_std_dist = jnp.log(std_diag_dist + 1e-8)
        
        approx_kl = ((ratio - 1) - log_ratio_clamped).mean()
        ratio_clipped_upper = jnp.mean(ratio > upper_bound)
        ratio_clipped_lower = jnp.mean(ratio < lower_bound)
        log_ratio_clipped_frac = jnp.mean(jnp.abs(log_ratio) > log_ratio_clip)
        
        # Residual diagnostics
        if predict_a_exec:
            delta_derived = a_exec - base_action[:, :query_frequency, :]
            delta_norm = jnp.linalg.norm(delta_derived.reshape(B, -1), axis=-1)
        else:
            delta_norm = jnp.linalg.norm(stored_actions_flat, axis=-1)
        base_flat = base_action[:, :query_frequency, :].reshape(B, action_dim_flat)
        base_norm = jnp.linalg.norm(base_flat, axis=-1)
        # When predict_a_exec, delta IS the actual exec change, no scaling needed
        eff_delta_norm = jnp.where(
            predict_a_exec, delta_norm, jnp.abs(residual_alpha) * delta_norm
        )
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
            # On-policy PPO mode indicator
            'ppo/on_policy': 1.0,
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


def update_actor_residual_parl(
        key: PRNGKey,
        actor: TrainState,
        critic: TrainState,
        batch: DatasetDict,
        residual_alpha: float,
        query_frequency: int,
        action_dim: int,
        parl_num_samples: int = 16,
        parl_num_elites: int = 4,
        parl_num_grad_steps: int = 5,
        parl_step_size: float = 0.01,
        critic_reduction: str = 'min',
        predict_a_exec: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:
    """Update actor via Policy-Agnostic RL (PARL) for residual setting.
    
    PARL decouples actor training from policy gradient entirely:
    1. Sample N action candidates from the current actor pi_theta(s),
       plus the base policy action as an additional (N+1)-th candidate
    2. Evaluate all N+1 with critic Q(s, a_exec), keep top-K elites
    3. Refine elites via gradient ascent: a <- a + eta * grad_a Q(s, a)
    4. Pick the best refined action as the distillation target a*
    5. Train the actor to imitate a* via MSE loss (BC distillation)
    
    Including the base policy action ensures the residual never degrades
    below the base policy's quality — if the base action has highest Q,
    it will be selected as the distillation target.
    
    The actor never differentiates through the critic — the critic only
    provides stop-gradient targets. This avoids issues with tanh squashing,
    action clipping killing gradients, etc.
    
    Args:
        key: PRNG key.
        actor: Current actor TrainState.
        critic: Critic TrainState (used for Q-evaluation and gradient ascent).
        batch: Batch of transitions.
        residual_alpha: Scaling factor for residual actions.
        query_frequency: Chunk length for action queries.
        action_dim: Action dimension per step.
        parl_num_samples: N, number of action candidates to sample from actor.
        parl_num_elites: K, number of top-Q actions to keep for refinement.
        parl_num_grad_steps: Number of gradient ascent steps on Q w.r.t. action.
        parl_step_size: Step size (learning rate) for gradient ascent on actions.
        critic_reduction: How to reduce Q ensemble ('min' or 'mean').
        predict_a_exec: If True, actor predicts a_exec directly (not delta).
        
    Returns:
        Updated actor TrainState and info dict.
    """
    key, sample_key, select_key = jax.random.split(key, 3)
    
    # Extract base actions from observations
    base_action_raw = batch['observations']['base_action']
    base_action = jnp.squeeze(base_action_raw, axis=-1)  # (B, T, A)
    B, T, A = base_action.shape
    action_dim_flat = query_frequency * action_dim
    N = parl_num_samples
    K = parl_num_elites
    
    # ----------------------------------------------------------------
    # Step 1: Sample N action candidates from current actor (frozen)
    # ----------------------------------------------------------------
    frozen_dist = actor.apply_fn(
        {'params': jax.lax.stop_gradient(actor.params)},
        batch['observations']
    )
    
    sample_keys = jax.random.split(sample_key, N)
    
    def sample_one(k):
        return frozen_dist.sample(seed=k)  # (B, action_dim_flat)
    
    # (N, B, action_dim_flat)
    delta_candidates = jax.vmap(sample_one)(sample_keys)
    delta_candidates = jax.lax.stop_gradient(delta_candidates)
    
    # ----------------------------------------------------------------
    # Step 2: Evaluate Q for all candidates → keep top-K elites
    # ----------------------------------------------------------------
    # Include the base policy action as an additional candidate.
    # The base action IS already a valid a_exec (no residual needed).
    base_a_exec_flat = jnp.clip(
        base_action[:, :query_frequency, :], -1.0, 1.0
    ).reshape(B, action_dim_flat)  # (B, action_dim_flat)
    
    def delta_to_a_exec_flat(delta_flat):
        """Convert actor output (delta or a_exec) to executed action for critic."""
        delta_chunked = delta_flat.reshape(B, query_frequency, action_dim)
        if predict_a_exec:
            a_exec = jnp.clip(delta_chunked, -1.0, 1.0)
        else:
            a_exec = jnp.clip(
                base_action[:, :query_frequency, :] + residual_alpha * delta_chunked,
                -1.0, 1.0
            )
        return a_exec.reshape(B, action_dim_flat)
    
    def compute_q(a_exec_flat):
        """Evaluate critic on a_exec_flat (B, action_dim_flat) → (B,)."""
        qs = critic.apply_fn(
            {'params': critic.params},
            batch['observations'], a_exec_flat
        )  # (num_qs, B)
        if critic_reduction == 'min':
            return qs.min(axis=0)
        else:
            return qs.mean(axis=0)
    
    # Evaluate all N actor candidates: (N, B, action_dim_flat) → (N, B)
    a_exec_actor = jax.vmap(delta_to_a_exec_flat)(delta_candidates)  # (N, B, action_dim_flat)
    q_actor = jax.vmap(compute_q)(a_exec_actor)  # (N, B)
    
    # Append base policy action as the (N+1)-th candidate
    # a_exec_candidates: (N+1, B, action_dim_flat), q_candidates: (N+1, B)
    a_exec_candidates = jnp.concatenate(
        [a_exec_actor, base_a_exec_flat[None, :, :]], axis=0
    )  # (N+1, B, action_dim_flat)
    q_base = compute_q(base_a_exec_flat)  # (B,)
    q_candidates = jnp.concatenate(
        [q_actor, q_base[None, :]], axis=0
    )  # (N+1, B)
    
    # Select top-K per batch element from N+1 candidates
    # Transpose to (B, N+1) for per-batch-element sorting
    q_candidates_T = q_candidates.T  # (B, N+1)
    a_exec_candidates_T = jnp.transpose(a_exec_candidates, (1, 0, 2))  # (B, N+1, action_dim_flat)
    
    top_k_indices = jnp.argsort(q_candidates_T, axis=-1)[:, -K:]  # (B, K)
    
    # Gather elite actions: (B, K, action_dim_flat)
    elite_actions = jnp.take_along_axis(
        a_exec_candidates_T,
        top_k_indices[:, :, None],
        axis=1,
    )  # (B, K, action_dim_flat)
    
    # Q values before refinement (for logging)
    elite_q_before = jnp.take_along_axis(q_candidates_T, top_k_indices, axis=1)  # (B, K)
    
    # Track how often the base policy action (index N) is among elites
    base_in_elites = (top_k_indices == N).any(axis=-1).mean()  # fraction of batch
    
    # ----------------------------------------------------------------
    # Step 3: Gradient ascent on Q w.r.t. a_exec for each elite
    # ----------------------------------------------------------------
    # We work in a_exec space (flattened). The gradient ascent is:
    #   a_exec <- clip(a_exec + step_size * grad_a Q(s, a_exec), -1, 1)
    
    def q_sum_fn(a_exec_flat, observations):
        """Scalar Q sum for gradient computation."""
        qs = critic.apply_fn(
            {'params': critic.params},
            observations, a_exec_flat
        )
        if critic_reduction == 'min':
            return qs.min(axis=0).sum()
        else:
            return qs.mean(axis=0).sum()
    
    grad_q_fn = jax.grad(q_sum_fn, argnums=0)
    
    def refine_one_elite(a_exec_flat_BK):
        """Run gradient ascent on a single elite action set (B, action_dim_flat)."""
        def body_fn(_, a):
            grad = grad_q_fn(a, batch['observations'])
            a = a + parl_step_size * grad
            a = jnp.clip(a, -1.0, 1.0)
            return a
        
        return jax.lax.fori_loop(0, parl_num_grad_steps, body_fn, a_exec_flat_BK)
    
    # Reshape elites for vmap over K: (K, B, action_dim_flat)
    elite_actions_KBD = jnp.transpose(elite_actions, (1, 0, 2))
    
    # Refine all K elites in parallel
    refined_elites_KBD = jax.vmap(refine_one_elite)(elite_actions_KBD)  # (K, B, action_dim_flat)
    
    # ----------------------------------------------------------------
    # Step 4: Pick the best refined action as distillation target
    # ----------------------------------------------------------------
    refined_q = jax.vmap(compute_q)(refined_elites_KBD)  # (K, B)
    refined_q_T = refined_q.T  # (B, K)
    refined_elites_BKD = jnp.transpose(refined_elites_KBD, (1, 0, 2))  # (B, K, action_dim_flat)
    
    best_idx = jnp.argmax(refined_q_T, axis=-1)  # (B,)
    best_a_exec = jnp.take_along_axis(
        refined_elites_BKD,
        best_idx[:, None, None],
        axis=1,
    ).squeeze(axis=1)  # (B, action_dim_flat)
    
    best_q = refined_q_T[jnp.arange(B), best_idx]  # (B,)
    
    # Convert best a_exec back to delta (the actor's output space) for distillation
    best_a_exec_chunked = best_a_exec.reshape(B, query_frequency, action_dim)
    if predict_a_exec:
        # Actor predicts a_exec directly; target IS a_exec (clipped to tanh range)
        bc_target = jnp.clip(best_a_exec_chunked, -1.0 + 1e-6, 1.0 - 1e-6).reshape(B, action_dim_flat)
    else:
        # Actor predicts delta; back-derive: delta = (a_exec - base) / alpha
        bc_target_delta = (best_a_exec_chunked - base_action[:, :query_frequency, :]) / (residual_alpha + 1e-8)
        # Clip to actor output range (TanhNormal outputs in (-1, 1))
        bc_target = jnp.clip(bc_target_delta, -1.0 + 1e-6, 1.0 - 1e-6).reshape(B, action_dim_flat)
    
    # Stop gradient on target
    bc_target = jax.lax.stop_gradient(bc_target)
    
    # ----------------------------------------------------------------
    # Step 5: Distill into actor via MSE loss
    # ----------------------------------------------------------------
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Dict[str, float], Dict]]:
        dist = actor.apply_fn({'params': actor_params}, batch['observations'])
        new_model_state = {}
        
        # Sample-based MSE for gradient to both mean and std
        sampled_actions = dist.sample(seed=key)  # (B, action_dim_flat)
        mse_loss = jnp.mean((sampled_actions - bc_target) ** 2)
        
        # Also compute mode-based MSE for logging
        policy_mode = dist.mode()
        mode_mse = jnp.mean((policy_mode - bc_target) ** 2)
        
        # Distribution diagnostics
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        log_std_dist = jnp.log(std_diag_dist + 1e-8)
        
        # Entropy (for monitoring, not in loss)
        base_entropy = dist.distribution.entropy()
        mean_entropy = base_entropy.mean()
        
        # Compute a_exec from sampled actions for diagnostics
        sampled_chunked = sampled_actions.reshape(B, query_frequency, action_dim)
        if predict_a_exec:
            a_exec_diag = jnp.clip(sampled_chunked, -1.0, 1.0)
            delta_diag = a_exec_diag - base_action[:, :query_frequency, :]
        else:
            delta_diag = sampled_chunked
            a_exec_diag = jnp.clip(
                base_action[:, :query_frequency, :] + residual_alpha * sampled_chunked,
                -1.0, 1.0
            )
        
        delta_norm = jnp.linalg.norm(delta_diag.reshape(B, -1), axis=-1)
        base_flat = base_action[:, :query_frequency, :].reshape(B, -1)
        base_norm = jnp.linalg.norm(base_flat, axis=-1)
        eff_delta_norm = jnp.where(predict_a_exec, delta_norm, jnp.abs(residual_alpha) * delta_norm)
        clipping_rate = (jnp.abs(a_exec_diag) >= 1.0).mean()
        
        # Target diagnostics
        bc_target_chunked = bc_target.reshape(B, query_frequency, action_dim)
        target_norm = jnp.linalg.norm(bc_target, axis=-1)
        
        info = {
            'actor_loss': mse_loss,
            'parl/mse_loss': mse_loss,
            'parl/mode_mse': mode_mse,
            'parl/q_before_refinement': elite_q_before.mean(),
            'parl/q_after_refinement': best_q.mean(),
            'parl/q_improvement': (best_q - elite_q_before.max(axis=-1)).mean(),
            'parl/target_norm_mean': target_norm.mean(),
            'parl/target_mean': bc_target.mean(),
            'parl/target_std': jnp.std(bc_target),
            'parl/num_samples': float(N),
            'parl/num_elites': float(K),
            'parl/num_grad_steps': float(parl_num_grad_steps),
            'parl/step_size': parl_step_size,
            'parl/base_in_elites_frac': base_in_elites,
            'parl/base_q': q_base.mean(),
            'entropy': mean_entropy,
            'mean_pi_norm': jnp.linalg.norm(mean_dist, axis=-1).mean(),
            'std_pi_norm': jnp.linalg.norm(std_diag_dist, axis=-1).mean(),
            'mean_pi_avg': mean_dist.mean(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_min': std_diag_dist.min(),
            'std_pi_max': std_diag_dist.max(),
            'log_std_mean': log_std_dist.mean(),
            'log_std_min': log_std_dist.min(),
            'log_std_max': log_std_dist.max(),
            'actor/delta_norm_mean': delta_norm.mean(),
            'actor/clipping_rate': clipping_rate,
            'actor/effective_residual_norm': eff_delta_norm.mean(),
            'collapse/ratio_eff_delta_to_base_mean': (eff_delta_norm / (base_norm + 1e-8)).mean(),
        }
        
        return mse_loss, (info, new_model_state)
    
    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    # NaN guard
    grads = _nan_to_num_tree(grads)
    
    new_actor = actor.apply_gradients(grads=grads)
    
    return new_actor, info


def update_actor_bc_residual(
        key: PRNGKey,
        actor: TrainState,
        batch: DatasetDict,
        query_frequency: int,
        predict_a_exec: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:
    """BC warmup actor update: distill base policy actions into residual policy.
    
    During BC warmup, the actor is trained to reproduce the base policy's
    behavior via supervised learning (MSE on policy mode):
    
    - If predict_a_exec=False (delta mode): target = zeros
      (actor should predict zero residual so a_exec = a_base)
    - If predict_a_exec=True (a_exec mode): target = a_base
      (actor should predict the base policy action directly)
    
    This gives the actor a warm initialization near the identity mapping
    before RL training begins, ensuring a smooth transition.
    
    Args:
        key: PRNG key.
        actor: Actor TrainState.
        batch: Batch of transitions containing:
            - observations['base_action']: (B, chunk_len, action_dim, 1) base actions
        query_frequency: Chunk length for action queries.
        predict_a_exec: If True, BC target is a_base; if False, BC target is zeros.
        
    Returns:
        Updated actor TrainState and info dict with BC diagnostics.
    """
    # Extract base actions from observations
    base_action_raw = batch['observations']['base_action']
    base_action = jnp.squeeze(base_action_raw, axis=-1)  # (B, T, A)
    B, T, A = base_action.shape
    
    # Construct BC target
    if predict_a_exec:
        # Actor should predict a_base directly.
        # IMPORTANT: clip to [-1, 1] because Pi-0.5 outputs can exceed this range
        # (quantile unnormalization is unbounded), but the actor's output is
        # tanh-squashed and bounded in (-1, 1). Without clipping, the MSE target
        # would be unreachable, pushing actor mean toward ±∞ and causing NaN.
        # This matches the RL update which also clips: a_exec = clip(actor_output, -1, 1).
        bc_target = jnp.clip(
            base_action[:, :query_frequency, :], -1.0, 1.0
        ).reshape(B, query_frequency * A)
    else:
        # Actor should predict zero residual (delta = 0)
        bc_target = jnp.zeros((B, query_frequency * A))
    
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Dict[str, float], Dict]]:
        # Forward pass through actor
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
        
        # Sample-based MSE: trains both mean and std via reparameterization trick
        # Avoids atanh boundary issues from log_prob while providing gradient to std
        sampled_actions = dist.sample(seed=key)  # (B, action_dim_flat)
        
        # MSE loss between sampled actions and BC target
        mse_per_sample = jnp.mean((sampled_actions - bc_target) ** 2, axis=-1)  # (B,)
        bc_loss = jnp.mean(mse_per_sample)
        
        # Also compute mode for diagnostics (not used in loss)
        policy_mode = dist.mode()  # (B, action_dim_flat)
        
        # Distribution diagnostics
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        log_std_dist = jnp.log(std_diag_dist + 1e-8)
        
        # Entropy from base distribution
        base_entropy = dist.distribution.entropy()  # (B,)
        mean_entropy = base_entropy.mean()
        
        # Compute a_exec from sampled actions for diagnostics
        sampled_actions_chunked = sampled_actions.reshape(B, query_frequency, A)
        if predict_a_exec:
            a_exec = jnp.clip(sampled_actions_chunked, -1.0, 1.0)
            delta_from_base = a_exec - base_action[:, :query_frequency, :]
        else:
            a_exec = jnp.clip(
                base_action[:, :query_frequency, :] + sampled_actions_chunked, -1.0, 1.0
            )
            delta_from_base = sampled_actions_chunked
        
        delta_norm = jnp.linalg.norm(delta_from_base.reshape(B, -1), axis=-1)
        base_flat = base_action[:, :query_frequency, :].reshape(B, -1)
        base_norm = jnp.linalg.norm(base_flat, axis=-1)
        a_exec_norm = jnp.linalg.norm(a_exec.reshape(B, -1), axis=-1)
        clipping_rate = (jnp.abs(a_exec) >= 1.0).mean()
        bc_target_norm = jnp.linalg.norm(bc_target, axis=-1)
        # Fraction of raw base_action values outside [-1, 1] (before clipping)
        base_out_of_range = (jnp.abs(base_flat) > 1.0).mean()
        
        info = {
            'actor_loss': bc_loss,
            'bc_warmup/mse_loss': bc_loss,
            'bc_warmup/mse_per_sample_mean': mse_per_sample.mean(),
            'bc_warmup/mse_per_sample_max': mse_per_sample.max(),
            'bc_warmup/mse_per_sample_min': mse_per_sample.min(),
            'bc_warmup/sampled_action_mean': sampled_actions.mean(),
            'bc_warmup/sampled_action_std': jnp.std(sampled_actions),
            'bc_warmup/policy_mode_mean': policy_mode.mean(),
            'bc_warmup/policy_mode_std': jnp.std(policy_mode),
            'bc_warmup/target_mean': bc_target.mean(),
            'bc_warmup/target_std': jnp.std(bc_target),
            'bc_warmup/delta_norm_mean': delta_norm.mean(),
            'bc_warmup/base_norm_mean': base_norm.mean(),
            'bc_warmup/base_out_of_range_frac': base_out_of_range,
            'bc_warmup/clipping_rate': clipping_rate,
            'bc_warmup/is_warmup': 1.0,
            # Standard actor diagnostics (for continuity in wandb)
            'entropy': mean_entropy,
            'mean_pi_norm': jnp.linalg.norm(mean_dist, axis=-1).mean(),
            'std_pi_norm': jnp.linalg.norm(std_diag_dist, axis=-1).mean(),
            'mean_pi_avg': mean_dist.mean(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_min': std_diag_dist.min(),
            'std_pi_max': std_diag_dist.max(),
            'log_std_mean': log_std_dist.mean(),
            'log_std_min': log_std_dist.min(),
            'log_std_max': log_std_dist.max(),
            'actor/delta_norm_mean': delta_norm.mean(),
            'actor/clipping_rate': clipping_rate,
            'actor/predict_a_exec': predict_a_exec, 
            'actor/a_exec_norm_mean': a_exec_norm.mean(), 
            'actor/bc_target_norm_mean': bc_target_norm.mean()
        }
        
        return bc_loss, (info, new_model_state)
    
    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    # NaN guard
    grads = _nan_to_num_tree(grads)
    
    if 'batch_stats' in new_model_state and new_model_state.get('batch_stats'):
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)
    
    return new_actor, info