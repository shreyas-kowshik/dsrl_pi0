"""JIT-compiled JAX functions for diagnostic computations.

Provides Q-value evaluation, gradient computation, V_soft estimation,
and gradient ascent utilities used by all diagnostic plot generators.
"""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Q-value evaluation
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames='critic_apply_fn')
def compute_q_values_all_heads(critic_params, critic_apply_fn, obs_dict, action_flat):
    """Evaluate critic ensemble, returning per-head Q values.

    Args:
        critic_params: Critic parameters.
        critic_apply_fn: Critic apply function.
        obs_dict: Observation dict (batch dim 1).
        action_flat: (B, query_freq * action_dim) flattened executed action.

    Returns:
        (num_qs, B) Q values from each ensemble head.
    """
    return critic_apply_fn({'params': critic_params}, obs_dict, action_flat)


@functools.partial(jax.jit, static_argnames=('critic_apply_fn', 'reduction'))
def compute_q_reduced(critic_params, critic_apply_fn, obs_dict, action_flat,
                      reduction='mean'):
    """Evaluate critic ensemble with reduction across heads.

    Returns:
        (B,) reduced Q values.
    """
    qs = critic_apply_fn({'params': critic_params}, obs_dict, action_flat)
    if reduction == 'min':
        return qs.min(axis=0)
    return qs.mean(axis=0)


# ---------------------------------------------------------------------------
# Q gradient w.r.t. action
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=('critic_apply_fn', 'reduction'))
def compute_q_action_gradient(critic_params, critic_apply_fn, obs_dict,
                              action_flat, reduction='mean'):
    """Compute gradient of reduced Q w.r.t. action.

    Args:
        action_flat: (B, action_dim_flat) — must be a JAX array.

    Returns:
        (B, action_dim_flat) gradient.
    """
    def q_sum_fn(a):
        qs = critic_apply_fn({'params': critic_params}, obs_dict, a)
        if reduction == 'min':
            return qs.min(axis=0).sum()
        return qs.mean(axis=0).sum()

    return jax.grad(q_sum_fn)(action_flat)


@functools.partial(jax.jit, static_argnames=('critic_apply_fn', 'num_qs'))
def compute_q_action_gradient_per_head(critic_params, critic_apply_fn,
                                       obs_dict, action_flat, num_qs=2):
    """Compute per-head gradient norms of Q w.r.t. action.

    Args:
        action_flat: (1, action_dim_flat) single observation.
        num_qs: Number of critic heads.

    Returns:
        grad_norms: (num_qs,) L2 norm of gradient per head.
        grads: (num_qs, action_dim_flat) gradients per head.
    """
    def q_head_fn(a, head_idx):
        qs = critic_apply_fn({'params': critic_params}, obs_dict, a)
        return qs[head_idx, 0]

    def grad_for_head(head_idx):
        return jax.grad(lambda a: q_head_fn(a, head_idx))(action_flat)

    grads = jax.vmap(grad_for_head)(jnp.arange(num_qs))  # (num_qs, 1, action_dim_flat)
    grads = grads.squeeze(1)  # (num_qs, action_dim_flat)
    grad_norms = jnp.linalg.norm(grads, axis=-1)  # (num_qs,)
    return grad_norms, grads


# ---------------------------------------------------------------------------
# TD-error computation (1-step, per-head)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=(
    'actor_apply_fn', 'critic_apply_fn', 'temp_apply_fn',
    'query_frequency', 'predict_a_exec',
))
def compute_td_error_single_step(
    rng,
    actor_apply_fn, actor_params, actor_batch_stats,
    critic_apply_fn, critic_params,
    target_critic_params,
    temp_apply_fn, temp_params,
    obs_t, a_exec_flat_t,
    reward_t, obs_tp1, mask_t, discount,
    residual_alpha, query_frequency,
    predict_a_exec=False,
):
    """Compute per-head TD errors for a single transition.

    Args:
        obs_t, obs_tp1: Observation dicts at t and t+1 (batch dim 1).
        a_exec_flat_t: (1, action_dim_flat) executed action at t.
        reward_t: scalar reward.
        mask_t: 1.0 if not terminal, 0.0 if terminal.
        discount: gamma^query_freq discount factor.

    Returns:
        td_errors: (num_qs,) per-head TD errors.
        q_pred_heads: (num_qs,) per-head Q predictions.
        target_q: scalar target Q value.
    """
    # Sample next action from actor at s_{t+1}
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist_tp1 = actor_apply_fn(input_collections, obs_tp1, training=False, mutable=False)
    rng, key = jax.random.split(rng)
    next_action_sampled, next_log_prob = dist_tp1.sample_and_log_prob(seed=key)

    # Compose next executed action
    next_base_raw = obs_tp1['base_action']
    next_base = jnp.squeeze(next_base_raw, axis=-1)  # (1, chunk_len, action_dim)
    next_action_chunked = next_action_sampled.reshape(1, query_frequency, -1)

    if predict_a_exec:
        next_a_exec = jnp.clip(next_action_chunked, -1.0, 1.0)
    else:
        next_a_exec = jnp.clip(
            next_base[:, :query_frequency, :] + residual_alpha * next_action_chunked,
            -1.0, 1.0
        )
    next_a_exec_flat = next_a_exec.reshape(1, -1)

    # Target Q
    next_qs = critic_apply_fn({'params': target_critic_params}, obs_tp1, next_a_exec_flat)
    next_q_min = next_qs.min(axis=0)  # (1,)

    # Temperature
    alpha_temp = temp_apply_fn({'params': temp_params})

    target_q = reward_t + discount * mask_t * (next_q_min[0] - alpha_temp * next_log_prob[0])

    # Online Q predictions
    q_pred_heads = critic_apply_fn({'params': critic_params}, obs_t, a_exec_flat_t)  # (num_qs, 1)
    q_pred_heads = q_pred_heads[:, 0]  # (num_qs,)

    td_errors = q_pred_heads - target_q

    return td_errors, q_pred_heads, target_q


# ---------------------------------------------------------------------------
# V_soft estimation (Monte-Carlo, for Q01)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=(
    'actor_apply_fn', 'critic_apply_fn', 'temp_apply_fn',
    'K', 'reduction', 'query_frequency', 'predict_a_exec',
))
def estimate_v_soft(
    rng,
    actor_apply_fn, actor_params, actor_batch_stats,
    critic_apply_fn, target_critic_params,
    temp_apply_fn, temp_params,
    obs_dict,
    residual_alpha, query_frequency,
    K=10, reduction='min', predict_a_exec=False,
):
    """Estimate soft state value V_soft(s) via K Monte-Carlo samples.

    V_soft(s) = E_{a~pi}[Q_reduce(s,a) - alpha * log pi(a|s)]

    Returns:
        v_soft: scalar estimated soft value.
    """
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, obs_dict, training=False, mutable=False)

    alpha_temp = temp_apply_fn({'params': temp_params})

    base_raw = obs_dict['base_action']
    base = jnp.squeeze(base_raw, axis=-1)  # (1, chunk_len, action_dim)

    def sample_and_eval(rng_k):
        action_sampled, log_prob = dist.sample_and_log_prob(seed=rng_k)
        action_chunked = action_sampled.reshape(1, query_frequency, -1)
        if predict_a_exec:
            a_exec = jnp.clip(action_chunked, -1.0, 1.0)
        else:
            a_exec = jnp.clip(
                base[:, :query_frequency, :] + residual_alpha * action_chunked,
                -1.0, 1.0
            )
        a_exec_flat = a_exec.reshape(1, -1)
        qs = critic_apply_fn({'params': target_critic_params}, obs_dict, a_exec_flat)
        if reduction == 'min':
            q = qs.min(axis=0)[0]
        else:
            q = qs.mean(axis=0)[0]
        return q - alpha_temp * log_prob[0]

    keys = jax.random.split(rng, K)
    v_samples = jax.vmap(sample_and_eval)(keys)  # (K,)
    return v_samples.mean()


# ---------------------------------------------------------------------------
# Gradient ascent on Q (for UMAP landscape plots L02/L03)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=('critic_apply_fn', 'num_steps', 'reduction'))
def gradient_ascent_on_q(critic_params, critic_apply_fn, obs_dict,
                         a_init_flat, step_size, num_steps=20,
                         reduction='mean'):
    """Run gradient ascent on Q w.r.t. action.

    Args:
        a_init_flat: (1, action_dim_flat) starting action.
        step_size: Learning rate (eta).
        num_steps: Number of ascent steps S.

    Returns:
        path_actions: (S+1, action_dim_flat) actions along the path.
        path_q_values: (S+1,) Q values along the path.
    """
    def q_fn(a):
        qs = critic_apply_fn({'params': critic_params}, obs_dict, a)
        if reduction == 'min':
            return qs.min(axis=0)[0]
        return qs.mean(axis=0)[0]

    def body_fn(carry, _):
        a = carry
        g = jax.grad(lambda a_: q_fn(a_))(a)
        g_norm = jnp.linalg.norm(g) + 1e-8
        a_new = jnp.clip(a + step_size * g / g_norm, -1.0, 1.0)
        q_val = q_fn(a_new)
        return a_new, (a_new[0], q_val)

    q_init = q_fn(a_init_flat)
    _, (path_actions, path_q_values) = jax.lax.scan(
        body_fn, a_init_flat, None, length=num_steps
    )
    # Prepend initial point
    path_actions = jnp.concatenate([a_init_flat, path_actions], axis=0)
    path_q_values = jnp.concatenate([q_init[None], path_q_values], axis=0)
    return path_actions, path_q_values


# ---------------------------------------------------------------------------
# 1D gradient line probing (for L04/L05)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=('critic_apply_fn', 'num_points', 'reduction'))
def probe_q_along_gradient_line(critic_params, critic_apply_fn, obs_dict,
                                a0_flat, num_points=21, line_range=0.3,
                                reduction='mean'):
    """Sample Q values along the gradient direction at a0.

    Args:
        a0_flat: (1, action_dim_flat) anchor action.
        num_points: Number of sample points along the line.
        line_range: Half-length L of the line in action space.

    Returns:
        lambdas: (num_points,) lambda values.
        q_values: (num_points,) Q values at each point.
        line_actions: (num_points, action_dim_flat) actions along the line.
    """
    def q_fn(a):
        qs = critic_apply_fn({'params': critic_params}, obs_dict, a)
        if reduction == 'min':
            return qs.min(axis=0)[0]
        return qs.mean(axis=0)[0]

    g0 = jax.grad(lambda a: q_fn(a))(a0_flat)
    g0_norm = jnp.linalg.norm(g0) + 1e-8
    g0_dir = g0 / g0_norm  # normalized gradient direction

    lambdas = jnp.linspace(-line_range, line_range, num_points)

    def eval_at_lambda(lam):
        a = jnp.clip(a0_flat + lam * g0_dir, -1.0, 1.0)
        return q_fn(a), a[0]

    q_values, line_actions = jax.vmap(eval_at_lambda)(lambdas)
    return lambdas, q_values, line_actions


# ---------------------------------------------------------------------------
# Actor distribution parameter extraction (for E02)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames='actor_apply_fn')
def get_actor_distribution_params(actor_apply_fn, actor_params, actor_batch_stats,
                                  obs_dict):
    """Extract pre-tanh mean and std from the actor distribution.

    Returns:
        mean: (action_dim_flat,) pre-tanh mean (distribution.distribution._loc).
        std: (action_dim_flat,) pre-tanh std (distribution.distribution._scale_diag).
    """
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, obs_dict, training=False, mutable=False)
    # TanhMultivariateNormalDiag wraps a MultivariateNormalDiag
    # dist.distribution is the base Gaussian
    base_dist = dist.distribution
    mean = base_dist.loc[0]  # (action_dim_flat,) — remove batch dim
    std = base_dist.scale_diag[0]  # (action_dim_flat,)
    return mean, std


# ---------------------------------------------------------------------------
# Sample K actions from residual actor (for Q09/Q10)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=(
    'actor_apply_fn', 'K', 'query_frequency', 'predict_a_exec',
))
def sample_k_exec_actions(
    rng,
    actor_apply_fn, actor_params, actor_batch_stats,
    obs_dict, base_action_flat, residual_alpha,
    query_frequency, K=20, predict_a_exec=False,
):
    """Sample K executed actions from the residual actor at a given state.

    Args:
        obs_dict: Observation dict (batch dim 1).
        base_action_flat: (1, query_freq * action_dim) flattened base action.
        residual_alpha: Scaling factor for residual.
        query_frequency: Steps per query.
        K: Number of samples.
        predict_a_exec: Whether actor directly predicts a_exec.

    Returns:
        a_exec_flat: (K, action_dim_flat) sampled executed actions.
    """
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, obs_dict, training=False, mutable=False)

    action_dim_flat = base_action_flat.shape[-1]
    action_dim = action_dim_flat // query_frequency

    def sample_one(rng_k):
        action_sampled, _ = dist.sample_and_log_prob(seed=rng_k)
        action_chunked = action_sampled.reshape(1, query_frequency, action_dim)
        base_chunked = base_action_flat.reshape(1, query_frequency, action_dim)
        if predict_a_exec:
            a_exec = jnp.clip(action_chunked, -1.0, 1.0)
        else:
            a_exec = jnp.clip(
                base_chunked + residual_alpha * action_chunked, -1.0, 1.0
            )
        return a_exec.reshape(-1)  # (action_dim_flat,)

    keys = jax.random.split(rng, K)
    return jax.vmap(sample_one)(keys)  # (K, action_dim_flat)


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------

def compose_a_exec_np(base_action, delta_or_aexec, residual_alpha,
                      query_frequency, predict_a_exec):
    """Compose executed action from base + residual (numpy).

    Args:
        base_action: (chunk_len, action_dim) or with extra dims to squeeze.
        delta_or_aexec: (query_freq, action_dim) raw actor output.

    Returns:
        a_exec: (query_freq, action_dim) executed action.
        delta: (query_freq, action_dim) residual delta.
        base_sliced: (query_freq, action_dim) base action sliced to query_freq.
    """
    base = np.squeeze(base_action)
    if base.ndim == 2:
        base_sliced = base[:query_frequency]
    else:
        base_sliced = base

    if predict_a_exec:
        a_exec = np.clip(delta_or_aexec, -1.0, 1.0)
        delta = a_exec - base_sliced
    else:
        delta = delta_or_aexec
        a_exec = np.clip(base_sliced + residual_alpha * delta, -1.0, 1.0)

    return a_exec, delta, base_sliced


def compute_mc_returns(rewards, gamma, query_frequency):
    """Compute discounted Monte-Carlo returns from per-step rewards.

    The rewards are per env step; returns are computed per query step.

    Args:
        rewards: (T_steps,) per env-step rewards.
        gamma: Discount factor.
        query_frequency: Steps per query.

    Returns:
        returns: (T_query,) MC returns at each query step.
    """
    T = len(rewards)
    # Aggregate rewards per query step
    n_query = (T + query_frequency - 1) // query_frequency
    query_rewards = np.zeros(n_query)
    for q in range(n_query):
        start = q * query_frequency
        end = min(start + query_frequency, T)
        for k, t in enumerate(range(start, end)):
            query_rewards[q] += (gamma ** k) * rewards[t]

    # Compute returns from the end
    returns = np.zeros(n_query)
    running = 0.0
    discount_per_query = gamma ** query_frequency
    for q in reversed(range(n_query)):
        running = query_rewards[q] + discount_per_query * running
        returns[q] = running

    return returns
