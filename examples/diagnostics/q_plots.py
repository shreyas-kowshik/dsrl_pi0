"""Q-function diagnostic plots Q01-Q10.

Each function generates a list of matplotlib figures (frames for animation)
and saves them as .mp4 via the video_writer module.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from examples.diagnostics.jax_compute import (
    compute_q_values_all_heads,
    compute_q_reduced,
    compute_td_error_single_step,
    estimate_v_soft,
    compute_q_action_gradient_per_head,
    compute_mc_returns,
    sample_k_exec_actions,
)
from examples.diagnostics.video_writer import save_figures_as_mp4, save_static_plot_as_mp4


def _get_keyframe_strip(images, num_frames=4):
    """Select evenly-spaced keyframes and concatenate horizontally."""
    if not images or len(images) == 0:
        return None
    n = len(images)
    indices = np.linspace(0, n - 1, num_frames, dtype=int)
    frames = [images[i] for i in indices]
    # Handle different image shapes
    if frames[0].ndim == 3:
        return np.concatenate(frames, axis=1)
    return None


def _get_frame_at_step(images, query_step, query_freq):
    """Get the camera frame corresponding to a given query step."""
    if not images or len(images) == 0:
        return None
    env_step = max(0, min(query_step * query_freq, len(images) - 1))
    img = images[env_step]
    if img.ndim == 3:
        return img
    return None


def _prepare_obs_for_critic(obs_dict):
    """Ensure obs_dict has batch dim for critic/actor. Returns a copy."""
    out = {}
    for k, v in obs_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = jnp.asarray(v)
        else:
            out[k] = v
    return out


def _get_a_exec_flat(traj, t, query_frequency):
    """Get flattened executed action at query step t."""
    return jnp.asarray(traj.a_exec[t].reshape(1, -1))


def _get_a_base_flat(traj, t, query_frequency):
    """Get flattened base action at query step t."""
    base = np.squeeze(traj.base_actions[t])  # (chunk_len, action_dim)
    return jnp.asarray(base[:query_frequency].reshape(1, -1))


# ============================================================================
# Q01 — Multi-step consistency curve
# ============================================================================

def plot_q01_multistep_consistency(all_traj_data, agent_internals, variant, save_dir):
    """Q01: Multi-step consistency curve (Q vs n-step bootstrapped return).

    Generates animated .mp4 where each frame adds a new n value to the curve.
    Saved to {save_dir}/Q01_multistep_consistency.mp4
    """
    n_list = [1, 2, 4, 8, 16, 32]
    gamma = variant.discount
    qf = variant.query_freq
    predict_a_exec = variant.get('predict_a_exec', False)

    ai = agent_internals
    rng = jax.random.PRNGKey(42)

    # Collect all (Q_pred, G_t^n) pairs for each n
    mae_per_n = []
    bias_per_n = []

    for n in n_list:
        q_preds = []
        g_targets = []

        for traj in all_traj_data:
            T_query = len(traj.obs_dicts) - 1  # last obs is for bootstrapping
            for t in range(T_query):
                obs_t = _prepare_obs_for_critic(traj.obs_dicts[t])
                a_exec_flat = _get_a_exec_flat(traj, t, qf)

                # Q prediction at (s_t, a_exec_t)
                q_pred = float(compute_q_reduced(
                    ai['critic_params'], ai['critic_apply_fn'],
                    obs_t, a_exec_flat, ai['critic_reduction']).squeeze())
                q_preds.append(q_pred)

                # n-step return
                t_end = min(t + n, T_query)
                actual_n = t_end - t

                # Sum discounted rewards over n steps
                R_n = 0.0
                terminated_before = False
                for k in range(actual_n):
                    step_idx = t + k
                    # Map query step to env reward (use query_frequency)
                    env_start = step_idx * qf
                    env_end = min(env_start + qf, len(traj.rewards))
                    for s, env_t in enumerate(range(env_start, env_end)):
                        R_n += (gamma ** (k * qf + s)) * traj.rewards[env_t]

                    if traj.terminated and env_end >= len(traj.rewards):
                        terminated_before = True
                        break

                # Bootstrap at s_{t+n} if not terminated
                bootstrap = 0.0
                if not terminated_before and t_end < len(traj.obs_dicts):
                    rng, key = jax.random.split(rng)
                    obs_tn = _prepare_obs_for_critic(traj.obs_dicts[t_end])
                    v_soft = float(estimate_v_soft(
                        key,
                        ai['actor_apply_fn'], ai['actor_params'], ai['actor_batch_stats'],
                        ai['critic_apply_fn'], ai['target_critic_params'],
                        ai['temp_apply_fn'], ai['temp_params'],
                        obs_tn,
                        ai['residual_alpha'], qf,
                        K=10, reduction=ai['critic_reduction'],
                        predict_a_exec=predict_a_exec,
                    ))
                    bootstrap = (gamma ** (actual_n * qf)) * v_soft

                G_n = R_n + bootstrap
                g_targets.append(G_n)

        q_preds = np.array(q_preds)
        g_targets = np.array(g_targets)
        mae_per_n.append(np.mean(np.abs(q_preds - g_targets)))
        bias_per_n.append(np.mean(q_preds - g_targets))

    # Animate: each frame adds next n
    figures = []
    for frame_idx in range(1, len(n_list) + 1):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ns = n_list[:frame_idx]
        maes = mae_per_n[:frame_idx]
        biases = bias_per_n[:frame_idx]

        ax1.plot(ns, maes, 'o-', color='steelblue', linewidth=2)
        ax1.set_xscale('log', base=2)
        ax1.set_xlabel('n (bootstrap horizon)')
        ax1.set_ylabel('MAE')
        ax1.set_title('Q01: Multi-step Consistency (MAE)')
        ax1.grid(True, alpha=0.3)

        ax2.plot(ns, biases, 's-', color='coral', linewidth=2)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xscale('log', base=2)
        ax2.set_xlabel('n (bootstrap horizon)')
        ax2.set_ylabel('Bias (Q_pred - G_n)')
        ax2.set_title('Q01: Multi-step Consistency (Bias)')
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Q01: Multi-step Consistency Curve', fontsize=14)
        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'Q01_multistep_consistency.mp4')
    save_figures_as_mp4(figures, filepath, fps=1)


# ============================================================================
# Q02 — Q(s,a) vs Q_targ(s,a) scatter
# ============================================================================

def plot_q02_q_vs_qtarg(all_traj_data, agent_internals, variant, save_dir):
    """Q02: Q_online vs Q_target scatter, animated by trajectory.

    Saved to {save_dir}/Q02_q_vs_qtarg.mp4
    """
    ai = agent_internals
    qf = variant.query_freq

    all_q_online = []
    all_q_target = []
    traj_boundaries = [0]

    for traj in all_traj_data:
        T_query = len(traj.obs_dicts) - 1
        for t in range(T_query):
            obs_t = _prepare_obs_for_critic(traj.obs_dicts[t])
            a_exec_flat = _get_a_exec_flat(traj, t, qf)

            q_online = float(compute_q_reduced(
                ai['critic_params'], ai['critic_apply_fn'],
                obs_t, a_exec_flat, ai['critic_reduction']).squeeze())
            q_target = float(compute_q_reduced(
                ai['target_critic_params'], ai['critic_apply_fn'],
                obs_t, a_exec_flat, ai['critic_reduction']).squeeze())

            all_q_online.append(q_online)
            all_q_target.append(q_target)
        traj_boundaries.append(len(all_q_online))

    all_q_online = np.array(all_q_online)
    all_q_target = np.array(all_q_target)

    # Animate: add one trajectory at a time
    figures = []
    for traj_idx in range(len(all_traj_data)):
        end = traj_boundaries[traj_idx + 1]
        qo = all_q_online[:end]
        qt = all_q_target[:end]

        fig, ax = plt.subplots(figsize=(7, 7))
        vmin = min(qo.min(), qt.min())
        vmax = max(qo.max(), qt.max())
        margin = (vmax - vmin) * 0.1 + 1e-6
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'k--', alpha=0.4, label='y=x')
        ax.scatter(qt, qo, alpha=0.5, s=10, c='steelblue')

        # Linear fit
        if len(qt) > 2:
            coeffs = np.polyfit(qt, qo, 1)
            corr = np.corrcoef(qt, qo)[0, 1]
            ax.set_title(f'Q02: Q_online vs Q_target (corr={corr:.3f}, slope={coeffs[0]:.3f})')
        else:
            ax.set_title('Q02: Q_online vs Q_target')

        ax.set_xlabel('Q_target')
        ax.set_ylabel('Q_online')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'Q02_q_vs_qtarg.mp4')
    save_figures_as_mp4(figures, filepath, fps=1)


# ============================================================================
# Q03 — TD-error across time within a single trajectory
# ============================================================================

def plot_q03_td_error_trajectory(traj, traj_idx, agent_internals, variant, save_dir):
    """Q03: TD-error over time for a single trajectory, animated.

    Saved to {save_dir}/Q03_td_error.mp4
    """
    ai = agent_internals
    qf = variant.query_freq
    gamma = variant.discount
    predict_a_exec = variant.get('predict_a_exec', False)

    T_query = len(traj.obs_dicts) - 1
    rng = jax.random.PRNGKey(123)

    td_means = []
    td_stds = []
    all_td_per_head = []

    discount_per_query = gamma ** qf

    for t in range(T_query):
        obs_t = _prepare_obs_for_critic(traj.obs_dicts[t])
        a_exec_flat = _get_a_exec_flat(traj, t, qf)

        # Reward for this query step: sum of env-step rewards
        env_start = t * qf
        env_end = min(env_start + qf, len(traj.rewards))
        reward_t = 0.0
        for k, env_t in enumerate(range(env_start, env_end)):
            reward_t += (gamma ** k) * traj.rewards[env_t]

        # Mask: 0 if terminal within this query step
        mask_t = 1.0
        if traj.terminated and env_end >= len(traj.rewards):
            mask_t = 0.0

        if t + 1 < len(traj.obs_dicts):
            obs_tp1 = _prepare_obs_for_critic(traj.obs_dicts[t + 1])
        else:
            obs_tp1 = obs_t  # fallback, masked anyway

        rng, key = jax.random.split(rng)
        td_errors, q_pred_heads, target_q = compute_td_error_single_step(
            key,
            ai['actor_apply_fn'], ai['actor_params'], ai['actor_batch_stats'],
            ai['critic_apply_fn'], ai['critic_params'],
            ai['target_critic_params'],
            ai['temp_apply_fn'], ai['temp_params'],
            obs_t, a_exec_flat,
            reward_t, obs_tp1, mask_t, discount_per_query,
            ai['residual_alpha'], qf,
            predict_a_exec=predict_a_exec,
        )

        td_np = np.array(td_errors)
        all_td_per_head.append(td_np)
        td_means.append(td_np.mean())
        td_stds.append(td_np.std())

    td_means = np.array(td_means)
    td_stds = np.array(td_stds)

    # Animate: progressive reveal
    figures = []
    step_interval = max(1, T_query // 20)  # ~20 frames

    for end_t in range(step_interval, T_query + 1, step_interval):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                                 gridspec_kw={'height_ratios': [1, 3]})
        cur_frame = _get_frame_at_step(traj.images, end_t - 1, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].set_title(f'Trajectory {traj_idx}, t={end_t-1} (success={traj.is_success})')
        axes[0].axis('off')

        t_range = np.arange(end_t)
        axes[1].plot(t_range, td_means[:end_t], color='steelblue', linewidth=1.5)
        axes[1].fill_between(t_range,
                             td_means[:end_t] - td_stds[:end_t],
                             td_means[:end_t] + td_stds[:end_t],
                             alpha=0.25, color='steelblue')
        axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Query step t')
        axes[1].set_ylabel('TD error')
        axes[1].set_title('Q03: TD-error across time')
        axes[1].set_xlim(0, T_query)
        axes[1].grid(True, alpha=0.3)

        if traj.terminated:
            axes[1].axvline(T_query - 1, color='red', linestyle=':', alpha=0.7,
                            label='terminal')
            axes[1].legend()

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'Q03_td_error.mp4')
    save_figures_as_mp4(figures, filepath, fps=2)

    return all_td_per_head  # return for Q08 reuse


# ============================================================================
# Q04 — Q-values of base action across trajectory
# ============================================================================

def plot_q04_q_base_trajectory(traj, traj_idx, agent_internals, variant, save_dir):
    """Q04: Q(s, a_base) over time with MC returns, animated.

    3-row panel: returns, Q(base) mean±std, disagreement.
    Saved to {save_dir}/Q04_q_base.mp4
    """
    ai = agent_internals
    qf = variant.query_freq
    gamma = variant.discount

    T_query = len(traj.obs_dicts) - 1

    mc_returns = compute_mc_returns(traj.rewards, gamma, qf)[:T_query]
    q_means = []
    q_stds = []

    for t in range(T_query):
        obs_t = _prepare_obs_for_critic(traj.obs_dicts[t])
        a_base_flat = _get_a_base_flat(traj, t, qf)

        qs = np.array(compute_q_values_all_heads(
            ai['critic_params'], ai['critic_apply_fn'], obs_t, a_base_flat))
        q_means.append(qs[:, 0].mean())
        q_stds.append(qs[:, 0].std())

    q_means = np.array(q_means)
    q_stds = np.array(q_stds)

    figures = []
    step_interval = max(1, T_query // 20)

    for end_t in range(step_interval, T_query + 1, step_interval):
        fig, axes = plt.subplots(4, 1, figsize=(10, 10),
                                 gridspec_kw={'height_ratios': [1, 2, 2, 2]})
        cur_frame = _get_frame_at_step(traj.images, end_t - 1, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].axis('off')
        axes[0].set_title(f'Traj {traj_idx}, t={end_t-1} (success={traj.is_success})')

        t_range = np.arange(end_t)

        # Row 1: MC returns
        axes[1].plot(t_range, mc_returns[:end_t], color='green', linewidth=1.5)
        axes[1].set_ylabel('MC Return')
        axes[1].set_xlim(0, T_query)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Return over time')

        # Row 2: Q(base) mean ± std
        axes[2].plot(t_range, q_means[:end_t], color='steelblue', linewidth=1.5)
        axes[2].fill_between(t_range,
                             q_means[:end_t] - q_stds[:end_t],
                             q_means[:end_t] + q_stds[:end_t],
                             alpha=0.25, color='steelblue')
        axes[2].set_ylabel('Q(s, a_base)')
        axes[2].set_xlim(0, T_query)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Q04: Q(base action) over time')

        # Row 3: Disagreement
        axes[3].plot(t_range, q_stds[:end_t], color='coral', linewidth=1.5)
        axes[3].set_xlabel('Query step t')
        axes[3].set_ylabel('Q std (disagreement)')
        axes[3].set_xlim(0, T_query)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_title('Ensemble disagreement')

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'Q04_q_base.mp4')
    save_figures_as_mp4(figures, filepath, fps=2)


# ============================================================================
# Q05 — Q-values of executed action across trajectory
# ============================================================================

def plot_q05_q_exec_trajectory(traj, traj_idx, agent_internals, variant, save_dir):
    """Q05: Q(s, a_exec) over time with Q(base) overlay, animated.

    Saved to {save_dir}/Q05_q_exec.mp4
    """
    ai = agent_internals
    qf = variant.query_freq
    gamma = variant.discount

    T_query = len(traj.obs_dicts) - 1

    mc_returns = compute_mc_returns(traj.rewards, gamma, qf)[:T_query]
    q_exec_means = []
    q_exec_stds = []
    q_base_means = []

    for t in range(T_query):
        obs_t = _prepare_obs_for_critic(traj.obs_dicts[t])
        a_exec_flat = _get_a_exec_flat(traj, t, qf)
        a_base_flat = _get_a_base_flat(traj, t, qf)

        qs_exec = np.array(compute_q_values_all_heads(
            ai['critic_params'], ai['critic_apply_fn'], obs_t, a_exec_flat))
        qs_base = np.array(compute_q_values_all_heads(
            ai['critic_params'], ai['critic_apply_fn'], obs_t, a_base_flat))

        q_exec_means.append(qs_exec[:, 0].mean())
        q_exec_stds.append(qs_exec[:, 0].std())
        q_base_means.append(qs_base[:, 0].mean())

    q_exec_means = np.array(q_exec_means)
    q_exec_stds = np.array(q_exec_stds)
    q_base_means = np.array(q_base_means)

    figures = []
    step_interval = max(1, T_query // 20)

    for end_t in range(step_interval, T_query + 1, step_interval):
        fig, axes = plt.subplots(4, 1, figsize=(10, 10),
                                 gridspec_kw={'height_ratios': [1, 2, 2, 2]})
        cur_frame = _get_frame_at_step(traj.images, end_t - 1, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].axis('off')
        axes[0].set_title(f'Traj {traj_idx}, t={end_t-1} (success={traj.is_success})')

        t_range = np.arange(end_t)

        # Row 1: MC returns
        axes[1].plot(t_range, mc_returns[:end_t], color='green', linewidth=1.5)
        axes[1].set_ylabel('MC Return')
        axes[1].set_xlim(0, T_query)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Return over time')

        # Row 2: Q(exec) mean ± std with Q(base) overlay
        axes[2].plot(t_range, q_exec_means[:end_t], color='steelblue',
                     linewidth=2, label='Q(a_exec)')
        axes[2].fill_between(t_range,
                             q_exec_means[:end_t] - q_exec_stds[:end_t],
                             q_exec_means[:end_t] + q_exec_stds[:end_t],
                             alpha=0.25, color='steelblue')
        axes[2].plot(t_range, q_base_means[:end_t], color='gray',
                     linewidth=1, linestyle='--', alpha=0.7, label='Q(a_base)')
        axes[2].set_ylabel('Q value')
        axes[2].set_xlim(0, T_query)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Q05: Q(exec action) over time')

        # Row 3: Disagreement
        axes[3].plot(t_range, q_exec_stds[:end_t], color='coral', linewidth=1.5)
        axes[3].set_xlabel('Query step t')
        axes[3].set_ylabel('Q std')
        axes[3].set_xlim(0, T_query)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_title('Ensemble disagreement')

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'Q05_q_exec.mp4')
    save_figures_as_mp4(figures, filepath, fps=2)


# ============================================================================
# Q06 — Gradient norm of Q w.r.t. executed action
# ============================================================================

def plot_q06_grad_norm_exec(traj, traj_idx, agent_internals, variant, save_dir):
    """Q06: ||dQ/da||_2 at executed action, mean±std across heads, animated.

    Saved to {save_dir}/Q06_grad_norm_exec.mp4
    """
    ai = agent_internals
    qf = variant.query_freq
    num_qs = ai.get('num_qs', 2)

    T_query = len(traj.obs_dicts) - 1
    gnorm_means = []
    gnorm_stds = []

    for t in range(T_query):
        obs_t = _prepare_obs_for_critic(traj.obs_dicts[t])
        a_exec_flat = _get_a_exec_flat(traj, t, qf)

        grad_norms, _ = compute_q_action_gradient_per_head(
            ai['critic_params'], ai['critic_apply_fn'],
            obs_t, a_exec_flat, num_qs=num_qs)
        gn = np.array(grad_norms)
        gnorm_means.append(gn.mean())
        gnorm_stds.append(gn.std())

    gnorm_means = np.array(gnorm_means)
    gnorm_stds = np.array(gnorm_stds)

    figures = []
    step_interval = max(1, T_query // 20)

    for end_t in range(step_interval, T_query + 1, step_interval):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                                 gridspec_kw={'height_ratios': [1, 3]})
        cur_frame = _get_frame_at_step(traj.images, end_t - 1, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].axis('off')
        axes[0].set_title(f'Traj {traj_idx}, t={end_t-1} (success={traj.is_success})')

        t_range = np.arange(end_t)
        axes[1].plot(t_range, gnorm_means[:end_t], color='purple', linewidth=1.5)
        axes[1].fill_between(t_range,
                             gnorm_means[:end_t] - gnorm_stds[:end_t],
                             gnorm_means[:end_t] + gnorm_stds[:end_t],
                             alpha=0.25, color='purple')
        axes[1].set_xlabel('Query step t')
        axes[1].set_ylabel('||dQ/da||_2')
        axes[1].set_title('Q06: Gradient norm w.r.t. executed action')
        axes[1].set_xlim(0, T_query)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'Q06_grad_norm_exec.mp4')
    save_figures_as_mp4(figures, filepath, fps=2)


# ============================================================================
# Q07 — Gradient norm of Q w.r.t. base action
# ============================================================================

def plot_q07_grad_norm_base(traj, traj_idx, agent_internals, variant, save_dir):
    """Q07: ||dQ/da||_2 at base action, mean±std across heads, animated.

    Saved to {save_dir}/Q07_grad_norm_base.mp4
    """
    ai = agent_internals
    qf = variant.query_freq
    num_qs = ai.get('num_qs', 2)

    T_query = len(traj.obs_dicts) - 1
    gnorm_means = []
    gnorm_stds = []

    for t in range(T_query):
        obs_t = _prepare_obs_for_critic(traj.obs_dicts[t])
        a_base_flat = _get_a_base_flat(traj, t, qf)

        grad_norms, _ = compute_q_action_gradient_per_head(
            ai['critic_params'], ai['critic_apply_fn'],
            obs_t, a_base_flat, num_qs=num_qs)
        gn = np.array(grad_norms)
        gnorm_means.append(gn.mean())
        gnorm_stds.append(gn.std())

    gnorm_means = np.array(gnorm_means)
    gnorm_stds = np.array(gnorm_stds)

    figures = []
    step_interval = max(1, T_query // 20)

    for end_t in range(step_interval, T_query + 1, step_interval):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                                 gridspec_kw={'height_ratios': [1, 3]})
        cur_frame = _get_frame_at_step(traj.images, end_t - 1, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].axis('off')
        axes[0].set_title(f'Traj {traj_idx}, t={end_t-1} (success={traj.is_success})')

        t_range = np.arange(end_t)
        axes[1].plot(t_range, gnorm_means[:end_t], color='darkorange', linewidth=1.5)
        axes[1].fill_between(t_range,
                             gnorm_means[:end_t] - gnorm_stds[:end_t],
                             gnorm_means[:end_t] + gnorm_stds[:end_t],
                             alpha=0.25, color='darkorange')
        axes[1].set_xlabel('Query step t')
        axes[1].set_ylabel('||dQ/da||_2')
        axes[1].set_title('Q07: Gradient norm w.r.t. base action')
        axes[1].set_xlim(0, T_query)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'Q07_grad_norm_base.mp4')
    save_figures_as_mp4(figures, filepath, fps=2)


# ============================================================================
# Q08 — Histogram of TD errors across all evaluation trajectories
# ============================================================================

def plot_q08_td_histogram(all_td_errors_per_traj, all_traj_data, save_dir):
    """Q08: TD-error histogram, animated by adding trajectories.

    Args:
        all_td_errors_per_traj: List of lists. Each inner list contains per-head
            TD errors for each query step of a trajectory.
        all_traj_data: List of EvalTrajectoryData.

    Saved to {save_dir}/Q08_td_histogram.mp4
    """
    figures = []

    cumulative_td = []
    for traj_idx, traj_tds in enumerate(all_td_errors_per_traj):
        for td_per_head in traj_tds:
            val = float(np.nanmean(td_per_head))
            if np.isfinite(val):
                cumulative_td.append(val)

        if not cumulative_td:
            continue

        td_arr = np.array(cumulative_td)
        # Filter out any remaining NaN/Inf
        td_arr = td_arr[np.isfinite(td_arr)]
        if len(td_arr) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(td_arr, bins=50, color='steelblue', alpha=0.7, edgecolor='white')

        mean_td = td_arr.mean()
        median_td = np.median(td_arr)
        p95 = np.percentile(np.abs(td_arr), 95)

        ax.axvline(mean_td, color='red', linestyle='-', alpha=0.8, label=f'Mean={mean_td:.3f}')
        ax.axvline(median_td, color='orange', linestyle='--', alpha=0.8, label=f'Median={median_td:.3f}')
        ax.axvline(p95, color='green', linestyle=':', alpha=0.8, label=f'|TD| 95th={p95:.3f}')
        ax.axvline(-p95, color='green', linestyle=':', alpha=0.8)

        ax.set_xlabel('TD error')
        ax.set_ylabel('Count')
        ax.set_title(f'Q08: TD-error histogram ({traj_idx + 1}/{len(all_td_errors_per_traj)} trajectories)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'Q08_td_histogram.mp4')
    save_figures_as_mp4(figures, filepath, fps=1)


# ============================================================================
# Q09 — Variance of Q across sampled base actions
# ============================================================================

def plot_q09_q_variance_base(traj, traj_idx, agent_internals, variant, save_dir,
                             K=20, noise_std=0.1):
    """Q09: Var_a[Q_i(s, a_base)] per Q-head across K perturbed base actions.

    At each state, K base action variants are created by adding Gaussian noise
    to the stored base action. For each Q-head, variance of Q across K actions
    is computed. Plots mean ± std of per-head variances over the trajectory.

    Saved to {save_dir}/Q09_q_variance_base.mp4
    """
    ai = agent_internals
    qf = variant.query_freq

    T_query = len(traj.obs_dicts) - 1

    rng_np = np.random.RandomState(42)
    var_means = []  # mean of per-head variances at each t
    var_stds = []   # std of per-head variances at each t

    for t in range(T_query):
        obs_t = _prepare_obs_for_critic(traj.obs_dicts[t])
        a_base_flat = np.squeeze(traj.base_actions[t])[:qf].flatten()

        # Generate K perturbed base actions
        noise = rng_np.randn(K, *a_base_flat.shape) * noise_std
        a_base_samples = np.clip(a_base_flat[None] + noise, -1, 1)  # (K, action_dim_flat)
        a_base_jax = jnp.asarray(a_base_samples)

        # Evaluate all Q heads on K actions: (num_qs, K)
        qs = np.array(compute_q_values_all_heads(
            ai['critic_params'], ai['critic_apply_fn'],
            obs_t, a_base_jax))  # (num_qs, K)

        # Per-head variance across K actions
        per_head_var = qs.var(axis=1)  # (num_qs,)
        var_means.append(per_head_var.mean())
        var_stds.append(per_head_var.std())

    var_means = np.array(var_means)
    var_stds = np.array(var_stds)

    figures = []
    step_interval = max(1, T_query // 20)

    for end_t in range(step_interval, T_query + 1, step_interval):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                                 gridspec_kw={'height_ratios': [1, 3]})
        cur_frame = _get_frame_at_step(traj.images, end_t - 1, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].axis('off')
        axes[0].set_title(f'Traj {traj_idx}, t={end_t-1} (success={traj.is_success})')

        t_range = np.arange(end_t)
        axes[1].plot(t_range, var_means[:end_t], color='steelblue', linewidth=1.5,
                     label='Mean Var(Q)')
        axes[1].fill_between(t_range,
                             var_means[:end_t] - var_stds[:end_t],
                             var_means[:end_t] + var_stds[:end_t],
                             alpha=0.25, color='steelblue')
        axes[1].set_xlabel('Query step t')
        axes[1].set_ylabel('Var_a[Q_i(s, a_base)]')
        axes[1].set_title(f'Q09: Q variance across {K} sampled base actions')
        axes[1].set_xlim(0, T_query)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'Q09_q_variance_base.mp4')
    save_figures_as_mp4(figures, filepath, fps=2)


# ============================================================================
# Q10 — Variance of Q across sampled exec actions
# ============================================================================

def plot_q10_q_variance_exec(traj, traj_idx, agent_internals, variant, save_dir,
                             K=20):
    """Q10: Var_a[Q_i(s, a_exec)] per Q-head across K actor-sampled exec actions.

    At each state, K actions are sampled from the residual actor and composed
    with the base action to produce K executed actions. For each Q-head,
    variance of Q across K actions is computed. Plots mean ± std of per-head
    variances over the trajectory.

    Saved to {save_dir}/Q10_q_variance_exec.mp4
    """
    ai = agent_internals
    qf = variant.query_freq
    predict_a_exec = variant.get('predict_a_exec', False)

    T_query = len(traj.obs_dicts) - 1

    rng = jax.random.PRNGKey(42)
    var_means = []
    var_stds = []

    for t in range(T_query):
        obs_t = _prepare_obs_for_critic(traj.obs_dicts[t])
        a_base_flat = jnp.asarray(
            np.squeeze(traj.base_actions[t])[:qf].flatten()[None])  # (1, action_dim_flat)

        rng, key = jax.random.split(rng)
        a_exec_samples = sample_k_exec_actions(
            key,
            ai['actor_apply_fn'], ai['actor_params'], ai['actor_batch_stats'],
            obs_t, a_base_flat, ai['residual_alpha'],
            qf, K=K, predict_a_exec=predict_a_exec,
        )  # (K, action_dim_flat)

        # Evaluate all Q heads on K actions: (num_qs, K)
        qs = np.array(compute_q_values_all_heads(
            ai['critic_params'], ai['critic_apply_fn'],
            obs_t, a_exec_samples))  # (num_qs, K)

        # Per-head variance across K actions
        per_head_var = qs.var(axis=1)  # (num_qs,)
        var_means.append(per_head_var.mean())
        var_stds.append(per_head_var.std())

    var_means = np.array(var_means)
    var_stds = np.array(var_stds)

    figures = []
    step_interval = max(1, T_query // 20)

    for end_t in range(step_interval, T_query + 1, step_interval):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                                 gridspec_kw={'height_ratios': [1, 3]})
        cur_frame = _get_frame_at_step(traj.images, end_t - 1, qf)
        if cur_frame is not None:
            axes[0].imshow(cur_frame)
        axes[0].axis('off')
        axes[0].set_title(f'Traj {traj_idx}, t={end_t-1} (success={traj.is_success})')

        t_range = np.arange(end_t)
        axes[1].plot(t_range, var_means[:end_t], color='coral', linewidth=1.5,
                     label='Mean Var(Q)')
        axes[1].fill_between(t_range,
                             var_means[:end_t] - var_stds[:end_t],
                             var_means[:end_t] + var_stds[:end_t],
                             alpha=0.25, color='coral')
        axes[1].set_xlabel('Query step t')
        axes[1].set_ylabel('Var_a[Q_i(s, a_exec)]')
        axes[1].set_title(f'Q10: Q variance across {K} sampled exec actions')
        axes[1].set_xlim(0, T_query)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    filepath = os.path.join(save_dir, 'Q10_q_variance_exec.mp4')
    save_figures_as_mp4(figures, filepath, fps=2)
